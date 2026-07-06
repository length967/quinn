//! haul patch: BBRv2 congestion control over the per-ack rate-sample plumbing.
//!
//! This is the in-vendor successor to the bake-off experiment
//! `bench/bakeoff/src/bbr2full.rs` (kept there as the historical record). That
//! port was faithful to s2n-quic's BBRv2 *except* for five approximations
//! forced by quinn's batched `Controller` callbacks, and it failed its
//! fairness/throughput gates for exactly those reasons (see
//! bench/bakeoff/BBR2-PORT-NOTES.md). The connection now stamps delivery-rate
//! state into every sent packet (draft-cheng-iccrg-delivery-rate-estimation)
//! and delivers per-ack [`RateSample`]s and per-loss [`LossSample`]s, so the
//! five approximations are replaced with the real data model s2n's algorithm
//! was written against:
//!
//!   1. Loss bursts are counted per discrete loss event
//!      (`LossSample::new_loss_burst`, packet-number contiguity — s2n
//!      recovery/manager.rs), not per ack batch that happened to carry loss.
//!   2. `inflight_latest` is the loss-round max of per-ack `rs.delivered`
//!      (s2n congestion::State), not the whole round's delivered volume.
//!   3. `bw_probe_wait` is randomized 2..3 s (s2n `pick_probe_wait`) so
//!      competing flows' probe cycles desynchronize; the reno-coexistence
//!      round bound is ported with it.
//!   4. `adapt_upper_bounds` reads the rate sample's true `tx_in_flight`
//!      (in flight at the acked packet's send), not the batch's lagging
//!      post-ack `in_flight`.
//!   5. `probe_inflight_hi_upward` is driven by the acked bytes of real
//!      delivery samples, and the per-loss path reacts through
//!      `inflight_hi_from_lost_packet` (draft 4.5.6.2).
//!
//! Port provenance: state machine and bound formulas ported by reference from
//! s2n-quic (Apache-2.0, Amazon), commit 5eda4ad1bef555799ffce2eedb3567bcbaa73d74,
//! files quic/s2n-quic-core/src/recovery/bbr.rs and
//! bbr/{probe_bw,data_volume,congestion,round}.rs. The BBRv1 skeleton
//! (STARTUP/DRAIN/PROBE_RTT, pacing, cwnd growth, recovery window, ack
//! aggregation, min/max filter) is reproduced from this crate's `Bbr`, as the
//! bake-off port did.
//!
//! Documented deviations (not tuning):
//!   * ECN paths are not ported (no ECN-capable test path).
//!   * s2n's `AckPhase` bookkeeping (which advances the max-bw filter window
//!     between probes) is not ported; the max-bw filter stays round-windowed
//!     as in this crate's BBRv1.
//!   * s2n's `bw_lo`/`bw_hi` data-rate bounds are not ported (quinn's trait
//!     paces off `window()`, so the inflight bounds carry the yielding);
//!     this matches the bake-off port's scope.
//!   * `window()` clamps to min(cwnd, inflight_hi, inflight_lo) in all modes
//!     (the bake-off port's clamp); the draft's CRUISE-phase headroom cap is
//!     not applied.

use std::any::Any;
use std::sync::Arc;

use rand::{Rng, SeedableRng};

use super::{
    BASE_DATAGRAM_SIZE, Controller, ControllerFactory, ControllerMetrics, LossSample, RateSample,
};
use crate::connection::RttEstimator;
use crate::{Duration, Instant};

/// The four phases of the BBRv2 PROBE_BW cycle (s2n bbr/probe_bw.rs).
/// Legal transitions: Up -> Down -> (Cruise ->) Refill -> Up.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum CyclePhase {
    /// Pace below bw (0.9) to drain the queue and leave headroom for others.
    Down,
    /// Hold steady (1.0), staying below inflight_hi with headroom.
    Cruise,
    /// One round at 1.0 to refill the pipe before probing up.
    Refill,
    /// Probe upward (1.25) to look for more bandwidth.
    Up,
}

impl CyclePhase {
    /// Pacing gain per phase (s2n bbr/probe_bw.rs `CyclePhase::pacing_gain`).
    fn pacing_gain(&self) -> f32 {
        match self {
            Self::Down => 0.9,
            Self::Cruise | Self::Refill => 1.0,
            Self::Up => 1.25,
        }
    }
}

/// Delivered-bytes round counter (s2n bbr/round.rs `Counter`)
///
/// A round ends when a packet sent after the current round end is acked, i.e.
/// when an acked packet's `prior_delivered` reaches the connection's delivered
/// total recorded at the round start.
#[derive(Debug, Default, Clone, Copy)]
struct RoundCounter {
    next_round_delivered: u64,
    round_start: bool,
}

impl RoundCounter {
    /// s2n round.rs `on_ack`: called once per ack batch with the newest
    /// sample's `prior_delivered` and the current delivered total.
    fn on_ack(&mut self, prior_delivered: u64, delivered: u64) {
        if prior_delivered >= self.next_round_delivered {
            self.next_round_delivered = delivered;
            self.round_start = true;
        } else {
            self.round_start = false;
        }
    }

    /// s2n round.rs `set_round_end`: extend the round to the current
    /// delivered total.
    fn set_round_end(&mut self, delivered: u64) {
        self.next_round_delivered = self.next_round_delivered.max(delivered);
    }
}

/// Experimental! Use at your own risk.
///
/// BBRv2 over quinn's controller trait, extended with per-ack/per-loss rate
/// samples. See the module docs for provenance.
#[derive(Debug, Clone)]
pub struct Bbr2 {
    config: Arc<Bbr2Config>,
    current_mtu: u64,
    // --- delivery-rate model (fed by `on_ack_sample`) ---
    /// Windowed max filter over rate-sample delivery rates (bytes/s)
    max_bw_filter: MinMax,
    /// The newest rate sample (highest `prior_delivered`); persists across
    /// batches like s2n's `bw_estimator.rate_sample()`
    rate_sample: RateSample,
    /// The connection's delivered total (C.delivered) inferred from samples
    delivered_total: u64,
    /// Bytes acked by the samples of the current ack batch
    batch_acked: u64,
    /// Lifetime acked bytes (cwnd bootstrap, from `on_ack`)
    acked_bytes: u64,
    // --- BBRv1 skeleton (unchanged from this crate's `Bbr`) ---
    mode: Mode,
    loss_state: LossState,
    recovery_state: RecoveryState,
    recovery_window: u64,
    is_at_full_bandwidth: bool,
    pacing_gain: f32,
    high_gain: f32,
    drain_gain: f32,
    cwnd_gain: f32,
    high_cwnd_gain: f32,
    init_cwnd: u64,
    min_cwnd: u64,
    exit_probe_rtt_at: Option<Instant>,
    probe_rtt_last_started_at: Option<Instant>,
    min_rtt: Duration,
    exiting_quiescence: bool,
    pacing_rate: u64,
    max_acked_packet_number: u64,
    max_sent_packet_number: u64,
    end_recovery_at_packet_number: u64,
    cwnd: u64,
    current_round_trip_end_packet_number: u64,
    round_count: u64,
    bw_at_last_round: u64,
    round_wo_bw_gain: u64,
    ack_aggregation: AckAggregationState,
    // --- v2 upper/lower inflight bounds ---
    /// Inflight ceiling. `u64::MAX` = disengaged (no loss reaction yet).
    inflight_hi: u64,
    /// Short-term inflight lower bound. `u64::MAX` = disengaged. Collapses on
    /// a loss round (s2n data_volume.rs), reset on Refill.
    inflight_lo: u64,
    /// Bytes lost / delivered within the main round currently accumulating
    /// (drives the STARTUP/DRAIN `inflight_hi` ceiling, as in the bake-off
    /// port's v2 delta)
    round_lost: u64,
    round_delivered: u64,
    // --- PROBE_BW cycle (s2n bbr/probe_bw.rs) ---
    /// The current PROBE_BW cycle phase (only meaningful while mode == ProbeBw)
    cycle_phase: CyclePhase,
    /// Wall-clock stamp of the current cycle phase's start
    cycle_start: Option<Instant>,
    /// Randomized per-cycle wait before re-probing bandwidth
    /// (s2n `pick_probe_wait`, 2..3 s)
    bw_probe_wait: Duration,
    /// Whether the current samples come from an UP bandwidth probe. Gates the
    /// `on_inflight_too_high` reaction.
    bw_probe_samples: bool,
    /// Exponential-growth accounting for raising inflight_hi in UP
    /// (s2n `raise_inflight_hi_slope` / `probe_inflight_hi_upward`)
    bw_probe_up_rounds: u32,
    bw_probe_up_acks: u64,
    bw_probe_up_cnt: u64,
    /// Rounds elapsed since the last bandwidth probe, seeded randomly 0..=1
    /// (s2n `rounds_since_bw_probe`, reno coexistence)
    rounds_since_bw_probe: u8,
    /// Whether the flow was cwnd-limited at the start of the current round
    /// (s2n `cwnd_limited_in_round`; gates probing inflight_hi upward)
    cwnd_limited_in_round: bool,
    // --- congestion signals (s2n bbr/congestion.rs `State`) ---
    /// Loss-round counter: BBR reacts to congestion once per loss round
    loss_round: RoundCounter,
    /// 1-loss-round max of per-sample delivery rate (bytes/s)
    bw_latest: u64,
    /// 1-loss-round max of per-sample `rs.delivered`
    inflight_latest: u64,
    /// Discrete loss bursts in the current loss round (s2n counts a burst per
    /// run of contiguous lost packet numbers)
    loss_bursts_in_round: u8,
    /// RNG for `pick_probe_wait` (mirrors this crate's `Bbr`)
    random_number_generator: rand::rngs::StdRng,
}

impl Bbr2 {
    /// Construct a state using the given `config` and current time `now`
    pub fn new(config: Arc<Bbr2Config>, current_mtu: u16) -> Self {
        let initial_window = config.initial_window;
        Self {
            config,
            current_mtu: current_mtu as u64,
            max_bw_filter: MinMax::default(),
            rate_sample: RateSample::default(),
            delivered_total: 0,
            batch_acked: 0,
            acked_bytes: 0,
            mode: Mode::Startup,
            loss_state: Default::default(),
            recovery_state: RecoveryState::NotInRecovery,
            recovery_window: 0,
            is_at_full_bandwidth: false,
            pacing_gain: K_DEFAULT_HIGH_GAIN,
            high_gain: K_DEFAULT_HIGH_GAIN,
            drain_gain: 1.0 / K_DEFAULT_HIGH_GAIN,
            cwnd_gain: K_DEFAULT_HIGH_GAIN,
            high_cwnd_gain: K_DEFAULT_HIGH_GAIN,
            init_cwnd: initial_window,
            min_cwnd: calculate_min_window(current_mtu as u64),
            exit_probe_rtt_at: None,
            probe_rtt_last_started_at: None,
            min_rtt: Default::default(),
            exiting_quiescence: false,
            pacing_rate: 0,
            max_acked_packet_number: 0,
            max_sent_packet_number: 0,
            end_recovery_at_packet_number: 0,
            cwnd: initial_window,
            current_round_trip_end_packet_number: 0,
            round_count: 0,
            bw_at_last_round: 0,
            round_wo_bw_gain: 0,
            ack_aggregation: AckAggregationState::default(),
            inflight_hi: u64::MAX,
            inflight_lo: u64::MAX,
            round_lost: 0,
            round_delivered: 0,
            cycle_phase: CyclePhase::Up,
            cycle_start: None,
            bw_probe_wait: Duration::from_millis(2000),
            bw_probe_samples: false,
            bw_probe_up_rounds: 0,
            bw_probe_up_acks: 0,
            bw_probe_up_cnt: u64::MAX,
            rounds_since_bw_probe: 0,
            cwnd_limited_in_round: false,
            loss_round: RoundCounter::default(),
            bw_latest: 0,
            inflight_latest: 0,
            loss_bursts_in_round: 0,
            random_number_generator: rand::rngs::StdRng::from_os_rng(),
        }
    }

    /// The current maximum-bandwidth estimate (bytes/s)
    fn max_bw(&self) -> u64 {
        self.max_bw_filter.get()
    }

    /// s2n congestion.rs `loss_in_round`: any loss burst counted this round
    fn loss_in_round(&self) -> bool {
        self.loss_bursts_in_round > 0
    }

    /// s2n bbr.rs `is_probing_for_bandwidth`: states that accelerate sending
    /// to probe for bandwidth (Startup, ProbeBW_REFILL, ProbeBW_UP)
    fn is_probing_for_bandwidth(&self) -> bool {
        self.mode == Mode::Startup
            || (self.mode == Mode::ProbeBw
                && matches!(self.cycle_phase, CyclePhase::Refill | CyclePhase::Up))
    }

    fn enter_startup_mode(&mut self) {
        self.mode = Mode::Startup;
        self.pacing_gain = self.high_gain;
        self.cwnd_gain = self.high_cwnd_gain;
    }

    /// Enter PROBE_BW by starting the DOWN phase (s2n enter_probe_bw ->
    /// start_down). DOWN paces at 0.9 to drain the queue and yield headroom.
    fn enter_probe_bandwidth_mode(&mut self, now: Instant) {
        self.mode = Mode::ProbeBw;
        self.cwnd_gain = K_DERIVED_HIGH_CWNDGAIN;
        self.start_down(now);
    }

    /// BBRStartRound, adapted to this crate's packet-number round accounting
    fn start_new_round(&mut self) {
        self.current_round_trip_end_packet_number = self.max_sent_packet_number;
    }

    /// BBRResetCongestionSignals (s2n congestion.rs `reset`)
    fn reset_congestion_signals(&mut self) {
        self.loss_bursts_in_round = 0;
        self.bw_latest = 0;
        self.inflight_latest = 0;
    }

    // ---- PROBE_BW 4-phase cycle (ported from s2n bbr/probe_bw.rs) ----

    /// s2n start_down: reset congestion signals, stop growing inflight_hi,
    /// pick a randomized probe wait, stamp the phase, start a round, enter DOWN.
    fn start_down(&mut self, now: Instant) {
        self.reset_congestion_signals();
        self.bw_probe_up_cnt = u64::MAX;
        self.pick_probe_wait();
        self.cycle_start = Some(now);
        self.cycle_phase = CyclePhase::Down;
        self.pacing_gain = self.cycle_phase.pacing_gain();
        self.start_new_round();
    }

    /// s2n start_cruise: hold steady below inflight_hi.
    fn start_cruise(&mut self) {
        self.cycle_phase = CyclePhase::Cruise;
        self.pacing_gain = self.cycle_phase.pacing_gain();
    }

    /// s2n start_refill: reset the lower bound and probe-up accounting, one
    /// round of REFILL before probing up.
    fn start_refill(&mut self) {
        self.inflight_lo = u64::MAX; // reset_lower_bound (s2n data_volume.rs)
        self.bw_probe_up_rounds = 0;
        self.bw_probe_up_acks = 0;
        self.cycle_phase = CyclePhase::Refill;
        self.pacing_gain = self.cycle_phase.pacing_gain();
        self.start_new_round();
    }

    /// s2n start_up: stamp the phase and prime the inflight_hi growth slope.
    fn start_up(&mut self, now: Instant) {
        self.cycle_start = Some(now);
        self.cycle_phase = CyclePhase::Up;
        self.pacing_gain = self.cycle_phase.pacing_gain();
        self.start_new_round();
        self.raise_inflight_hi_slope();
    }

    /// s2n pick_probe_wait (BBRPickProbeWait): randomize the wall-clock wait
    /// (2..3 s) and the round bound (0 or 1) so competing flows' probe cycles
    /// desynchronize. This randomization is the coordination fairness depends
    /// on (bake-off approximation #3, now real).
    fn pick_probe_wait(&mut self) {
        self.rounds_since_bw_probe = self.random_number_generator.random_range(0..=1);
        self.bw_probe_wait =
            Duration::from_millis(self.random_number_generator.random_range(2000..=3000));
    }

    /// s2n raise_inflight_hi_slope: exponentially grow the per-round
    /// inflight_hi increment.
    fn raise_inflight_hi_slope(&mut self) {
        let growth_this_round: u64 = 1u64 << self.bw_probe_up_rounds.min(30);
        self.bw_probe_up_rounds = (self.bw_probe_up_rounds + 1).min(K_MAX_BW_PROBE_UP_ROUNDS);
        self.bw_probe_up_cnt = (self.cwnd / growth_this_round).max(self.current_mtu);
    }

    /// s2n probe_inflight_hi_upward: while UP and fully using inflight_hi,
    /// grow it by ~one MSS per `bw_probe_up_cnt` acked bytes. Driven by the
    /// batch's real delivered bytes (bake-off approximation #5, now real).
    fn probe_inflight_hi_upward(&mut self, bytes_acked: u64, round_start: bool) {
        self.bw_probe_up_acks += bytes_acked;
        if self.bw_probe_up_cnt > 0 && self.bw_probe_up_acks >= self.bw_probe_up_cnt {
            let delta = self.bw_probe_up_acks / self.bw_probe_up_cnt;
            self.bw_probe_up_acks -= delta * self.bw_probe_up_cnt;
            if self.inflight_hi != u64::MAX {
                self.inflight_hi = self.inflight_hi.saturating_add(delta * self.current_mtu);
            }
        }
        if round_start {
            self.raise_inflight_hi_slope();
        }
    }

    /// s2n target_inflight: min(bdp, cwnd).
    fn target_inflight(&self) -> u64 {
        self.get_bdp().min(self.cwnd)
    }

    /// BDP at unit gain (s2n bdp()).
    fn get_bdp(&self) -> u64 {
        let bw = self.max_bw();
        let bdp = self.min_rtt.as_micros() as u64 * bw / 1_000_000;
        if bdp == 0 { self.init_cwnd } else { bdp }
    }

    /// s2n inflight_with_headroom: 85% of inflight_hi, min-clamped.
    /// Returns u64::MAX when inflight_hi is disengaged.
    fn inflight_with_headroom(&self) -> u64 {
        if self.inflight_hi == u64::MAX {
            return u64::MAX;
        }
        let hr = (self.inflight_hi as f64 * K_HEADROOM as f64) as u64;
        hr.max(self.min_cwnd)
    }

    /// s2n is_loss_too_high (IsInflightTooHigh): the loss over the sample
    /// exceeded the ceiling. Gated on `loss_bursts_in_round` reaching
    /// PROBE_BW_FULL_LOSS_COUNT discrete bursts (bake-off approximation #1,
    /// now counted from real loss events).
    fn is_loss_too_high(&self, lost_bytes: u64, bytes_in_flight: u64) -> bool {
        self.loss_bursts_in_round >= K_PROBE_BW_FULL_LOSS_COUNT
            && lost_bytes as f32 > self.config.loss_thresh * bytes_in_flight as f32
    }

    /// s2n inflight_hi_from_lost_packet (BBRInflightHiFromLostPacket): the
    /// inflight volume at which losses crossed the loss threshold, computed
    /// from the lost packet's send-time stamps.
    fn inflight_hi_from_lost_packet(&self, sample: &LossSample) -> u64 {
        let loss_thresh = self.config.loss_thresh;
        // What was in flight before this packet?
        let inflight_prev = sample.tx_in_flight.saturating_sub(sample.bytes);
        // What was lost before this packet?
        let lost_prev = sample.lost.saturating_sub(sample.bytes);
        // BBRLossThresh * inflight_prev - lost_prev
        let loss_budget =
            ((loss_thresh * inflight_prev as f32) as u64).saturating_sub(lost_prev);
        // At what inflight value did losses cross BBRLossThresh?
        let lost_prefix = (loss_budget as f32 / (1.0 - loss_thresh)) as u64;
        inflight_prev + lost_prefix
    }

    /// s2n on_inflight_too_high (BBRHandleInflightTooHigh): react once per
    /// probe — set inflight_hi = max(inflight, BETA * target_inflight) unless
    /// the sample was app-limited; if in UP, start DOWN.
    fn on_inflight_too_high(&mut self, now: Instant, is_app_limited: bool, bytes_in_flight: u64) {
        self.bw_probe_samples = false; // only react once per bw probe
        if !is_app_limited {
            let beta_target = (K_BETA * self.target_inflight() as f32) as u64;
            self.inflight_hi = bytes_in_flight.max(beta_target);
        }
        if self.mode == Mode::ProbeBw && self.cycle_phase == CyclePhase::Up {
            self.start_down(now);
        }
    }

    /// s2n is_time_to_cruise: DOWN -> CRUISE readiness.
    fn is_time_to_cruise(&self, now: Instant, in_flight: u64) -> bool {
        // Chromium/Linux bound the time spent in DOWN to min_rtt, which
        // dominates on a real path.
        if let Some(cs) = self.cycle_start {
            if now.duration_since(cs) > self.min_rtt {
                return true;
            }
        }
        if in_flight > self.inflight_with_headroom() {
            return false; // not enough headroom
        }
        // inflight <= estimated BDP
        in_flight <= self.get_target_cwnd(1.0)
    }

    /// s2n is_time_to_probe_bw (BBRCheckTimeToProbeBW): DOWN/CRUISE -> REFILL
    /// when the randomized wait has elapsed or the reno-coexistence round
    /// bound is hit.
    fn is_time_to_probe_bw(&self, now: Instant) -> bool {
        if self
            .cycle_start
            .map(|cs| now.duration_since(cs) >= self.bw_probe_wait)
            .unwrap_or(false)
        {
            return true;
        }
        self.is_reno_coexistence_probe_time()
    }

    /// s2n is_reno_coexistence_probe_time (BBRIsRenoCoexistenceProbeTime)
    fn is_reno_coexistence_probe_time(&self) -> bool {
        let reno_rounds = self.target_inflight() / self.current_mtu.max(1);
        let rounds = reno_rounds.min(K_MAX_BW_PROBE_ROUNDS as u64) as u8;
        self.rounds_since_bw_probe >= rounds
    }

    /// s2n update_probe_bw_cycle_phase (BBRUpdateProbeBWCyclePhase): advance
    /// the cycle. Called once per ack batch while mode == ProbeBw and the pipe
    /// is filled; `adapt_upper_bounds` runs separately (s2n bbr.rs on_ack).
    fn update_gain_cycle_phase(&mut self, now: Instant, in_flight: u64, is_round_start: bool) {
        if is_round_start {
            // s2n ProbeBwState::on_round_start: reno-coexistence round count
            self.rounds_since_bw_probe = self.rounds_since_bw_probe.saturating_add(1);
        }
        match self.cycle_phase {
            CyclePhase::Down | CyclePhase::Cruise => {
                if self.is_time_to_probe_bw(now) {
                    self.start_refill();
                } else if self.cycle_phase == CyclePhase::Down
                    && self.is_time_to_cruise(now, in_flight)
                {
                    self.start_cruise();
                }
            }
            CyclePhase::Refill => {
                // After one full round of Refill, start Up.
                if is_round_start {
                    self.bw_probe_samples = true;
                    self.start_up(now);
                }
            }
            CyclePhase::Up => {
                // Leave UP once min_rtt has elapsed in-phase AND inflight has
                // exceeded the 1.25-gain target.
                let elapsed = self
                    .cycle_start
                    .map(|cs| now.duration_since(cs) > self.min_rtt)
                    .unwrap_or(false);
                if elapsed && in_flight > self.get_target_cwnd(1.25) {
                    self.start_down(now);
                }
            }
        }
    }

    /// s2n adapt_upper_bounds (BBRAdaptUpperBounds): if the current rate
    /// sample's loss is too high and we're probing, pull inflight_hi down;
    /// otherwise raise inflight_hi toward the sample's true `tx_in_flight`
    /// (bake-off approximation #4, now real) and, while UP and cwnd-limited,
    /// probe it upward.
    fn adapt_upper_bounds(&mut self, now: Instant, bytes_acked: u64, is_round_start: bool) {
        let rs = self.rate_sample;
        if self.is_loss_too_high(rs.lost, rs.tx_in_flight) {
            if self.bw_probe_samples {
                self.on_inflight_too_high(now, rs.is_app_limited, rs.tx_in_flight);
            }
        } else {
            if self.inflight_hi == u64::MAX {
                return; // no upper bound to raise
            }
            if rs.tx_in_flight > self.inflight_hi {
                self.inflight_hi = rs.tx_in_flight;
            }
            if self.mode == Mode::ProbeBw
                && self.cycle_phase == CyclePhase::Up
                && self.cwnd_limited_in_round
                && self.cwnd >= self.inflight_hi
            {
                // inflight_hi is fully utilized: probe if we can increase it
                self.probe_inflight_hi_upward(bytes_acked, is_round_start);
            }
        }
    }

    /// s2n update_lower_bound (data_volume.rs) — loss path only (ECN omitted).
    /// Called once per loss round when not probing for bandwidth: collapse
    /// inflight_lo toward BETA * inflight_lo but not below the loss round's
    /// `inflight_latest` (the per-sample delivered max — bake-off
    /// approximation #2, now real).
    fn update_inflight_lo(&mut self) {
        if !self.loss_in_round() {
            return;
        }
        if self.inflight_lo == u64::MAX {
            self.inflight_lo = self.cwnd;
        }
        let beta_bound = (K_BETA * self.inflight_lo as f32) as u64;
        self.inflight_lo = self.inflight_latest.max(beta_bound);
    }

    /// v2 delta retained from the bake-off port (bbr2.rs lineage): at each
    /// round boundary, adjust the persistent inflight ceiling from the
    /// completed round's loss rate. Kept as the STARTUP/DRAIN-phase ceiling;
    /// PROBE_BW drives inflight_hi via adapt_upper_bounds above.
    fn update_inflight_hi(&mut self, in_flight: u64) {
        let total = self.round_lost + self.round_delivered;
        if total > 0 {
            let loss_rate = self.round_lost as f32 / total as f32;
            if loss_rate > self.config.loss_thresh {
                let reduced = (in_flight as f32 * (1.0 - self.config.beta)) as u64;
                self.inflight_hi = self.inflight_hi.min(reduced.max(self.min_cwnd));
            } else if self.inflight_hi != u64::MAX && self.mode != Mode::ProbeBw {
                // In PROBE_BW, inflight_hi growth is handled by the cycle; only
                // additively re-probe here for the STARTUP/DRAIN ceiling.
                self.inflight_hi = self
                    .inflight_hi
                    .saturating_add(self.round_delivered.max(self.current_mtu));
            }
        }
    }

    fn update_recovery_state(&mut self, is_round_start: bool) {
        if self.loss_state.has_losses() {
            self.end_recovery_at_packet_number = self.max_sent_packet_number;
        }
        match self.recovery_state {
            RecoveryState::NotInRecovery if self.loss_state.has_losses() => {
                self.recovery_state = RecoveryState::Conservation;
                self.recovery_window = 0;
                self.current_round_trip_end_packet_number = self.max_sent_packet_number;
            }
            RecoveryState::Growth | RecoveryState::Conservation => {
                if self.recovery_state == RecoveryState::Conservation && is_round_start {
                    self.recovery_state = RecoveryState::Growth;
                }
                if !self.loss_state.has_losses()
                    && self.max_acked_packet_number > self.end_recovery_at_packet_number
                {
                    self.recovery_state = RecoveryState::NotInRecovery;
                }
            }
            _ => {}
        }
    }

    fn maybe_exit_startup_or_drain(&mut self, now: Instant, in_flight: u64) {
        if self.mode == Mode::Startup && self.is_at_full_bandwidth {
            self.mode = Mode::Drain;
            self.pacing_gain = self.drain_gain;
            self.cwnd_gain = self.high_cwnd_gain;
        }
        if self.mode == Mode::Drain && in_flight <= self.get_target_cwnd(1.0) {
            self.enter_probe_bandwidth_mode(now);
        }
    }

    fn is_min_rtt_expired(&self, now: Instant, app_limited: bool) -> bool {
        !app_limited
            && self
                .probe_rtt_last_started_at
                .map(|last| now.saturating_duration_since(last) > Duration::from_secs(10))
                .unwrap_or(true)
    }

    fn maybe_enter_or_exit_probe_rtt(
        &mut self,
        now: Instant,
        is_round_start: bool,
        bytes_in_flight: u64,
        app_limited: bool,
    ) {
        let min_rtt_expired = self.is_min_rtt_expired(now, app_limited);
        if min_rtt_expired && !self.exiting_quiescence && self.mode != Mode::ProbeRtt {
            self.mode = Mode::ProbeRtt;
            self.pacing_gain = 1.0;
            self.exit_probe_rtt_at = None;
            self.probe_rtt_last_started_at = Some(now);
        }
        if self.mode == Mode::ProbeRtt {
            match self.exit_probe_rtt_at {
                None => {
                    if bytes_in_flight < self.get_probe_rtt_cwnd() + self.current_mtu {
                        const K_PROBE_RTT_TIME: Duration = Duration::from_millis(200);
                        self.exit_probe_rtt_at = Some(now + K_PROBE_RTT_TIME);
                    }
                }
                Some(exit_time) if is_round_start && now >= exit_time => {
                    if !self.is_at_full_bandwidth {
                        self.enter_startup_mode();
                    } else {
                        self.enter_probe_bandwidth_mode(now);
                    }
                }
                Some(_) => {}
            }
        }
        self.exiting_quiescence = false;
    }

    fn get_target_cwnd(&self, gain: f32) -> u64 {
        let bw = self.max_bw();
        let bdp = self.min_rtt.as_micros() as u64 * bw;
        let bdpf = bdp as f64;
        let cwnd = ((gain as f64 * bdpf) / 1_000_000f64) as u64;
        if cwnd == 0 {
            return self.init_cwnd;
        }
        cwnd.max(self.min_cwnd)
    }

    fn get_probe_rtt_cwnd(&self) -> u64 {
        const K_MODERATE_PROBE_RTT_MULTIPLIER: f32 = 0.75;
        if PROBE_RTT_BASED_ON_BDP {
            return self.get_target_cwnd(K_MODERATE_PROBE_RTT_MULTIPLIER);
        }
        self.min_cwnd
    }

    fn calculate_pacing_rate(&mut self) {
        let bw = self.max_bw();
        if bw == 0 {
            return;
        }
        let target_rate = (bw as f64 * self.pacing_gain as f64) as u64;
        if self.is_at_full_bandwidth {
            self.pacing_rate = target_rate;
            return;
        }
        if self.pacing_rate == 0 && self.min_rtt.as_nanos() != 0 {
            self.pacing_rate = bw_from_delta(self.init_cwnd, self.min_rtt).unwrap_or(0);
            return;
        }
        if self.pacing_rate < target_rate {
            self.pacing_rate = target_rate;
        }
    }

    fn calculate_cwnd(&mut self, bytes_acked: u64, excess_acked: u64) {
        if self.mode == Mode::ProbeRtt {
            return;
        }
        let mut target_window = self.get_target_cwnd(self.cwnd_gain);
        if self.is_at_full_bandwidth {
            // Add the max recently measured ack aggregation to CWND.
            target_window += self.ack_aggregation.max_ack_height.get();
        } else {
            target_window += excess_acked;
        }
        if self.is_at_full_bandwidth {
            self.cwnd = target_window.min(self.cwnd + bytes_acked);
        } else if (self.cwnd_gain < target_window as f32) || (self.acked_bytes < self.init_cwnd) {
            self.cwnd += bytes_acked;
        }
        if self.cwnd < self.min_cwnd {
            self.cwnd = self.min_cwnd;
        }
    }

    fn calculate_recovery_window(&mut self, bytes_acked: u64, bytes_lost: u64, in_flight: u64) {
        if !self.recovery_state.in_recovery() {
            return;
        }
        if self.recovery_window == 0 {
            self.recovery_window = self.min_cwnd.max(in_flight + bytes_acked);
            return;
        }
        if self.recovery_window >= bytes_lost {
            self.recovery_window -= bytes_lost;
        } else {
            self.recovery_window = self.current_mtu;
        }
        if self.recovery_state == RecoveryState::Growth {
            self.recovery_window += bytes_acked;
        }
        self.recovery_window = self
            .recovery_window
            .max(in_flight + bytes_acked)
            .max(self.min_cwnd);
    }

    fn check_if_full_bw_reached(&mut self, app_limited: bool) {
        if app_limited {
            return;
        }
        let target = (self.bw_at_last_round as f64 * K_STARTUP_GROWTH_TARGET as f64) as u64;
        let bw = self.max_bw();
        if bw >= target {
            self.bw_at_last_round = bw;
            self.round_wo_bw_gain = 0;
            self.ack_aggregation.max_ack_height.reset();
            return;
        }
        self.round_wo_bw_gain += 1;
        if self.round_wo_bw_gain >= K_ROUND_TRIPS_WITHOUT_GROWTH_BEFORE_EXITING_STARTUP as u64
            || (self.recovery_state.in_recovery())
        {
            self.is_at_full_bandwidth = true;
        }
    }
}

impl Controller for Bbr2 {
    fn on_sent(&mut self, _now: Instant, _bytes: u64, last_packet_number: u64) {
        self.max_sent_packet_number = last_packet_number;
    }

    fn on_ack(
        &mut self,
        now: Instant,
        _sent: Instant,
        bytes: u64,
        app_limited: bool,
        rtt: &RttEstimator,
    ) {
        self.acked_bytes += bytes;
        if self.is_min_rtt_expired(now, app_limited) || self.min_rtt > rtt.min() {
            self.min_rtt = rtt.min();
        }
    }

    fn on_ack_sample(&mut self, _now: Instant, sample: &RateSample) {
        // Newest-sample rule (draft-cheng §3.3): keep the sample from the
        // packet with the highest P.delivered. Persists across batches, like
        // s2n's bw_estimator.rate_sample().
        if sample.prior_delivered >= self.rate_sample.prior_delivered {
            self.rate_sample = *sample;
        }
        self.delivered_total = self
            .delivered_total
            .max(sample.prior_delivered + sample.delivered);
        self.batch_acked += sample.bytes_acked;

        let rate = sample_delivery_rate(sample);
        // BBRUpdateMaxBw: app-limited samples may only raise the estimate
        if let Some(rate) = rate {
            if !sample.is_app_limited || rate >= self.max_bw() {
                self.max_bw_filter.update_max(self.round_count, rate);
            }
        }
        // BBRUpdateLatestDeliverySignals: 1-loss-round maxima
        // (s2n congestion::State::update)
        if let Some(rate) = rate {
            self.bw_latest = self.bw_latest.max(rate);
        }
        self.inflight_latest = self.inflight_latest.max(sample.delivered);
    }

    fn on_loss_sample(&mut self, now: Instant, sample: &LossSample) {
        // s2n congestion::State::on_packet_lost: pin the loss round's end and
        // count discrete loss bursts (bake-off approximation #1, now real).
        if !self.loss_in_round() {
            self.loss_round.set_round_end(sample.delivered);
        }
        if sample.new_loss_burst {
            self.loss_bursts_in_round = self.loss_bursts_in_round.saturating_add(1);
        }

        // s2n handle_lost_packet (draft 4.5.6.2)
        if !self.bw_probe_samples {
            return; // not a packet sent while probing bandwidth
        }
        if self.is_loss_too_high(sample.lost, sample.tx_in_flight) {
            let inflight_hi_from_lost_packet = self.inflight_hi_from_lost_packet(sample);
            self.on_inflight_too_high(now, sample.is_app_limited, inflight_hi_from_lost_packet);
        }
    }

    fn on_end_acks(
        &mut self,
        now: Instant,
        in_flight: u64,
        app_limited: bool,
        largest_packet_num_acked: Option<u64>,
    ) {
        let bytes_acked = self.batch_acked;
        self.batch_acked = 0;
        // s2n is_congestion_limited is computed before acked bytes leave the
        // pipe; quinn reports post-ack in_flight, so add them back.
        let is_cwnd_limited =
            self.cwnd.saturating_sub(in_flight + bytes_acked) < self.current_mtu;
        let excess_acked = self.ack_aggregation.update_ack_aggregation_bytes(
            bytes_acked,
            now,
            self.round_count,
            self.max_bw(),
        );
        if let Some(largest_acked_packet) = largest_packet_num_acked {
            self.max_acked_packet_number = largest_acked_packet;
        }

        // v2 delta: accumulate this batch's loss + delivery into the main
        // round (STARTUP/DRAIN ceiling accounting).
        self.round_lost += self.loss_state.lost_bytes;
        self.round_delivered += bytes_acked;

        let mut is_round_start = false;
        if bytes_acked > 0 {
            is_round_start =
                self.max_acked_packet_number > self.current_round_trip_end_packet_number;
            if is_round_start {
                self.current_round_trip_end_packet_number = self.max_sent_packet_number;
                self.round_count += 1;
                // s2n: latch the cwnd-limited flag once per round
                self.cwnd_limited_in_round = is_cwnd_limited;
            }
        }

        self.update_recovery_state(is_round_start);

        // Loss-round counter, driven by the newest real sample
        // (s2n round::Counter::on_ack once per ack event).
        let mut loss_round_start = false;
        if bytes_acked > 0 {
            self.loss_round
                .on_ack(self.rate_sample.prior_delivered, self.delivered_total);
            loss_round_start = self.loss_round.round_start;
        }

        // v2 delta: persistent STARTUP/DRAIN ceiling per main round.
        if is_round_start {
            self.update_inflight_hi(in_flight);
        }
        // s2n BBRAdaptLowerBoundsFromCongestion: once per loss round, only
        // when not accelerating to probe for bandwidth.
        if loss_round_start && !self.is_probing_for_bandwidth() {
            self.update_inflight_lo();
        }

        // s2n bbr.rs on_ack: adapt upper bounds whenever the pipe is filled;
        // advance the cycle only while in ProbeBw.
        if self.is_at_full_bandwidth {
            self.adapt_upper_bounds(now, bytes_acked, is_round_start);
            if self.mode == Mode::ProbeBw {
                self.update_gain_cycle_phase(now, in_flight, is_round_start);
            }
        }

        if is_round_start && !self.is_at_full_bandwidth {
            self.check_if_full_bw_reached(app_limited);
        }
        self.maybe_exit_startup_or_drain(now, in_flight);
        self.maybe_enter_or_exit_probe_rtt(now, is_round_start, in_flight, app_limited);

        self.calculate_pacing_rate();
        self.calculate_cwnd(bytes_acked, excess_acked);
        self.calculate_recovery_window(bytes_acked, self.loss_state.lost_bytes, in_flight);

        // BBRAdvanceLatestDeliverySignals (s2n congestion::State::advance):
        // reseed the loss-round maxima from the current sample.
        if loss_round_start {
            self.bw_latest = sample_delivery_rate(&self.rate_sample).unwrap_or(0);
            self.inflight_latest = self.rate_sample.delivered;
            self.loss_bursts_in_round = 0;
        }

        // Clear the main-round accumulators at a round boundary.
        if is_round_start {
            self.round_lost = 0;
            self.round_delivered = 0;
        }

        self.loss_state.reset();
    }

    fn on_congestion_event(
        &mut self,
        _now: Instant,
        _sent: Instant,
        _is_persistent_congestion: bool,
        lost_bytes: u64,
    ) {
        self.loss_state.lost_bytes += lost_bytes;
    }

    fn on_mtu_update(&mut self, new_mtu: u16) {
        self.current_mtu = new_mtu as u64;
        self.min_cwnd = calculate_min_window(self.current_mtu);
        self.init_cwnd = self.config.initial_window.max(self.min_cwnd);
        self.cwnd = self.cwnd.max(self.min_cwnd);
    }

    fn window(&self) -> u64 {
        let base = if self.mode == Mode::ProbeRtt {
            self.get_probe_rtt_cwnd()
        } else if self.recovery_state.in_recovery() && self.mode != Mode::Startup {
            self.cwnd.min(self.recovery_window)
        } else {
            self.cwnd
        };
        // Clamp to BOTH the loss-driven upper ceiling AND the short-term
        // lower bound. This min(cwnd, inflight_hi, inflight_lo) is the
        // yielding that produces fairness.
        base.min(self.inflight_hi)
            .min(self.inflight_lo)
            .max(self.min_cwnd)
    }

    fn metrics(&self) -> ControllerMetrics {
        ControllerMetrics {
            congestion_window: self.window(),
            ssthresh: None,
            pacing_rate: Some(self.pacing_rate.saturating_mul(8)),
        }
    }

    fn clone_box(&self) -> Box<dyn Controller> {
        Box::new(self.clone())
    }

    fn initial_window(&self) -> u64 {
        self.config.initial_window
    }

    fn into_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }
}

/// Configuration for [`Bbr2`]
///
/// `loss_thresh` and `beta` retain the bake-off port's knobs: they feed the
/// STARTUP/DRAIN `inflight_hi` ceiling and the loss-too-high test. Defaults
/// are the BBRv2 values; the s2n `BETA`/`HEADROOM` constants are hard-coded
/// separately as `K_BETA`/`K_HEADROOM`.
#[derive(Debug, Clone)]
pub struct Bbr2Config {
    initial_window: u64,
    loss_thresh: f32,
    beta: f32,
}

impl Bbr2Config {
    /// Default limit on the amount of outstanding data in bytes.
    ///
    /// Recommended value: `min(10 * max_datagram_size, max(2 * max_datagram_size, 14720))`
    pub fn initial_window(&mut self, value: u64) -> &mut Self {
        self.initial_window = value;
        self
    }

    /// Loss threshold as a fraction of inflight above which the round is
    /// considered congested (s2n LOSS_THRESH = 1/50)
    pub fn loss_thresh(&mut self, value: f32) -> &mut Self {
        self.loss_thresh = value;
        self
    }

    /// Multiplicative reduction applied to the STARTUP/DRAIN inflight ceiling
    /// on a congested round
    pub fn beta(&mut self, value: f32) -> &mut Self {
        self.beta = value;
        self
    }
}

impl Default for Bbr2Config {
    fn default() -> Self {
        Self {
            initial_window: K_MAX_INITIAL_CONGESTION_WINDOW * BASE_DATAGRAM_SIZE,
            loss_thresh: 0.02, // 2%, per BBRv2/v3 (s2n LOSS_THRESH = 1/50)
            beta: 0.3,
        }
    }
}

impl ControllerFactory for Bbr2Config {
    fn build(self: Arc<Self>, _now: Instant, current_mtu: u16) -> Box<dyn Controller> {
        Box::new(Bbr2::new(self, current_mtu))
    }
}

/// Delivery rate of a sample in bytes/s, `None` when the interval is empty
fn sample_delivery_rate(sample: &RateSample) -> Option<u64> {
    let micros = sample.interval.as_micros() as u64;
    if micros == 0 {
        return None;
    }
    Some(sample.delivered.saturating_mul(1_000_000) / micros)
}

const fn bw_from_delta(bytes: u64, delta: Duration) -> Option<u64> {
    let window_duration_ns = delta.as_nanos();
    if window_duration_ns == 0 {
        return None;
    }
    let b_ns = bytes * 1_000_000_000;
    let bytes_per_second = b_ns / (window_duration_ns as u64);
    Some(bytes_per_second)
}

fn calculate_min_window(current_mtu: u64) -> u64 {
    4 * current_mtu
}

// ---- BBRv1 skeleton helpers (unchanged from this crate's `Bbr`) ----

#[derive(Debug, Default, Copy, Clone)]
struct AckAggregationState {
    max_ack_height: MinMax,
    aggregation_epoch_start_time: Option<Instant>,
    aggregation_epoch_bytes: u64,
}

impl AckAggregationState {
    fn update_ack_aggregation_bytes(
        &mut self,
        newly_acked_bytes: u64,
        now: Instant,
        round: u64,
        max_bandwidth: u64,
    ) -> u64 {
        // Compute how many bytes are expected to be delivered, assuming max
        // bandwidth is correct.
        let expected_bytes_acked = max_bandwidth
            * now
                .saturating_duration_since(self.aggregation_epoch_start_time.unwrap_or(now))
                .as_micros() as u64
            / 1_000_000;

        // Reset the current aggregation epoch as soon as the ack arrival rate
        // is less than or equal to the max bandwidth.
        if self.aggregation_epoch_bytes <= expected_bytes_acked {
            // Reset to start measuring a new aggregation epoch.
            self.aggregation_epoch_bytes = newly_acked_bytes;
            self.aggregation_epoch_start_time = Some(now);
            return 0;
        }

        // Compute how many extra bytes were delivered vs max bandwidth.
        self.aggregation_epoch_bytes += newly_acked_bytes;
        let diff = self.aggregation_epoch_bytes - expected_bytes_acked;
        self.max_ack_height.update_max(round, diff);
        diff
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum Mode {
    // Startup phase of the connection.
    Startup,
    // After achieving the highest possible bandwidth during the startup, lower
    // the pacing rate in order to drain the queue.
    Drain,
    // Cruising mode.
    ProbeBw,
    // Temporarily slow down sending in order to empty the buffer and measure
    // the real minimum RTT.
    ProbeRtt,
}

// Indicates how the congestion control limits the amount of bytes in flight.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum RecoveryState {
    // Do not limit.
    NotInRecovery,
    // Allow an extra outstanding byte for each byte acknowledged.
    Conservation,
    // Allow two extra outstanding bytes for each byte acknowledged (slow
    // start).
    Growth,
}

impl RecoveryState {
    fn in_recovery(&self) -> bool {
        !matches!(self, Self::NotInRecovery)
    }
}

#[derive(Debug, Clone, Default)]
struct LossState {
    lost_bytes: u64,
}

impl LossState {
    fn reset(&mut self) {
        self.lost_bytes = 0;
    }

    fn has_losses(&self) -> bool {
        self.lost_bytes != 0
    }
}

const K_DEFAULT_HIGH_GAIN: f32 = 2.885;
const K_DERIVED_HIGH_CWNDGAIN: f32 = 2.0;
const K_STARTUP_GROWTH_TARGET: f32 = 1.25;
const K_ROUND_TRIPS_WITHOUT_GROWTH_BEFORE_EXITING_STARTUP: u8 = 3;
const K_MAX_INITIAL_CONGESTION_WINDOW: u64 = 200;
const PROBE_RTT_BASED_ON_BDP: bool = true;

// full-BBRv2 constants (from s2n-quic bbr.rs / probe_bw.rs)
/// s2n BETA = 7/10 — multiplicative factor for the inflight lower/upper bounds
const K_BETA: f32 = 0.7;
/// s2n HEADROOM = 85/100 — fraction of inflight_hi to stay under for headroom
const K_HEADROOM: f32 = 0.85;
/// s2n PROBE_BW_FULL_LOSS_COUNT = 2 — loss bursts before reacting in PROBE_BW
const K_PROBE_BW_FULL_LOSS_COUNT: u8 = 2;
/// s2n MAX_BW_PROBE_UP_ROUNDS = 30 — cap on the inflight_hi growth exponent
const K_MAX_BW_PROBE_UP_ROUNDS: u32 = 30;
/// s2n MAX_BW_PROBE_ROUNDS = 63 — cap on the reno-coexistence round bound
const K_MAX_BW_PROBE_ROUNDS: u8 = 63;

// Kathleen Nichols' windowed max filter, reproduced from this crate's
// bbr/min_max.rs (`pub(super)` there, so scoped to the `bbr` module; based on
// Google code released under BSD license, see that file for provenance).
#[derive(Copy, Clone, Debug)]
struct MinMax {
    /// round count, not time
    window: u64,
    samples: [MinMaxSample; 3],
}

impl MinMax {
    fn get(&self) -> u64 {
        self.samples[0].value
    }

    fn fill(&mut self, sample: MinMaxSample) {
        self.samples.fill(sample);
    }

    fn reset(&mut self) {
        self.fill(Default::default())
    }

    /// Check if new measurement updates the 1st, 2nd or 3rd choice max.
    fn update_max(&mut self, current_round: u64, measurement: u64) {
        let sample = MinMaxSample {
            time: current_round,
            value: measurement,
        };

        if self.samples[0].value == 0 /* uninitialized */
            || /* found new max? */ sample.value >= self.samples[0].value
            || /* nothing left in window? */ sample.time - self.samples[2].time > self.window
        {
            self.fill(sample); /* forget earlier samples */
            return;
        }

        if sample.value >= self.samples[1].value {
            self.samples[2] = sample;
            self.samples[1] = sample;
        } else if sample.value >= self.samples[2].value {
            self.samples[2] = sample;
        }

        self.subwin_update(sample);
    }

    /* As time advances, update the 1st, 2nd, and 3rd choices. */
    fn subwin_update(&mut self, sample: MinMaxSample) {
        let dt = sample.time - self.samples[0].time;
        if dt > self.window {
            /*
             * Passed entire window without a new sample so make 2nd
             * choice the new sample & 3rd choice the new 2nd choice.
             * we may have to iterate this since our 2nd choice
             * may also be outside the window (we checked on entry
             * that the third choice was in the window).
             */
            self.samples[0] = self.samples[1];
            self.samples[1] = self.samples[2];
            self.samples[2] = sample;
            if sample.time - self.samples[0].time > self.window {
                self.samples[0] = self.samples[1];
                self.samples[1] = self.samples[2];
                self.samples[2] = sample;
            }
        } else if self.samples[1].time == self.samples[0].time && dt > self.window / 4 {
            /*
             * We've passed a quarter of the window without a new sample
             * so take a 2nd choice from the 2nd quarter of the window.
             */
            self.samples[2] = sample;
            self.samples[1] = sample;
        } else if self.samples[2].time == self.samples[1].time && dt > self.window / 2 {
            /*
             * We've passed half the window without finding a new sample
             * so take a 3rd choice from the last half of the window
             */
            self.samples[2] = sample;
        }
    }
}

impl Default for MinMax {
    fn default() -> Self {
        Self {
            window: 10,
            samples: [Default::default(); 3],
        }
    }
}

#[derive(Debug, Copy, Clone, Default)]
struct MinMaxSample {
    /// round number, not time
    time: u64,
    value: u64,
}

#[cfg(test)]
mod tests {
    //! Deterministic state-machine tests for the BBRv2 machinery
    //! (`inflight_lo`, the 4-phase PROBE_BW cycle, the min(cwnd, hi, lo)
    //! window clamp, and the real-sample loss-burst gating), ported from the
    //! bake-off port's suite and adapted to the rate-sample data model. Each
    //! test encodes WHY the behavior matters (fairness / correctness), not
    //! just the mechanical transition.
    use super::*;

    const MTU: u16 = 1200;

    fn controller() -> Bbr2 {
        Bbr2::new(Arc::new(Bbr2Config::default()), MTU)
    }

    /// Put the controller in a steady PROBE_BW state with a known bandwidth
    /// and min_rtt so cycle/bound math is deterministic. `bw` is bytes/sec,
    /// `rtt` the path min RTT. Seeds the max-bw filter directly (round 0).
    fn prime_steady(cc: &mut Bbr2, bw: u64, rtt: Duration, now: Instant) {
        cc.min_rtt = rtt;
        cc.max_bw_filter.update_max(0, bw);
        cc.is_at_full_bandwidth = true;
        cc.mode = Mode::ProbeBw;
        cc.start_down(now);
    }

    fn loss_sample(bytes: u64, tx_in_flight: u64, lost: u64, new_loss_burst: bool) -> LossSample {
        LossSample {
            bytes,
            tx_in_flight,
            lost,
            delivered: 0,
            is_app_limited: false,
            new_loss_burst,
        }
    }

    // (1) STARTUP exits to DRAIN once the pipe is detected full. DRAIN is the
    // step that stops the exponential STARTUP ramp; without it the flow never
    // stops overshooting.
    #[test]
    fn startup_exits_to_drain_on_full_pipe() {
        let mut cc = controller();
        let now = Instant::now();
        assert_eq!(cc.mode, Mode::Startup);
        // Not full yet: stays in Startup.
        cc.maybe_exit_startup_or_drain(now, u64::MAX);
        assert_eq!(
            cc.mode,
            Mode::Startup,
            "must not leave Startup before full pipe"
        );
        // Full-pipe detected -> Drain, with the draining pacing gain (< 1.0).
        cc.is_at_full_bandwidth = true;
        cc.maybe_exit_startup_or_drain(now, u64::MAX);
        assert_eq!(
            cc.mode,
            Mode::Drain,
            "full pipe must transition Startup -> Drain"
        );
        assert!(
            cc.pacing_gain < 1.0,
            "Drain must pace below 1.0 to drain the queue"
        );
    }

    // (2a) A round whose loss rate exceeds the threshold ratchets the
    // persistent inflight_hi ceiling DOWN. This is the yield-under-congestion
    // mechanism for STARTUP/DRAIN.
    #[test]
    fn congestion_round_ratchets_inflight_hi_down() {
        let mut cc = controller();
        cc.mode = Mode::Startup; // exercise the STARTUP/DRAIN ceiling path
        let in_flight = 100 * MTU as u64;
        // A round that lost 10% (>> 2% thresh) of (lost+delivered).
        cc.round_delivered = 90 * MTU as u64;
        cc.round_lost = 10 * MTU as u64;
        cc.update_inflight_hi(in_flight);
        assert!(
            cc.inflight_hi < u64::MAX,
            "lossy round must engage the ceiling"
        );
        assert!(
            cc.inflight_hi <= in_flight,
            "ceiling must be at or below the outstanding volume, got {} vs {}",
            cc.inflight_hi,
            in_flight
        );
        assert!(
            cc.inflight_hi >= cc.min_cwnd,
            "ceiling must never drop below min window"
        );
    }

    // (2b) A clean round while probing in PROBE_BW:UP raises inflight_hi
    // (probes for more bandwidth once congestion clears — throughput
    // recovery). Driven by the real rate sample: tx_in_flight and per-ack
    // delivered bytes.
    #[test]
    fn clean_probe_up_round_raises_inflight_hi() {
        let mut cc = controller();
        let now = Instant::now();
        prime_steady(&mut cc, 10_000_000, Duration::from_millis(20), now);
        // Force UP with an engaged ceiling that cwnd is fully using.
        cc.cycle_phase = CyclePhase::Up;
        cc.inflight_hi = 50 * MTU as u64;
        cc.cwnd = 60 * MTU as u64;
        cc.bw_probe_samples = true;
        cc.bw_probe_up_cnt = 1; // make growth fire on the acked bytes below
        cc.cwnd_limited_in_round = true; // s2n probe-up gate
        // Clean rate sample (no loss since the acked packet's send).
        cc.rate_sample = RateSample {
            tx_in_flight: 45 * MTU as u64,
            lost: 0,
            ..RateSample::default()
        };
        let before = cc.inflight_hi;
        // Clean batch: adapt_upper_bounds grows the ceiling.
        cc.adapt_upper_bounds(now, 20 * MTU as u64, true);
        assert!(
            cc.inflight_hi > before,
            "clean PROBE_BW:UP batch must raise inflight_hi ({} !> {})",
            cc.inflight_hi,
            before
        );
    }

    // (3) PROBE_BW visits its phases in the canonical order Down -> Cruise ->
    // Refill -> Up. The DOWN phase (gain 0.9) is what creates headroom for the
    // competing flow; the order is what makes the cycle converge to fairness.
    #[test]
    fn probe_bw_cycles_down_cruise_refill_up() {
        let mut cc = controller();
        let t0 = Instant::now();
        let rtt = Duration::from_millis(10);
        prime_steady(&mut cc, 10_000_000, rtt, t0);
        // bw_probe_wait must exceed min_rtt so DOWN reaches CRUISE (min_rtt
        // elapsed) before it would jump to REFILL (bw_probe_wait elapsed).
        // start_down randomized it to 2..3s; pin it for determinism.
        cc.bw_probe_wait = Duration::from_millis(50);
        cc.rounds_since_bw_probe = 0; // pin the randomized reno-coexistence seed
        assert_eq!(cc.cycle_phase, CyclePhase::Down, "PROBE_BW starts in Down");

        // Advance past min_rtt but not past bw_probe_wait, low inflight -> Cruise.
        let t1 = t0 + Duration::from_millis(15);
        cc.update_gain_cycle_phase(t1, cc.min_cwnd, false);
        assert_eq!(
            cc.cycle_phase,
            CyclePhase::Cruise,
            "Down -> Cruise after min_rtt"
        );

        // Advance past bw_probe_wait -> Refill.
        let t2 = t0 + Duration::from_millis(60);
        cc.update_gain_cycle_phase(t2, cc.min_cwnd, false);
        assert_eq!(
            cc.cycle_phase,
            CyclePhase::Refill,
            "Cruise -> Refill after bw_probe_wait"
        );

        // One round-start in Refill -> Up.
        cc.update_gain_cycle_phase(t2, cc.min_cwnd, true);
        assert_eq!(cc.cycle_phase, CyclePhase::Up, "Refill -> Up after one round");
        assert!(
            (cc.pacing_gain - 1.25).abs() < 1e-6,
            "Up must pace at 1.25"
        );
    }

    // (3b) The DOWN phase paces below the bottleneck rate (0.9) — this is the
    // concrete yielding gain that leaves room for the competing flow.
    #[test]
    fn down_phase_paces_below_one() {
        assert!(CyclePhase::Down.pacing_gain() < 1.0);
        assert_eq!(CyclePhase::Cruise.pacing_gain(), 1.0);
        assert_eq!(CyclePhase::Refill.pacing_gain(), 1.0);
        assert!(CyclePhase::Up.pacing_gain() > 1.0);
    }

    // (4) window() is never below the minimum window, no matter how far the
    // loss-driven bounds have collapsed. A window below min_cwnd would stall
    // the connection entirely.
    #[test]
    fn window_never_below_min() {
        let mut cc = controller();
        // Collapse both bounds to (near) zero.
        cc.inflight_hi = 1;
        cc.inflight_lo = 1;
        cc.cwnd = 1;
        assert_eq!(cc.window(), cc.min_cwnd, "window must clamp up to min_cwnd");
        // Even in recovery / probe-rtt the floor holds.
        cc.mode = Mode::ProbeRtt;
        assert!(
            cc.window() >= cc.min_cwnd,
            "ProbeRtt window must respect the floor"
        );
    }

    // (4b) The lower bound actually participates in window(): once
    // inflight_lo collapses on a loss round, window() reflects it (the flow
    // yields), while still respecting the floor. The collapse floor is the
    // loss round's real per-sample delivered max (inflight_latest), not the
    // whole round's delivered volume.
    #[test]
    fn inflight_lo_collapses_window_on_loss() {
        let mut cc = controller();
        let now = Instant::now();
        cc.cwnd = 500 * MTU as u64;
        cc.inflight_hi = u64::MAX;
        // No loss yet: window == cwnd (both bounds disengaged).
        assert_eq!(cc.window(), cc.cwnd);
        // A real loss event opens the loss round (burst counted from the
        // sample, not inferred from batch timing).
        cc.on_loss_sample(now, &loss_sample(MTU as u64, 200 * MTU as u64, MTU as u64, true));
        assert!(cc.loss_in_round(), "loss sample must open the loss round");
        // The loss round's delivered ceiling observed via ack samples.
        cc.inflight_latest = 100 * MTU as u64;
        cc.update_inflight_lo();
        assert!(
            cc.inflight_lo < u64::MAX,
            "loss round must engage inflight_lo"
        );
        assert!(
            cc.window() < cc.cwnd,
            "engaged inflight_lo must reduce window below cwnd ({} !< {})",
            cc.window(),
            cc.cwnd
        );
        assert!(cc.window() >= cc.min_cwnd);
    }

    // (5) The factory produces independent per-connection controllers:
    // cloning / building twice from one config must not share mutable state,
    // or two connections would corrupt each other's congestion window.
    #[test]
    fn factory_builds_independent_state() {
        let cfg = Arc::new(Bbr2Config::default());
        let mut a = cfg.clone().build(Instant::now(), MTU);
        let b = cfg.clone().build(Instant::now(), MTU);
        // Mutate a's state via the trait, leaving b untouched.
        a.on_congestion_event(Instant::now(), Instant::now(), false, 999_999);
        // Downcast to compare private state.
        let a_cc = a.into_any().downcast::<Bbr2>().unwrap();
        let b_cc = b.into_any().downcast::<Bbr2>().unwrap();
        assert_eq!(a_cc.loss_state.lost_bytes, 999_999, "a recorded its own loss");
        assert_eq!(
            b_cc.loss_state.lost_bytes, 0,
            "b must be unaffected by a's loss"
        );
        // Same starting window, independent instances.
        assert_eq!(b_cc.window(), controller().window());
    }

    // (6) The PROBE_BW loss reaction is gated on >= 2 DISCRETE loss bursts
    // per round (s2n PROBE_BW_FULL_LOSS_COUNT): one contiguous burst must not
    // collapse inflight_hi, or a single stray drop run would forfeit
    // throughput. This is the burst semantics the bake-off port could only
    // approximate per ack batch.
    #[test]
    fn single_loss_burst_does_not_trigger_inflight_too_high() {
        let mut cc = controller();
        let now = Instant::now();
        prime_steady(&mut cc, 10_000_000, Duration::from_millis(20), now);
        cc.bw_probe_samples = true;
        let initial_hi = 200 * MTU as u64;
        cc.inflight_hi = initial_hi;
        // One burst with heavy loss (10 MTU lost of 100 MTU in flight >> 2%):
        // burst count 1 < 2, so no reaction yet.
        cc.on_loss_sample(
            now,
            &loss_sample(MTU as u64, 100 * MTU as u64, 10 * MTU as u64, true),
        );
        assert_eq!(
            cc.inflight_hi, initial_hi,
            "a single loss burst must not collapse inflight_hi"
        );
        assert!(cc.bw_probe_samples, "reaction must not have been consumed");
        // A second discrete burst crosses PROBE_BW_FULL_LOSS_COUNT: react.
        cc.on_loss_sample(
            now,
            &loss_sample(MTU as u64, 100 * MTU as u64, 10 * MTU as u64, true),
        );
        assert!(
            cc.inflight_hi < initial_hi,
            "two discrete bursts above loss_thresh must pull inflight_hi down"
        );
        assert!(
            !cc.bw_probe_samples,
            "the reaction fires once per bandwidth probe"
        );
    }
}
