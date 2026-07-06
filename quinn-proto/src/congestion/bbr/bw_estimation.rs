use std::collections::VecDeque;
use std::fmt::{Debug, Display, Formatter};

use super::min_max::MinMax;
use crate::{Duration, Instant};

/// Snapshot spacing / retention for `delivered_history` (haul patch, below):
/// ~4 ms grain × 256 entries ≈ the last second of delivery history, enough to
/// look back one full RTT on any realistic WAN path.
const HISTORY_GRAIN: Duration = Duration::from_millis(4);
const HISTORY_LEN: usize = 256;

#[derive(Clone, Debug, Default)]
pub(crate) struct BandwidthEstimation {
    total_acked: u64,
    prev_total_acked: u64,
    acked_time: Option<Instant>,
    prev_acked_time: Option<Instant>,
    total_sent: u64,
    prev_total_sent: u64,
    sent_time: Option<Instant>,
    prev_sent_time: Option<Instant>,
    max_filter: MinMax,
    acked_at_last_window: u64,
    /// haul patch (see `on_ack`): (time, total_acked) snapshots ~4 ms apart
    /// covering the last ~1 s, so an ack can be rated over the acked packet's
    /// whole flight time instead of the gap since the previous ack event.
    delivered_history: VecDeque<(Instant, u64)>,
}

impl BandwidthEstimation {
    pub(crate) fn on_sent(&mut self, now: Instant, bytes: u64) {
        self.prev_total_sent = self.total_sent;
        self.total_sent += bytes;
        self.prev_sent_time = self.sent_time;
        self.sent_time = Some(now);
    }

    pub(crate) fn on_ack(
        &mut self,
        now: Instant,
        sent: Instant,
        bytes: u64,
        round: u64,
        app_limited: bool,
    ) {
        self.prev_total_acked = self.total_acked;
        self.total_acked += bytes;
        self.prev_acked_time = self.acked_time;
        self.acked_time = Some(now);

        let prev_sent_time = match self.prev_sent_time {
            Some(prev_sent_time) => prev_sent_time,
            None => return,
        };

        let send_rate = match self.sent_time {
            Some(sent_time) if sent_time > prev_sent_time => Self::bw_from_delta(
                self.total_sent - self.prev_total_sent,
                sent_time - prev_sent_time,
            )
            .unwrap_or(0),
            _ => u64::MAX, // will take the min of send and ack, so this is just a skip
        };

        // haul patch — delivery-rate sample over the acked packet's FLIGHT
        // time (draft-cheng-iccrg-delivery-rate-estimation), not the gap since
        // the previous ack event. The upstream sample (this batch's bytes over
        // the inter-ack gap) explodes under ack compression: when many
        // connections share one receiver socket, ACKs arrive in bursts with
        // microsecond gaps, the burst samples latch the max filter 10-30x
        // above the real rate, and cwnd (= gain x bw x min_rtt) settles at
        // 20-30x BDP -> ~60% sustained ensemble loss (measured, haul
        // wan-test/results-autotune-2026-07-06.md). Rating over the flight
        // time bounds the sample by what the path actually delivered in an
        // RTT, which ack batching cannot inflate.
        let delivered_at_send = self.delivered_before(sent);
        let ack_rate = match now > sent {
            true => Self::bw_from_delta(
                self.total_acked.saturating_sub(delivered_at_send),
                now - sent,
            )
            .unwrap_or(0),
            false => 0,
        };

        // Snapshot AFTER sampling, throttled to the history grain.
        match self.delivered_history.back() {
            Some(&(t, _)) if now.saturating_duration_since(t) < HISTORY_GRAIN => {}
            _ => {
                self.delivered_history.push_back((now, self.total_acked));
                if self.delivered_history.len() > HISTORY_LEN {
                    self.delivered_history.pop_front();
                }
            }
        }

        let bandwidth = send_rate.min(ack_rate);
        if !app_limited && self.max_filter.get() < bandwidth {
            self.max_filter.update_max(round, bandwidth);
        }
    }

    /// haul patch: `total_acked` as of the newest snapshot at-or-before `t`.
    /// Falls back to the oldest snapshot when history doesn't reach back to
    /// `t` (short-lived underestimate — safe direction), and to 0 when empty
    /// (connection start: everything delivered happened within this flight).
    fn delivered_before(&self, t: Instant) -> u64 {
        let mut best = None;
        for &(ts, total) in &self.delivered_history {
            if ts <= t {
                best = Some(total);
            } else {
                break;
            }
        }
        best.or_else(|| self.delivered_history.front().map(|&(_, total)| total))
            .unwrap_or(0)
    }

    pub(crate) fn bytes_acked_this_window(&self) -> u64 {
        self.total_acked - self.acked_at_last_window
    }

    pub(crate) fn end_acks(&mut self, _current_round: u64, _app_limited: bool) {
        self.acked_at_last_window = self.total_acked;
    }

    pub(crate) fn get_estimate(&self) -> u64 {
        self.max_filter.get()
    }

    pub(crate) const fn bw_from_delta(bytes: u64, delta: Duration) -> Option<u64> {
        let window_duration_ns = delta.as_nanos();
        if window_duration_ns == 0 {
            return None;
        }
        let b_ns = bytes * 1_000_000_000;
        let bytes_per_second = b_ns / (window_duration_ns as u64);
        Some(bytes_per_second)
    }
}

impl Display for BandwidthEstimation {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:.3} MB/s",
            self.get_estimate() as f32 / (1024 * 1024) as f32
        )
    }
}
