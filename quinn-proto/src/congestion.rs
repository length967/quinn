//! Logic for controlling the rate at which data is sent

use crate::connection::RttEstimator;
use crate::{Duration, Instant};
use std::any::Any;
use std::sync::Arc;

mod bbr;
mod bbr2; // haul patch: BBRv2 controller over the per-ack rate-sample plumbing
mod cubic;
mod new_reno;

pub use bbr::{Bbr, BbrConfig};
pub use bbr2::{Bbr2, Bbr2Config}; // haul patch
pub use cubic::{Cubic, CubicConfig};
pub use new_reno::{NewReno, NewRenoConfig};

/// Common interface for different congestion controllers
pub trait Controller: Send + Sync {
    /// One or more packets were just sent
    #[allow(unused_variables)]
    fn on_sent(&mut self, now: Instant, bytes: u64, last_packet_number: u64) {}

    /// Packet deliveries were confirmed
    ///
    /// `app_limited` indicates whether the connection was blocked on outgoing
    /// application data prior to receiving these acknowledgements.
    #[allow(unused_variables)]
    fn on_ack(
        &mut self,
        now: Instant,
        sent: Instant,
        bytes: u64,
        app_limited: bool,
        rtt: &RttEstimator,
    ) {
    }

    /// Packets are acked in batches, all with the same `now` argument. This indicates one of those batches has completed.
    #[allow(unused_variables)]
    fn on_end_acks(
        &mut self,
        now: Instant,
        in_flight: u64,
        app_limited: bool,
        largest_packet_num_acked: Option<u64>,
    ) {
    }

    /// Packets were deemed lost or marked congested
    ///
    /// `in_persistent_congestion` indicates whether all packets sent within the persistent
    /// congestion threshold period ending when the most recent packet in this batch was sent were
    /// lost.
    /// `lost_bytes` indicates how many bytes were lost. This value will be 0 for ECN triggers.
    fn on_congestion_event(
        &mut self,
        now: Instant,
        sent: Instant,
        is_persistent_congestion: bool,
        lost_bytes: u64,
    );

    /// haul patch: a per-ack delivery-rate sample
    /// (draft-cheng-iccrg-delivery-rate-estimation)
    ///
    /// Called once per newly acked ack-eliciting packet, alongside
    /// [`Controller::on_ack`], with a sample computed from delivery state
    /// stamped into the packet at send time. Default no-op so existing
    /// controllers are unaffected.
    #[allow(unused_variables)]
    fn on_ack_sample(&mut self, now: Instant, sample: &RateSample) {}

    /// haul patch: a per-loss-event sample
    ///
    /// Called once per packet declared lost (in packet-number order) with the
    /// state stamped at that packet's send time. [`Self::on_congestion_event`]
    /// is still delivered once per loss batch; this adds the per-packet
    /// granularity BBRv2-style controllers need. Default no-op.
    #[allow(unused_variables)]
    fn on_loss_sample(&mut self, now: Instant, sample: &LossSample) {}

    /// The known MTU for the current network path has been updated
    fn on_mtu_update(&mut self, new_mtu: u16);

    /// Number of ack-eliciting bytes that may be in flight
    fn window(&self) -> u64;

    /// Retrieve implementation-specific metrics used to populate `qlog` traces when they are enabled
    fn metrics(&self) -> ControllerMetrics {
        ControllerMetrics {
            congestion_window: self.window(),
            ssthresh: None,
            pacing_rate: None,
        }
    }

    /// Duplicate the controller's state
    fn clone_box(&self) -> Box<dyn Controller>;

    /// Initial congestion window
    fn initial_window(&self) -> u64;

    /// Returns Self for use in down-casting to extract implementation details
    fn into_any(self: Box<Self>) -> Box<dyn Any>;
}

/// Common congestion controller metrics
#[derive(Default)]
#[non_exhaustive]
pub struct ControllerMetrics {
    /// Congestion window (bytes)
    pub congestion_window: u64,
    /// Slow start threshold (bytes)
    pub ssthresh: Option<u64>,
    /// Pacing rate (bits/s)
    pub pacing_rate: Option<u64>,
}

/// haul patch: a per-ack delivery-rate sample
/// (draft-cheng-iccrg-delivery-rate-estimation §3.3)
///
/// "C" refers to the connection's running delivery totals, "P" to the state
/// stamped into the acked packet when it was sent.
#[derive(Debug, Clone, Copy, Default)]
pub struct RateSample {
    /// Bytes delivered between the acked packet's send and its ack
    /// (`C.delivered - P.delivered`)
    pub delivered: u64,
    /// `C.delivered` when the acked packet was sent (`P.delivered`); round
    /// counters compare this against a delivered-bytes round end
    pub prior_delivered: u64,
    /// The sampling interval: `max(send_elapsed, ack_elapsed)`
    pub interval: Duration,
    /// Bytes in flight when the acked packet was sent, including that packet
    pub tx_in_flight: u64,
    /// Bytes declared lost between the acked packet's send and its ack
    /// (`C.lost - P.lost`)
    pub lost: u64,
    /// Whether the connection was app-limited when the acked packet was sent
    pub is_app_limited: bool,
    /// Bytes newly acked by this packet (the packet's size)
    pub bytes_acked: u64,
}

/// haul patch: a per-loss-event sample (the data model of s2n-quic's
/// `CongestionController::on_packet_lost`)
#[derive(Debug, Clone, Copy)]
pub struct LossSample {
    /// Size in bytes of the lost packet
    pub bytes: u64,
    /// Bytes in flight when the lost packet was sent, including that packet
    pub tx_in_flight: u64,
    /// Bytes declared lost between the packet's send and its loss declaration,
    /// including the packet itself (`C.lost - P.lost`)
    pub lost: u64,
    /// Total bytes delivered (`C.delivered`) when the loss was declared
    pub delivered: u64,
    /// Whether the connection was app-limited when the lost packet was sent
    pub is_app_limited: bool,
    /// Whether this packet starts a new loss burst (i.e. it is not contiguous
    /// with the previously declared lost packet)
    pub new_loss_burst: bool,
}

/// Constructs controllers on demand
pub trait ControllerFactory {
    /// Construct a fresh `Controller`
    fn build(self: Arc<Self>, now: Instant, current_mtu: u16) -> Box<dyn Controller>;
}

const BASE_DATAGRAM_SIZE: u64 = 1200;
