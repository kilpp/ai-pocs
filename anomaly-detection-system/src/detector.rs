use crate::features::extract_features;
use crate::isolation_forest::IsolationForest;
use crate::parser::NetworkEvent;
use crate::reporter::AnomalyReport;

/// Configuration for the anomaly detector.
pub struct DetectorConfig {
    pub n_trees: usize,
    pub buffer_size: usize,
    pub threshold: f64,
    pub retrain_interval: usize,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            n_trees: 100,
            buffer_size: 256,
            threshold: 0.65,
            retrain_interval: 1000,
        }
    }
}

/// Orchestrates the anomaly detection pipeline:
/// buffering → training → scoring → reporting.
pub struct Detector {
    config: DetectorConfig,
    forest: Option<IsolationForest>,
    buffer: Vec<Vec<f64>>,
    events_since_train: usize,
    total_events: usize,
    total_anomalies: usize,
}

impl Detector {
    pub fn new(config: DetectorConfig) -> Self {
        Self {
            config,
            forest: None,
            buffer: Vec::new(),
            events_since_train: 0,
            total_events: 0,
            total_anomalies: 0,
        }
    }

    /// Process a single network event.
    ///
    /// Returns `Some(AnomalyReport)` if the event is anomalous,
    /// `None` if normal or still buffering.
    pub fn process(&mut self, event: &NetworkEvent) -> Option<AnomalyReport> {
        let features = extract_features(event);
        self.total_events += 1;

        // Buffering phase: collect initial samples for training
        if self.forest.is_none() {
            self.buffer.push(features);
            if self.buffer.len() >= self.config.buffer_size {
                self.train();
            }
            return None;
        }

        // Score the raw feature vector directly.
        // Isolation Forest handles varying feature scales inherently
        // through its random split mechanism.
        let score = self.forest.as_ref().unwrap().score(&features);

        self.events_since_train += 1;

        // Buffer for periodic retraining
        self.buffer.push(features);
        if self.buffer.len() > self.config.buffer_size * 2 {
            let drain_count = self.buffer.len() - self.config.buffer_size;
            self.buffer.drain(0..drain_count);
        }
        if self.events_since_train >= self.config.retrain_interval {
            self.train();
        }

        if score >= self.config.threshold {
            self.total_anomalies += 1;
            Some(AnomalyReport {
                event: event.clone(),
                score,
                event_number: self.total_events,
            })
        } else {
            None
        }
    }

    fn train(&mut self) {
        self.forest = Some(IsolationForest::fit(
            &self.buffer,
            self.config.n_trees,
            self.config.buffer_size,
        ));
        self.events_since_train = 0;
    }

    pub fn total_events(&self) -> usize {
        self.total_events
    }

    pub fn total_anomalies(&self) -> usize {
        self.total_anomalies
    }

    pub fn is_trained(&self) -> bool {
        self.forest.is_some()
    }
}
