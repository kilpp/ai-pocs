use rand::Rng;

/// Average path length of unsuccessful search in a Binary Search Tree.
/// Used to normalize the anomaly score.
fn c(n: f64) -> f64 {
    if n <= 1.0 {
        return 0.0;
    }
    2.0 * (n.ln() + 0.5772156649) - (2.0 * (n - 1.0) / n)
}

/// A node in an isolation tree.
enum IsolationNode {
    /// Internal node: split on `feature` at `threshold`.
    Branch {
        feature: usize,
        threshold: f64,
        left: Box<IsolationNode>,
        right: Box<IsolationNode>,
    },
    /// Leaf node reached at a given depth, holding the count of samples that landed here.
    Leaf {
        size: usize,
    },
}

impl IsolationNode {
    /// Compute the path length for a given point through this subtree.
    fn path_length(&self, point: &[f64], depth: usize) -> f64 {
        match self {
            IsolationNode::Leaf { size } => {
                depth as f64 + c(*size as f64)
            }
            IsolationNode::Branch {
                feature,
                threshold,
                left,
                right,
            } => {
                if point[*feature] < *threshold {
                    left.path_length(point, depth + 1)
                } else {
                    right.path_length(point, depth + 1)
                }
            }
        }
    }
}

/// A single isolation tree.
pub struct IsolationTree {
    root: IsolationNode,
}

impl IsolationTree {
    /// Build an isolation tree from the given data with a maximum depth limit.
    pub fn fit(data: &[Vec<f64>], max_depth: usize, rng: &mut impl Rng) -> Self {
        let root = Self::build_node(data, 0, max_depth, rng);
        IsolationTree { root }
    }

    fn build_node(
        data: &[Vec<f64>],
        depth: usize,
        max_depth: usize,
        rng: &mut impl Rng,
    ) -> IsolationNode {
        // Base cases: max depth reached, or too few samples to split
        if depth >= max_depth || data.len() <= 1 {
            return IsolationNode::Leaf { size: data.len() };
        }

        let n_features = data[0].len();
        let feature = rng.gen_range(0..n_features);

        // Find min/max for the chosen feature
        let mut min_val = f64::MAX;
        let mut max_val = f64::MIN;
        for sample in data {
            if sample[feature] < min_val {
                min_val = sample[feature];
            }
            if sample[feature] > max_val {
                max_val = sample[feature];
            }
        }

        // If all values are the same, can't split
        if (max_val - min_val).abs() < f64::EPSILON {
            return IsolationNode::Leaf { size: data.len() };
        }

        // Random split point between min and max
        let threshold = rng.gen_range(min_val..max_val);

        let (left_data, right_data): (Vec<_>, Vec<_>) =
            data.iter().cloned().partition(|sample| sample[feature] < threshold);

        // Avoid empty partitions
        if left_data.is_empty() || right_data.is_empty() {
            return IsolationNode::Leaf { size: data.len() };
        }

        IsolationNode::Branch {
            feature,
            threshold,
            left: Box::new(Self::build_node(&left_data, depth + 1, max_depth, rng)),
            right: Box::new(Self::build_node(&right_data, depth + 1, max_depth, rng)),
        }
    }

    /// Compute path length for a single data point.
    fn path_length(&self, point: &[f64]) -> f64 {
        self.root.path_length(point, 0)
    }
}

/// An ensemble of isolation trees for anomaly detection.
pub struct IsolationForest {
    trees: Vec<IsolationTree>,
    sample_size: usize,
}

impl IsolationForest {
    /// Train an isolation forest on the provided data.
    ///
    /// - `n_trees`: number of isolation trees (default: 100)
    /// - `sample_size`: subsample size for each tree (default: 256)
    pub fn fit(data: &[Vec<f64>], n_trees: usize, sample_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let max_depth = (sample_size as f64).log2().ceil() as usize;
        let actual_sample_size = sample_size.min(data.len());

        let trees: Vec<IsolationTree> = (0..n_trees)
            .map(|_| {
                // Subsample the data
                let subsample: Vec<Vec<f64>> = (0..actual_sample_size)
                    .map(|_| {
                        let idx = rng.gen_range(0..data.len());
                        data[idx].clone()
                    })
                    .collect();
                IsolationTree::fit(&subsample, max_depth, &mut rng)
            })
            .collect();

        IsolationForest {
            trees,
            sample_size: actual_sample_size,
        }
    }

    /// Compute the anomaly score for a data point.
    ///
    /// Returns a score in [0, 1] where:
    /// - score close to 1.0 = anomaly
    /// - score close to 0.5 = normal
    /// - score close to 0.0 = very normal (dense region)
    pub fn score(&self, point: &[f64]) -> f64 {
        let avg_path_length: f64 =
            self.trees.iter().map(|t| t.path_length(point)).sum::<f64>() / self.trees.len() as f64;

        let cn = c(self.sample_size as f64);
        if cn == 0.0 {
            return 0.5;
        }

        // Anomaly score: s = 2^(-E(h(x)) / c(n))
        2.0_f64.powf(-avg_path_length / cn)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_c_function() {
        assert_eq!(c(1.0), 0.0);
        assert!(c(2.0) > 0.0);
        assert!(c(256.0) > c(2.0));
    }

    #[test]
    fn test_fit_and_score() {
        // Generate normal data: values clustered around [0.5, 0.5, 0.5]
        let mut data: Vec<Vec<f64>> = Vec::new();
        let mut rng = rand::thread_rng();
        for _ in 0..200 {
            data.push(vec![
                0.5 + rng.gen_range(-0.1..0.1),
                0.5 + rng.gen_range(-0.1..0.1),
                0.5 + rng.gen_range(-0.1..0.1),
            ]);
        }

        let forest = IsolationForest::fit(&data, 100, 128);

        // Normal point should have lower anomaly score
        let normal_score = forest.score(&[0.5, 0.5, 0.5]);
        // Anomalous point (far from cluster) should have higher score
        let anomaly_score = forest.score(&[10.0, 10.0, 10.0]);

        assert!(
            anomaly_score > normal_score,
            "Anomaly score ({}) should be greater than normal score ({})",
            anomaly_score,
            normal_score
        );
    }

    #[test]
    fn test_score_range() {
        let data = vec![
            vec![1.0, 2.0],
            vec![1.1, 2.1],
            vec![0.9, 1.9],
            vec![1.0, 2.0],
            vec![1.05, 1.95],
        ];
        let forest = IsolationForest::fit(&data, 50, 5);
        let score = forest.score(&[1.0, 2.0]);
        assert!(score >= 0.0 && score <= 1.0, "Score {} out of range", score);
    }
}
