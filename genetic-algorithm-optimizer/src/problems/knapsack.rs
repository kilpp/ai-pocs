use std::sync::OnceLock;

use rand::Rng;

use crate::genome::Genome;

#[derive(Debug, Clone, Copy)]
pub struct Item {
    pub weight: f64,
    pub value: f64,
}

#[derive(Debug)]
struct KnapsackConfig {
    items: Vec<Item>,
    capacity: f64,
}

static CONFIG: OnceLock<KnapsackConfig> = OnceLock::new();

pub fn configure(items: Vec<Item>, capacity: f64) {
    CONFIG
        .set(KnapsackConfig { items, capacity })
        .expect("knapsack already configured");
}

fn get_config() -> (&'static [Item], f64) {
    let cfg = CONFIG.get().expect("knapsack not configured");
    (&cfg.items, cfg.capacity)
}

/// Generate random knapsack items
pub fn generate_items(n: usize, rng: &mut impl Rng) -> (Vec<Item>, f64) {
    let items: Vec<Item> = (0..n)
        .map(|_| Item {
            weight: rng.gen::<f64>() * 20.0 + 1.0,
            value: rng.gen::<f64>() * 100.0 + 1.0,
        })
        .collect();

    // Capacity = ~40% of total weight
    let total_weight: f64 = items.iter().map(|i| i.weight).sum();
    let capacity = total_weight * 0.4;

    (items, capacity)
}

/// Binary genome: each bit indicates whether an item is included
#[derive(Debug, Clone)]
pub struct KnapsackGenome {
    pub bits: Vec<bool>,
}

impl KnapsackGenome {
    pub fn total_weight(&self) -> f64 {
        let (items, _) = get_config();
        self.bits
            .iter()
            .enumerate()
            .filter(|(_, &b)| b)
            .map(|(i, _)| items[i].weight)
            .sum()
    }

    pub fn total_value(&self) -> f64 {
        let (items, _) = get_config();
        self.bits
            .iter()
            .enumerate()
            .filter(|(_, &b)| b)
            .map(|(i, _)| items[i].value)
            .sum()
    }
}

impl Genome for KnapsackGenome {
    fn random(rng: &mut impl Rng) -> Self {
        let (items, _) = get_config();
        let bits = (0..items.len()).map(|_| rng.gen_bool(0.3)).collect();
        Self { bits }
    }

    fn fitness(&self) -> f64 {
        let (_, capacity) = get_config();
        let weight = self.total_weight();
        let value = self.total_value();

        if weight > capacity {
            // Penalty: value minus heavy overshoot penalty
            let overflow = weight - capacity;
            (value - overflow * 10.0).max(0.0)
        } else {
            value
        }
    }

    /// Uniform crossover: each bit picked from a random parent
    fn crossover(&self, other: &Self, rng: &mut impl Rng) -> Self {
        let bits = self
            .bits
            .iter()
            .zip(other.bits.iter())
            .map(|(&a, &b)| if rng.gen_bool(0.5) { a } else { b })
            .collect();
        Self { bits }
    }

    /// Bit-flip mutation
    fn mutate(&mut self, mutation_rate: f64, rng: &mut impl Rng) {
        for bit in &mut self.bits {
            if rng.gen::<f64>() < mutation_rate {
                *bit = !*bit;
            }
        }
    }

    fn display(&self) -> String {
        let selected: Vec<usize> = self
            .bits
            .iter()
            .enumerate()
            .filter(|(_, &b)| b)
            .map(|(i, _)| i)
            .collect();
        format!(
            "Value: {:.2} | Weight: {:.2} | Items: {:?}",
            self.total_value(),
            self.total_weight(),
            selected
        )
    }
}
