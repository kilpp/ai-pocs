use std::sync::OnceLock;

use rand::Rng;

use crate::genome::Genome;

/// Which mathematical function to optimize
#[derive(Debug, Clone, Copy)]
pub enum TargetFunction {
    /// Rastrigin: highly multimodal, global minimum at origin = 0.0
    /// f(x) = 10n + Σ(xi² - 10*cos(2πxi))
    Rastrigin,
    /// Sphere: simple convex, global minimum at origin = 0.0
    /// f(x) = Σ(xi²)
    Sphere,
    /// Ackley: multimodal with a large basin, global minimum at origin = 0.0
    Ackley,
}

#[derive(Debug)]
struct FnConfig {
    func: TargetFunction,
    dims: usize,
    bound: f64,
}

static CONFIG: OnceLock<FnConfig> = OnceLock::new();

pub fn configure(func: TargetFunction, dims: usize, bound: f64) {
    CONFIG
        .set(FnConfig { func, dims, bound })
        .expect("function optimizer already configured");
}

fn get_config() -> (TargetFunction, usize, f64) {
    let cfg = CONFIG.get().expect("function optimizer not configured");
    (cfg.func, cfg.dims, cfg.bound)
}

fn evaluate(x: &[f64], func: TargetFunction) -> f64 {
    match func {
        TargetFunction::Rastrigin => {
            let n = x.len() as f64;
            let sum: f64 = x
                .iter()
                .map(|&xi| xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos())
                .sum();
            10.0 * n + sum
        }
        TargetFunction::Sphere => x.iter().map(|&xi| xi * xi).sum(),
        TargetFunction::Ackley => {
            let n = x.len() as f64;
            let sum_sq: f64 = x.iter().map(|&xi| xi * xi).sum::<f64>() / n;
            let sum_cos: f64 = x
                .iter()
                .map(|&xi| (2.0 * std::f64::consts::PI * xi).cos())
                .sum::<f64>()
                / n;
            -20.0 * (-0.2 * sum_sq.sqrt()).exp() - sum_cos.exp() + 20.0 + std::f64::consts::E
        }
    }
}

#[derive(Debug, Clone)]
pub struct FunctionGenome {
    pub values: Vec<f64>,
}

impl Genome for FunctionGenome {
    fn random(rng: &mut impl Rng) -> Self {
        let (_, dims, bound) = get_config();
        let values = (0..dims)
            .map(|_| rng.gen::<f64>() * 2.0 * bound - bound)
            .collect();
        Self { values }
    }

    fn fitness(&self) -> f64 {
        let (func, _, _) = get_config();
        // Negate because the functions have minimums and we want to maximize fitness
        -evaluate(&self.values, func)
    }

    /// Blend crossover (BLX-α): offspring gene is a random point between parents, extended by α
    fn crossover(&self, other: &Self, rng: &mut impl Rng) -> Self {
        let alpha = 0.5;
        let (_, _, bound) = get_config();
        let values = self
            .values
            .iter()
            .zip(other.values.iter())
            .map(|(&a, &b)| {
                let min = a.min(b);
                let max = a.max(b);
                let range = max - min;
                let low = min - alpha * range;
                let high = max + alpha * range;
                let v = rng.gen::<f64>() * (high - low) + low;
                v.clamp(-bound, bound)
            })
            .collect();
        Self { values }
    }

    /// Gaussian mutation: add normally-distributed noise to each gene
    fn mutate(&mut self, mutation_rate: f64, rng: &mut impl Rng) {
        let (_, _, bound) = get_config();
        for val in &mut self.values {
            if rng.gen::<f64>() < mutation_rate {
                // Box-Muller transform for normal distribution
                let u1: f64 = rng.gen::<f64>().max(1e-10);
                let u2: f64 = rng.gen::<f64>();
                let normal = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                *val += normal * bound * 0.1;
                *val = val.clamp(-bound, bound);
            }
        }
    }

    fn display(&self) -> String {
        let (func, _, _) = get_config();
        let obj_value = evaluate(&self.values, func);
        let preview: Vec<String> = self.values.iter().take(5).map(|v| format!("{:.4}", v)).collect();
        format!(
            "{:?} value: {:.6} | x = [{}{}]",
            func,
            obj_value,
            preview.join(", "),
            if self.values.len() > 5 { ", ..." } else { "" }
        )
    }
}
