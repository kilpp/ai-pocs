use rand::Rng;

use crate::genome::Genome;

#[derive(Debug, Clone, Copy)]
pub enum SelectionStrategy {
    /// Pick the best out of a random subset of the population
    Tournament(usize),
    /// Probability proportional to fitness (shifted so minimum fitness maps to a small positive value)
    RouletteWheel,
    /// Probability proportional to rank (not raw fitness)
    RankBased,
}

impl SelectionStrategy {
    pub fn select<'a, G: Genome>(&self, population: &'a [G], rng: &mut impl Rng) -> &'a G {
        match self {
            SelectionStrategy::Tournament(k) => tournament(population, *k, rng),
            SelectionStrategy::RouletteWheel => roulette_wheel(population, rng),
            SelectionStrategy::RankBased => rank_based(population, rng),
        }
    }
}

fn tournament<'a, G: Genome>(pop: &'a [G], k: usize, rng: &mut impl Rng) -> &'a G {
    let k = k.max(2).min(pop.len());
    let mut best_idx = rng.gen_range(0..pop.len());
    let mut best_fit = pop[best_idx].fitness();

    for _ in 1..k {
        let idx = rng.gen_range(0..pop.len());
        let fit = pop[idx].fitness();
        if fit > best_fit {
            best_idx = idx;
            best_fit = fit;
        }
    }
    &pop[best_idx]
}

fn roulette_wheel<'a, G: Genome>(pop: &'a [G], rng: &mut impl Rng) -> &'a G {
    let fitnesses: Vec<f64> = pop.iter().map(|g| g.fitness()).collect();
    let min_fit = fitnesses.iter().cloned().fold(f64::INFINITY, f64::min);
    // Shift so all values are positive, add small epsilon to avoid zero-probability
    let shifted: Vec<f64> = fitnesses.iter().map(|f| f - min_fit + 1e-6).collect();
    let total: f64 = shifted.iter().sum();
    let mut spin = rng.gen::<f64>() * total;

    for (i, s) in shifted.iter().enumerate() {
        spin -= s;
        if spin <= 0.0 {
            return &pop[i];
        }
    }
    &pop[pop.len() - 1]
}

fn rank_based<'a, G: Genome>(pop: &'a [G], rng: &mut impl Rng) -> &'a G {
    let mut indices: Vec<usize> = (0..pop.len()).collect();
    indices.sort_by(|a, b| {
        pop[*a]
            .fitness()
            .partial_cmp(&pop[*b].fitness())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Rank 1 = worst, rank N = best
    let n = pop.len() as f64;
    let total = n * (n + 1.0) / 2.0;
    let mut spin = rng.gen::<f64>() * total;

    for (rank_minus_1, &idx) in indices.iter().enumerate() {
        let rank = (rank_minus_1 + 1) as f64;
        spin -= rank;
        if spin <= 0.0 {
            return &pop[idx];
        }
    }
    &pop[*indices.last().unwrap()]
}
