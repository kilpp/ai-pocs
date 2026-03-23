use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use crate::genome::Genome;
use crate::selection::SelectionStrategy;
use crate::stats::GenerationStats;

pub struct GaConfig {
    pub population_size: usize,
    pub generations: usize,
    pub mutation_rate: f64,
    pub crossover_rate: f64,
    pub elitism_count: usize,
    pub selection: SelectionStrategy,
    pub seed: Option<u64>,
}

impl Default for GaConfig {
    fn default() -> Self {
        Self {
            population_size: 200,
            generations: 500,
            mutation_rate: 0.05,
            crossover_rate: 0.8,
            elitism_count: 2,
            selection: SelectionStrategy::Tournament(3),
            seed: None,
        }
    }
}

pub struct EvolutionResult<G: Genome> {
    pub best: G,
    pub best_fitness: f64,
    pub history: Vec<GenerationStats>,
}

pub fn run<G: Genome>(config: &GaConfig) -> EvolutionResult<G> {
    let mut rng = match config.seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::from_entropy(),
    };

    // Initialize population
    let mut population: Vec<G> = (0..config.population_size)
        .map(|_| G::random(&mut rng))
        .collect();

    let mut history: Vec<GenerationStats> = Vec::with_capacity(config.generations);
    let mut global_best: Option<G> = None;
    let mut global_best_fitness = f64::NEG_INFINITY;

    for gen in 0..config.generations {
        // Evaluate fitness in parallel
        let fitnesses: Vec<f64> = population.par_iter().map(|g| g.fitness()).collect();

        // Collect stats
        let stats = GenerationStats::from_fitnesses(gen, &fitnesses);

        // Track global best
        for (i, &fit) in fitnesses.iter().enumerate() {
            if fit > global_best_fitness {
                global_best_fitness = fit;
                global_best = Some(population[i].clone());
            }
        }

        // Print progress every 10% or on last generation
        if gen % (config.generations / 10).max(1) == 0 || gen == config.generations - 1 {
            println!(
                "Gen {:>5} | best: {:>12.4} | avg: {:>12.4} | worst: {:>12.4}",
                gen, stats.best_fitness, stats.avg_fitness, stats.worst_fitness
            );
        }

        history.push(stats);

        // Sort population by fitness (descending) for elitism
        let mut indexed: Vec<(usize, f64)> =
            fitnesses.iter().enumerate().map(|(i, &f)| (i, f)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut next_gen: Vec<G> = Vec::with_capacity(config.population_size);

        // Elitism: carry over top individuals unchanged
        for &(idx, _) in indexed.iter().take(config.elitism_count) {
            next_gen.push(population[idx].clone());
        }

        // Fill the rest with crossover + mutation
        while next_gen.len() < config.population_size {
            let parent_a = config.selection.select(&population, &mut rng);
            let parent_b = config.selection.select(&population, &mut rng);

            let mut child = if rng.gen::<f64>() < config.crossover_rate {
                parent_a.crossover(parent_b, &mut rng)
            } else {
                parent_a.clone()
            };

            child.mutate(config.mutation_rate, &mut rng);
            next_gen.push(child);
        }

        population = next_gen;
    }

    EvolutionResult {
        best: global_best.unwrap(),
        best_fitness: global_best_fitness,
        history,
    }
}
