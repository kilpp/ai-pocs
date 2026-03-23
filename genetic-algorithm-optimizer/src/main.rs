mod engine;
mod genome;
mod problems;
mod selection;
mod stats;

use clap::{Parser, ValueEnum};
use genome::Genome;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use engine::GaConfig;
use problems::function_optimization::{FunctionGenome, TargetFunction};
use problems::knapsack::{self, KnapsackGenome};
use problems::tsp::{self, TspGenome};
use selection::SelectionStrategy;
use stats::print_fitness_chart;

#[derive(Debug, Clone, ValueEnum)]
enum Problem {
    /// Traveling Salesman Problem
    Tsp,
    /// Mathematical function minimization (Rastrigin, Sphere, Ackley)
    Function,
    /// 0/1 Knapsack Problem
    Knapsack,
}

#[derive(Debug, Clone, ValueEnum)]
enum Selection {
    Tournament,
    Roulette,
    Rank,
}

#[derive(Debug, Clone, ValueEnum)]
enum Function {
    Rastrigin,
    Sphere,
    Ackley,
}

#[derive(Parser, Debug)]
#[command(name = "genetic-algorithm-optimizer")]
#[command(about = "Solve optimization problems using genetic algorithms")]
struct Cli {
    /// Problem to solve
    #[arg(short, long, default_value = "tsp")]
    problem: Problem,

    /// Population size
    #[arg(long, default_value = "200")]
    pop_size: usize,

    /// Number of generations
    #[arg(short, long, default_value = "500")]
    generations: usize,

    /// Mutation rate (0.0 - 1.0)
    #[arg(short, long, default_value = "0.05")]
    mutation_rate: f64,

    /// Crossover rate (0.0 - 1.0)
    #[arg(short, long, default_value = "0.8")]
    crossover_rate: f64,

    /// Number of elite individuals preserved each generation
    #[arg(short, long, default_value = "2")]
    elitism: usize,

    /// Selection strategy
    #[arg(short, long, default_value = "tournament")]
    selection: Selection,

    /// Tournament size (only for tournament selection)
    #[arg(long, default_value = "3")]
    tournament_size: usize,

    /// Random seed for reproducibility
    #[arg(long)]
    seed: Option<u64>,

    /// Number of cities (TSP) or items (Knapsack) or dimensions (Function)
    #[arg(short = 'n', long, default_value = "30")]
    size: usize,

    /// Target function for function optimization
    #[arg(long, default_value = "rastrigin")]
    function: Function,

    /// Export evolution history as JSON
    #[arg(long)]
    export_json: Option<String>,
}

fn main() {
    let cli = Cli::parse();

    let selection = match cli.selection {
        Selection::Tournament => SelectionStrategy::Tournament(cli.tournament_size),
        Selection::Roulette => SelectionStrategy::RouletteWheel,
        Selection::Rank => SelectionStrategy::RankBased,
    };

    let config = GaConfig {
        population_size: cli.pop_size,
        generations: cli.generations,
        mutation_rate: cli.mutation_rate,
        crossover_rate: cli.crossover_rate,
        elitism_count: cli.elitism,
        selection,
        seed: cli.seed,
    };

    let seed = cli.seed.unwrap_or(42);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    match cli.problem {
        Problem::Tsp => {
            println!("=== Traveling Salesman Problem ({} cities) ===\n", cli.size);
            let cities = tsp::generate_cities(cli.size, &mut rng);
            tsp::set_cities(cities);

            let result = engine::run::<TspGenome>(&config);

            println!("\n--- Result ---");
            println!("Best fitness: {:.4}", result.best_fitness);
            println!("{}", result.best.display());

            print_fitness_chart(&result.history);

            if let Some(path) = &cli.export_json {
                export_history(&result.history, path);
            }
        }
        Problem::Function => {
            let target = match cli.function {
                Function::Rastrigin => TargetFunction::Rastrigin,
                Function::Sphere => TargetFunction::Sphere,
                Function::Ackley => TargetFunction::Ackley,
            };
            let bound = match target {
                TargetFunction::Rastrigin => 5.12,
                TargetFunction::Sphere => 5.0,
                TargetFunction::Ackley => 5.0,
            };

            println!(
                "=== Function Optimization: {:?} ({} dimensions) ===\n",
                target, cli.size
            );
            problems::function_optimization::configure(target, cli.size, bound);

            let result = engine::run::<FunctionGenome>(&config);

            println!("\n--- Result ---");
            println!("Best fitness: {:.6} (objective: {:.6})", result.best_fitness, -result.best_fitness);
            println!("{}", result.best.display());

            print_fitness_chart(&result.history);

            if let Some(path) = &cli.export_json {
                export_history(&result.history, path);
            }
        }
        Problem::Knapsack => {
            println!("=== 0/1 Knapsack Problem ({} items) ===\n", cli.size);
            let (items, capacity) = knapsack::generate_items(cli.size, &mut rng);
            println!("Capacity: {:.2}", capacity);
            println!("Items:");
            for (i, item) in items.iter().enumerate() {
                println!(
                    "  [{:>2}] weight: {:>6.2}, value: {:>6.2}, ratio: {:.2}",
                    i,
                    item.weight,
                    item.value,
                    item.value / item.weight
                );
            }
            println!();
            knapsack::configure(items, capacity);

            let result = engine::run::<KnapsackGenome>(&config);

            println!("\n--- Result ---");
            println!("Best fitness: {:.2}", result.best_fitness);
            println!("{}", result.best.display());
            let (_, cap) = (result.best.total_weight(), capacity);
            println!("Capacity used: {:.2} / {:.2}", result.best.total_weight(), cap);

            print_fitness_chart(&result.history);

            if let Some(path) = &cli.export_json {
                export_history(&result.history, path);
            }
        }
    }
}

fn export_history(history: &[stats::GenerationStats], path: &str) {
    match serde_json::to_string_pretty(history) {
        Ok(json) => match std::fs::write(path, json) {
            Ok(_) => println!("History exported to {}", path),
            Err(e) => eprintln!("Failed to write {}: {}", path, e),
        },
        Err(e) => eprintln!("Failed to serialize history: {}", e),
    }
}
