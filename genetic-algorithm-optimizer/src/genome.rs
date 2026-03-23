use rand::Rng;

/// Trait that defines a genome (individual) in the genetic algorithm.
/// Implement this trait to define your own optimization problem.
pub trait Genome: Clone + Send + Sync {
    /// Create a random individual
    fn random(rng: &mut impl Rng) -> Self;

    /// Evaluate the fitness of this individual (higher is better)
    fn fitness(&self) -> f64;

    /// Perform crossover between two parents to produce an offspring
    fn crossover(&self, other: &Self, rng: &mut impl Rng) -> Self;

    /// Mutate this individual in place
    fn mutate(&mut self, mutation_rate: f64, rng: &mut impl Rng);

    /// Human-readable description of this individual
    fn display(&self) -> String;
}
