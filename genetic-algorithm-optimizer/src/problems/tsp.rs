use std::sync::OnceLock;

use rand::seq::SliceRandom;
use rand::Rng;

use crate::genome::Genome;

/// A city with (x, y) coordinates
#[derive(Debug, Clone, Copy)]
pub struct City {
    pub x: f64,
    pub y: f64,
}

static CITIES: OnceLock<Vec<City>> = OnceLock::new();

pub fn set_cities(cities: Vec<City>) {
    CITIES.set(cities).expect("cities already initialized");
}

fn get_cities() -> &'static [City] {
    CITIES.get().expect("cities not initialized")
}

/// Generate N random cities in a [0, 100] x [0, 100] grid
pub fn generate_cities(n: usize, rng: &mut impl Rng) -> Vec<City> {
    (0..n)
        .map(|_| City {
            x: rng.gen::<f64>() * 100.0,
            y: rng.gen::<f64>() * 100.0,
        })
        .collect()
}

fn distance(a: &City, b: &City) -> f64 {
    ((a.x - b.x).powi(2) + (a.y - b.y).powi(2)).sqrt()
}

/// A tour represented as a permutation of city indices
#[derive(Debug, Clone)]
pub struct TspGenome {
    pub tour: Vec<usize>,
}

impl TspGenome {
    pub fn total_distance(&self) -> f64 {
        let cities = get_cities();
        let mut dist = 0.0;
        for i in 0..self.tour.len() {
            let from = &cities[self.tour[i]];
            let to = &cities[self.tour[(i + 1) % self.tour.len()]];
            dist += distance(from, to);
        }
        dist
    }
}

impl Genome for TspGenome {
    fn random(rng: &mut impl Rng) -> Self {
        let n = get_cities().len();
        let mut tour: Vec<usize> = (0..n).collect();
        tour.shuffle(rng);
        Self { tour }
    }

    fn fitness(&self) -> f64 {
        // Negate distance so shorter tours have higher fitness
        -self.total_distance()
    }

    /// Order crossover (OX): preserves relative order from both parents
    fn crossover(&self, other: &Self, rng: &mut impl Rng) -> Self {
        let n = self.tour.len();
        let start = rng.gen_range(0..n);
        let end = rng.gen_range(start + 1..=n);

        let mut child = vec![usize::MAX; n];

        // Copy segment from parent A
        for i in start..end {
            child[i] = self.tour[i];
        }

        // Fill remaining positions with cities from parent B in order
        let mut pos = end % n;
        for &city in other.tour.iter().cycle().skip(end).take(n) {
            if !child.contains(&city) {
                child[pos] = city;
                pos = (pos + 1) % n;
            }
        }

        Self { tour: child }
    }

    /// Swap mutation: swap two random cities in the tour
    fn mutate(&mut self, mutation_rate: f64, rng: &mut impl Rng) {
        if rng.gen::<f64>() < mutation_rate {
            let n = self.tour.len();
            let i = rng.gen_range(0..n);
            let j = rng.gen_range(0..n);
            self.tour.swap(i, j);
        }
        // Also try a 2-opt reversal with lower probability
        if rng.gen::<f64>() < mutation_rate * 0.5 {
            let n = self.tour.len();
            let mut i = rng.gen_range(0..n);
            let mut j = rng.gen_range(0..n);
            if i > j {
                std::mem::swap(&mut i, &mut j);
            }
            self.tour[i..=j].reverse();
        }
    }

    fn display(&self) -> String {
        let dist = self.total_distance();
        format!(
            "Tour distance: {:.2} | Route: {:?}",
            dist,
            &self.tour[..self.tour.len().min(20)]
        )
    }
}
