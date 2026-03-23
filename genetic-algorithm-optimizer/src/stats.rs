use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct GenerationStats {
    pub generation: usize,
    pub best_fitness: f64,
    pub worst_fitness: f64,
    pub avg_fitness: f64,
    pub std_dev: f64,
}

impl GenerationStats {
    pub fn from_fitnesses(generation: usize, fitnesses: &[f64]) -> Self {
        let n = fitnesses.len() as f64;
        let best = fitnesses.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let worst = fitnesses.iter().cloned().fold(f64::INFINITY, f64::min);
        let sum: f64 = fitnesses.iter().sum();
        let avg = sum / n;
        let variance: f64 = fitnesses.iter().map(|f| (f - avg).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();

        Self {
            generation,
            best_fitness: best,
            worst_fitness: worst,
            avg_fitness: avg,
            std_dev,
        }
    }
}

/// Print an ASCII chart of fitness over generations
pub fn print_fitness_chart(history: &[GenerationStats]) {
    if history.is_empty() {
        return;
    }

    let width = 60;
    let height = 15;

    let best_vals: Vec<f64> = history.iter().map(|s| s.best_fitness).collect();
    let avg_vals: Vec<f64> = history.iter().map(|s| s.avg_fitness).collect();

    let all_min = best_vals
        .iter()
        .chain(avg_vals.iter())
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let all_max = best_vals
        .iter()
        .chain(avg_vals.iter())
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    let range = if (all_max - all_min).abs() < 1e-10 {
        1.0
    } else {
        all_max - all_min
    };

    // Sample points to fit the chart width
    let sample = |vals: &[f64]| -> Vec<usize> {
        (0..width)
            .map(|col| {
                let idx = (col as f64 / width as f64 * vals.len() as f64) as usize;
                let idx = idx.min(vals.len() - 1);
                let normalized = (vals[idx] - all_min) / range;
                (normalized * (height - 1) as f64).round() as usize
            })
            .collect()
    };

    let best_rows = sample(&best_vals);
    let avg_rows = sample(&avg_vals);

    println!("\n  Fitness over generations");
    println!("  {:>10.2} ┤", all_max);

    for row in (0..height).rev() {
        if row == 0 {
            print!("  {:>10.2} ┤", all_min);
        } else if row == height / 2 {
            let mid = all_min + range / 2.0;
            print!("  {:>10.2} ┤", mid);
        } else {
            print!("             │");
        }

        for col in 0..width {
            if best_rows[col] == row && avg_rows[col] == row {
                print!("█");
            } else if best_rows[col] == row {
                print!("*");
            } else if avg_rows[col] == row {
                print!("·");
            } else {
                print!(" ");
            }
        }
        println!();
    }

    println!(
        "             └{}┘",
        "─".repeat(width)
    );
    println!(
        "              Gen 0{:>width$}",
        format!("Gen {}", history.len() - 1),
        width = width - 5
    );
    println!("              * = best   · = average\n");
}
