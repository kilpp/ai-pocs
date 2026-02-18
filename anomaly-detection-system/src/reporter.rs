use crate::parser::NetworkEvent;
use colored::Colorize;
use serde::Serialize;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

#[derive(Debug, Serialize)]
pub struct AnomalyReport {
    pub event: NetworkEvent,
    pub score: f64,
    pub event_number: usize,
}

/// Print an anomaly to the terminal with colored output.
pub fn print_anomaly(report: &AnomalyReport) {
    let severity = if report.score >= 0.8 {
        "HIGH".red().bold()
    } else if report.score >= 0.7 {
        "MEDIUM".yellow().bold()
    } else {
        "LOW".yellow()
    };

    let event = &report.event;
    eprintln!(
        "{} [{}] #{} | {}:{} -> {}:{} | {} | {} bytes | {:.3}s | score: {:.4}",
        "[ANOMALY]".red().bold(),
        severity,
        report.event_number,
        event.src_ip,
        event.src_port,
        event.dst_ip,
        event.dst_port,
        event.protocol,
        event.bytes,
        event.duration,
        report.score,
    );
}

/// Append an anomaly report to a JSON file.
///
/// Each anomaly is written as a single JSON object per line (JSON Lines format).
pub fn write_json(report: &AnomalyReport, output_path: &Path) -> std::io::Result<()> {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(output_path)?;

    let json = serde_json::to_string(report)?;
    writeln!(file, "{}", json)?;
    Ok(())
}

/// Print a summary of the detection session.
pub fn print_summary(total_events: usize, total_anomalies: usize) {
    let rate = if total_events > 0 {
        (total_anomalies as f64 / total_events as f64) * 100.0
    } else {
        0.0
    };

    eprintln!();
    eprintln!("{}", "=== Detection Summary ===".bold());
    eprintln!("Total events processed: {}", total_events);
    eprintln!("Anomalies detected:     {}", total_anomalies);
    eprintln!("Anomaly rate:           {:.2}%", rate);
}

/// Print a status update during processing.
pub fn print_status(total_events: usize, total_anomalies: usize, trained: bool) {
    let status = if trained {
        "detecting".green()
    } else {
        "buffering".yellow()
    };
    eprintln!(
        "[STATUS] {} | events: {} | anomalies: {}",
        status, total_events, total_anomalies
    );
}
