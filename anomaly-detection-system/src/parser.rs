use chrono::NaiveDateTime;
use serde::Serialize;
use std::fmt;

#[derive(Debug, Clone, Serialize)]
pub struct NetworkEvent {
    pub timestamp: NaiveDateTime,
    pub src_ip: String,
    pub src_port: u16,
    pub dst_ip: String,
    pub dst_port: u16,
    pub protocol: Protocol,
    pub bytes: u64,
    pub duration: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Protocol {
    Tcp,
    Udp,
    Icmp,
    Other,
}

impl Protocol {
    pub fn as_f64(self) -> f64 {
        match self {
            Protocol::Tcp => 0.0,
            Protocol::Udp => 1.0,
            Protocol::Icmp => 2.0,
            Protocol::Other => 3.0,
        }
    }
}

impl fmt::Display for Protocol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Protocol::Tcp => write!(f, "TCP"),
            Protocol::Udp => write!(f, "UDP"),
            Protocol::Icmp => write!(f, "ICMP"),
            Protocol::Other => write!(f, "OTHER"),
        }
    }
}

fn parse_protocol(s: &str) -> Protocol {
    match s.to_uppercase().as_str() {
        "TCP" => Protocol::Tcp,
        "UDP" => Protocol::Udp,
        "ICMP" => Protocol::Icmp,
        _ => Protocol::Other,
    }
}

/// Parse a single line of network traffic data.
///
/// Expected format (space-separated):
/// `timestamp src_ip src_port dst_ip dst_port protocol bytes duration`
///
/// Example:
/// `2024-01-15T10:30:00 192.168.1.10 54321 10.0.0.1 443 TCP 1500 0.05`
pub fn parse_line(line: &str) -> Option<NetworkEvent> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
        return None;
    }

    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 8 {
        return None;
    }

    let timestamp = NaiveDateTime::parse_from_str(parts[0], "%Y-%m-%dT%H:%M:%S").ok()?;
    let src_ip = parts[1].to_string();
    let src_port: u16 = parts[2].parse().ok()?;
    let dst_ip = parts[3].to_string();
    let dst_port: u16 = parts[4].parse().ok()?;
    let protocol = parse_protocol(parts[5]);
    let bytes: u64 = parts[6].parse().ok()?;
    let duration: f64 = parts[7].parse().ok()?;

    Some(NetworkEvent {
        timestamp,
        src_ip,
        src_port,
        dst_ip,
        dst_port,
        protocol,
        bytes,
        duration,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_valid_line() {
        let line = "2024-01-15T10:30:00 192.168.1.10 54321 10.0.0.1 443 TCP 1500 0.05";
        let event = parse_line(line).unwrap();
        assert_eq!(event.src_ip, "192.168.1.10");
        assert_eq!(event.dst_port, 443);
        assert_eq!(event.protocol, Protocol::Tcp);
        assert_eq!(event.bytes, 1500);
    }

    #[test]
    fn test_parse_comment_line() {
        assert!(parse_line("# this is a comment").is_none());
    }

    #[test]
    fn test_parse_empty_line() {
        assert!(parse_line("").is_none());
        assert!(parse_line("   ").is_none());
    }

    #[test]
    fn test_parse_malformed_line() {
        assert!(parse_line("not enough fields").is_none());
    }

    #[test]
    fn test_parse_udp() {
        let line = "2024-01-15T10:30:00 10.0.0.1 12345 10.0.0.2 53 UDP 64 0.01";
        let event = parse_line(line).unwrap();
        assert_eq!(event.protocol, Protocol::Udp);
    }
}
