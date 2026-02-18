#!/usr/bin/env python3
"""Generate sample network traffic data for the anomaly detection system."""
import random
from datetime import datetime, timedelta

random.seed(42)

dst_servers = [
    ("93.184.216.34", 443, "TCP"),
    ("172.217.14.206", 443, "TCP"),
    ("151.101.1.69", 443, "TCP"),
    ("104.244.42.1", 443, "TCP"),
    ("52.94.236.248", 443, "TCP"),
    ("8.8.8.8", 53, "UDP"),
    ("8.8.4.4", 53, "UDP"),
]

src_ips = ["192.168.1.10", "192.168.1.11", "192.168.1.12", "192.168.1.13", "192.168.1.14"]

print("# Sample network traffic data for anomaly detection")
print("# Format: timestamp src_ip src_port dst_ip dst_port protocol bytes duration")

base_time = datetime(2024, 1, 15, 10, 0, 0)

# Generate 350 normal events
for i in range(350):
    ts = base_time + timedelta(seconds=i * 2 + random.uniform(-0.5, 0.5))
    src_ip = random.choice(src_ips)
    src_port = random.randint(49152, 51999)
    dst_ip, dst_port, proto = random.choice(dst_servers)

    if proto == "UDP":  # DNS
        bytes_val = random.randint(50, 120)
        dur = round(random.uniform(0.005, 0.03), 4)
    else:  # HTTPS
        bytes_val = random.randint(500, 3000)
        dur = round(random.uniform(0.02, 0.12), 4)

    print(f"{ts.strftime('%Y-%m-%dT%H:%M:%S')} {src_ip} {src_port} {dst_ip} {dst_port} {proto} {bytes_val} {dur}")

print("#")
print("# ===== ANOMALIES BELOW =====")
print("#")

# ANOMALY 1: Massive data exfiltration
ts = base_time + timedelta(minutes=12)
print(f"{ts.strftime('%Y-%m-%dT%H:%M:%S')} 192.168.1.10 49200 45.33.32.156 8443 TCP 9500000 120.50")

# ANOMALY 2: Port scan (ICMP, unusual ports)
ts = base_time + timedelta(minutes=12, seconds=5)
print(f"{ts.strftime('%Y-%m-%dT%H:%M:%S')} 192.168.1.99 1 10.0.0.1 0 ICMP 28 0.001")

# ANOMALY 3: DNS exfiltration (huge DNS packet)
ts = base_time + timedelta(minutes=12, seconds=10)
print(f"{ts.strftime('%Y-%m-%dT%H:%M:%S')} 192.168.1.10 51250 8.8.8.8 53 UDP 4500000 30.00")

# ANOMALY 4: Connection at 3 AM to high port
ts = datetime(2024, 1, 15, 3, 15, 0)
print(f"{ts.strftime('%Y-%m-%dT%H:%M:%S')} 192.168.1.50 60000 198.51.100.1 31337 TCP 5000000 300.00")

# ANOMALY 5: Zero-length rapid connection to SSH
ts = base_time + timedelta(minutes=12, seconds=20)
print(f"{ts.strftime('%Y-%m-%dT%H:%M:%S')} 192.168.1.10 49300 10.0.0.1 22 TCP 0 0.0001")
