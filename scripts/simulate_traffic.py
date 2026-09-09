"""Send bursts of varied /predict requests against the API to populate
dashboards (Grafana/Prometheus) with a realistic, wavy traffic pattern
instead of a single flat spike.

Usage:
    python scripts/simulate_traffic.py --url http://<alb-dns-or-host>
    python scripts/simulate_traffic.py --url http://localhost:8000
"""

import argparse
import json
import random
import threading
import time
import urllib.request

CARRIERS = ["AA", "DL", "UA", "WN", "B6", "AS", "NK", "F9", "HA", "G4"]
AIRPORTS = [
    ("JFK", "NY"), ("LAX", "CA"), ("ORD", "IL"), ("ATL", "GA"),
    ("DFW", "TX"), ("DEN", "CO"), ("SFO", "CA"), ("MIA", "FL"),
    ("SEA", "WA"), ("BOS", "MA"), ("PHX", "AZ"), ("IAH", "TX"),
    ("CLT", "NC"), ("MCO", "FL"), ("EWR", "NJ"),
]

# (number of requests, ~seconds between requests within the wave)
WAVES = [(5, 0.15), (25, 0.15), (10, 0.2), (40, 0.1),
         (15, 0.2), (30, 0.12), (8, 0.25), (20, 0.15)]


def random_flight():
    origin, origin_state = random.choice(AIRPORTS)
    dest, dest_state = random.choice(AIRPORTS)
    return {
        "FL_DATE": time.strftime("%Y-%m-%d"),
        "OP_UNIQUE_CARRIER": random.choice(CARRIERS),
        "ORIGIN": origin,
        "DEST": dest,
        "ORIGIN_STATE_ABR": origin_state,
        "DEST_STATE_ABR": dest_state,
        "DISTANCE": random.randint(150, 2950),
    }


def send_one(url: str, timeout: float) -> None:
    payload = json.dumps({"flights": [random_flight()]}).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=timeout)
    except Exception:
        pass  # a handful of dropped requests don't matter for a traffic demo


def run_wave(url: str, count: int, gap: float, timeout: float) -> None:
    threads = []
    for _ in range(count):
        t = threading.Thread(target=send_one, args=(url, timeout))
        t.start()
        threads.append(t)
        time.sleep(gap)
    for t in threads:
        t.join()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="API base URL, e.g. http://localhost:8000")
    parser.add_argument("--timeout", type=float, default=5.0, help="Per-request timeout (s)")
    parser.add_argument("--min-pause", type=float, default=15.0, help="Min seconds between waves")
    parser.add_argument("--max-pause", type=float, default=40.0, help="Max seconds between waves")
    args = parser.parse_args()

    total = 0
    for count, gap in WAVES:
        run_wave(args.url, count, gap, args.timeout)
        total += count
        print(f"wave done, total sent: {total}")
        time.sleep(random.uniform(args.min_pause, args.max_pause))

    print(f"ALL DONE total={total}")


if __name__ == "__main__":
    main()
