import argparse
import time
from pathlib import Path

import httpx
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "model/data/creditcard.csv"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--name", default="python")
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--mode", choices=["single", "batch"], default="single")
    return p.parse_args()


def load_samples(n, mode):
    df = pd.read_csv(DATA_DIR)
    test_data = df.sample(n, random_state=42).drop(columns=["Class"])
    match mode:
        case "single":
            payload = [{"features": row.to_dict()} for _, row in test_data.iterrows()]
        case "batch":
            payload = [{"rows": test_data.values.tolist()}]
        case _:
            raise ValueError("Wrong bench mode")
    return payload


def warmup(client, url, payloads, k):
    for _ in range(k):
        for payload in payloads:
            client.post(url, json=payload)


def measure(client, url, payloads):
    latencies = []
    for payload in payloads:
        start = time.perf_counter()
        r = client.post(url, json=payload)
        latencies.append(time.perf_counter() - start)
        if r.status_code != 200:
            raise RuntimeError(f"{r.status_code}: {r.text}")
    return latencies


def bench():
    args = parse_args()

    endpoints = ["/predict", "/predict/batch"]

    payload = load_samples(100, args.mode)

    with httpx.Client() as client:
        warmup(client, f"{args.url}{endpoints[0]}", payload, args.warmup)
        latencies = measure(client, f"{args.url}{endpoints[0]}", payload)
        print(latencies)


if __name__ == "__main__":
    bench()
