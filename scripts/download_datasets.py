"""Dataset download and preprocessing utility.

SWaT/WADI: Requires iTrust registration at https://www.itrust.sutd.edu.sg/
CICIoT2023: https://www.unb.ca/cic/datasets/iot-2023.html (direct CSV download)
Edge-IIoTSet: https://data.mendeley.com/datasets/y7hy5bcd59 (Mendeley Data)
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.settings import DATA_ROOT, RANDOM_SEED


SWAT_SOURCE = "https://www.itrust.sutd.edu.sg/itrust-labs_datasets/untad/dataset/"
WADI_SOURCE = "https://www.itrust.sutd.edu.sg/itrust-labs_datasets/untad/dataset/"
CIC_SOURCE = "https://www.unb.ca/cic/datasets/iot-2023.html"
EDGE_SOURCE = "https://data.mendeley.com/datasets/y7hy5bcd59/1"


def _find_csv_files(directory: Path) -> list[Path]:
    csvs = sorted(directory.rglob("*.csv"))
    if not csvs:
        csvs = sorted(directory.rglob("*.CSV"))
    return csvs


def _synthetic_swat(n: int = 50000, rng: np.random.RandomState | None = None) -> pd.DataFrame:
    rng = rng or np.random.RandomState(RANDOM_SEED)
    t = pd.date_range("2015-12-22", periods=n, freq="s")
    labels = np.zeros(n, dtype=int)
    attack_starts = rng.choice(range(n - 500), 8, replace=False)
    for s in attack_starts:
        length = rng.randint(100, 500)
        labels[s : s + length] = 1
    return pd.DataFrame({
        "Timestamp": t,
        "FIT101": rng.normal(2.5, 0.3, n) + rng.normal(0, 0.1, n),
        "LIT101": rng.normal(500, 20, n),
        "FIT102": rng.normal(2.4, 0.3, n),
        "LIT102": rng.normal(300, 15, n),
        "FIT103": rng.normal(1.2, 0.2, n),
        "LIT103": rng.normal(600, 25, n),
        "FIT104": rng.normal(2.3, 0.3, n),
        "LIT104": rng.normal(400, 18, n),
        "P101": rng.choice([0, 1], n, p=[0.3, 0.7]),
        "P102": rng.choice([0, 1], n, p=[0.3, 0.7]),
        "MV101": rng.choice([0, 1], n, p=[0.1, 0.9]),
        "MV201": rng.choice([0, 1], n, p=[0.1, 0.9]),
        "MV301": rng.choice([0, 1], n, p=[0.1, 0.9]),
        "MV401": rng.choice([0, 1], n, p=[0.1, 0.9]),
        "P101_Freq": rng.uniform(45, 55, n),
        "P102_Freq": rng.uniform(45, 55, n),
        "attack_label": labels,
    })


def _synthetic_wadi(n: int = 40000, rng: np.random.RandomState | None = None) -> pd.DataFrame:
    rng = rng or np.random.RandomState(RANDOM_SEED + 1)
    t = pd.date_range("2017-10-27", periods=n, freq="s")
    labels = np.zeros(n, dtype=int)
    attack_starts = rng.choice(range(n - 300), 5, replace=False)
    for s in attack_starts:
        length = rng.randint(50, 300)
        labels[s : s + length] = 1
    return pd.DataFrame({
        "Timestamp": t,
        "F1001": rng.normal(4.5, 0.5, n),
        "F1002": rng.normal(3.8, 0.4, n),
        "F1003": rng.normal(5.2, 0.6, n),
        "L1001": rng.normal(450, 30, n),
        "L1002": rng.normal(380, 25, n),
        "L1003": rng.normal(520, 35, n),
        "L1004": rng.normal(290, 20, n),
        "P1001": rng.choice([0, 1], n, p=[0.4, 0.6]),
        "P1002": rng.choice([0, 1], n, p=[0.4, 0.6]),
        "P1003": rng.choice([0, 1], n, p=[0.4, 0.6]),
        "MV1001": rng.choice([0, 1], n, p=[0.1, 0.9]),
        "MV1002": rng.choice([0, 1], n, p=[0.1, 0.9]),
        "MV1003": rng.choice([0, 1], n, p=[0.1, 0.9]),
        "attack_label": labels,
    })


def _synthetic_ciciot(n: int = 50000, rng: np.random.RandomState | None = None) -> pd.DataFrame:
    rng = rng or np.random.RandomState(RANDOM_SEED + 2)
    labels = rng.choice(
        ["Benign", "DDoS_UDP", "DDoS_TCP", "DoS", "BruteForce", "PortScan"],
        n, p=[0.45, 0.15, 0.12, 0.13, 0.08, 0.07],
    )
    return pd.DataFrame({
        "flow_duration": np.abs(rng.exponential(5, n)),
        "packet_count": rng.randint(1, 10000, n),
        "byte_count": rng.randint(100, 1000000, n),
        "packet_rate": np.abs(rng.exponential(50, n)),
        "byte_rate": np.abs(rng.exponential(5000, n)),
        "flow_iat_mean": np.abs(rng.exponential(1, n)),
        "flow_iat_std": np.abs(rng.exponential(0.5, n)),
        "fwd_iat_mean": np.abs(rng.exponential(1.2, n)),
        "bwd_iat_mean": np.abs(rng.exponential(1.5, n)),
        "fwd_pkt_len_mean": rng.uniform(20, 1500, n),
        "bwd_pkt_len_mean": rng.uniform(20, 1500, n),
        "fwd_header_len": rng.randint(20, 200, n),
        "bwd_header_len": rng.randint(20, 200, n),
        "fin_flag_count": rng.randint(0, 5, n),
        "syn_flag_count": rng.randint(0, 10, n),
        "rst_flag_count": rng.randint(0, 5, n),
        "ack_flag_count": rng.randint(0, 20, n),
        "psh_flag_count": rng.randint(0, 10, n),
        "src_port": rng.randint(1024, 65535, n),
        "dst_port": rng.choice([80, 443, 22, 53, 8080, 8443], n),
        "protocol": rng.choice([6, 17, 1], n),
        "attack_label": labels,
    })


def _synthetic_edge_iiot(n: int = 50000, rng: np.random.RandomState | None = None) -> pd.DataFrame:
    rng = rng or np.random.RandomState(RANDOM_SEED + 3)
    labels = rng.choice(
        ["Normal", "DDoS", "DoS", "Recon", "Backdoor", "XSS", "MitM"],
        n, p=[0.50, 0.12, 0.12, 0.10, 0.06, 0.05, 0.05],
    )
    return pd.DataFrame({
        "frame.time_delta": np.abs(rng.exponential(0.01, n)),
        "frame.len": rng.randint(60, 1514, n),
        "ip.proto": rng.choice([6, 17, 1], n),
        "tcp.srcport": rng.randint(1024, 65535, n),
        "tcp.dstport": rng.choice([80, 443, 22, 53, 8080], n),
        "tcp.flags": rng.randint(0, 47, n),
        "tcp.window_size": rng.choice([256, 512, 1024, 2048, 4096, 8192, 16384, 65535], n),
        "ip.ttl": rng.choice([32, 64, 128, 255], n),
        "ip.len": rng.randint(20, 1500, n),
        "icmp.type": rng.choice([0, 3, 8, 11], n),
        "mqtt.topic_len": rng.randint(0, 200, n),
        "mqtt.msg_len": rng.randint(0, 1000, n),
        "mqtt.qos": rng.choice([0, 1, 2], n),
        "attack_label": labels,
    })


def generate_synthetic(dataset: str, output_dir: Path) -> Path:
    rng = np.random.RandomState(RANDOM_SEED)
    generators = {
        "swat": _synthetic_swat,
        "wadi": _synthetic_wadi,
        "ciciot2023": _synthetic_ciciot,
        "edge_iiotset": _synthetic_edge_iiot,
    }
    gen_func = generators[dataset]
    df = gen_func(rng=rng)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{dataset}_synthetic.csv"
    df.to_csv(out, index=False)
    print(f"Generated {dataset}: {len(df)} rows, {len(df.columns)} cols -> {out}")
    return out


def generate_all_synthetic(output_base: Path | None = None) -> dict[str, Path]:
    output_base = output_base or DATA_ROOT / "raw"
    results = {}
    for ds in ["swat", "wadi", "ciciot2023", "edge_iiotset"]:
        results[ds] = generate_synthetic(ds, output_base / ds)
    return results


def show_instructions():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    DATASET DOWNLOAD INSTRUCTIONS                    ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  1. SWaT (Secure Water Treatment)                                    ║
║     Source: https://www.itrust.sutd.edu.sg/itrust-labs_datasets/      ║
║     Action: Register → Accept DDoS agreement → Download SWaT-A2     ║
║     Files:  SWaT-Dataset-Stable v1/ (or SWaT-A2.csv)                ║
║     Place:  data/raw/swat/SWaT-A2.csv                                ║
║     Shape:  ~509K rows × 51 columns (49 sensors + timestamp + label)║
║                                                                      ║
║  2. WADI (Water Distribution)                                        ║
║     Source: https://www.itrust.sutd.edu.sg/itrust-labs_datasets/      ║
║     Action: Register → Accept WADI agreement → Download WADI v2      ║
║     Files:  WADI-Dataset-v2/ (or WADI_14days.csv)                   ║
║     Place:  data/raw/wadi/WADI_14days.csv                            ║
║     Shape:  ~161K rows × 127 columns                                ║
║                                                                      ║
║  3. CICIoT2023                                                        ║
║     Source: https://www.unb.ca/cic/datasets/iot-2023.html            ║
║     Action: Click "3 screenshots" → Download 2 CSVs                 ║
║     Files:  0-firewall-acl1-513-73.csv  +  3-combined.csv           ║
║     Place:  data/raw/ciciot2023/                                     ║
║     Shape:  ~8.5M rows × 46 features + label                        ║
║                                                                      ║
║  4. Edge-IIoTSet                                                      ║
║     Source: https://data.mendeley.com/datasets/y7hy5bcd59/1          ║
║     Action: Download "Dataset for Intrusion Detection Systems"       ║
║     Files:  MT_001.csv ... MT_022.csv (22 CSVs) + one_label.csv     ║
║     Place:  data/raw/edge_iiotset/                                   ║
║     Shape:  ~2.3M rows × 46 features                                ║
║                                                                      ║
║  After placing files, run:                                           ║
║    python scripts/download_datasets.py --prepare --dataset swat       ║
║    python scripts/download_datasets.py --prepare --dataset all        ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")


def prepare_dataset(dataset: str) -> pd.DataFrame:
    csv_map = {
        "swat": "SWaT-A2.csv",
        "wadi": "WADI_14days.csv",
        "ciciot2023": "3-combined.csv",
        "edge_iiotset": "one_label.csv",
    }
    d = DATA_ROOT / "raw" / dataset
    target = csv_map.get(dataset)
    if target:
        csv_files = [d / target] if (d / target).exists() else _find_csv_files(d)
    else:
        csv_files = _find_csv_files(d)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in {d}. See --instructions.")
    if len(csv_files) == 1:
        df = pd.read_csv(csv_files[0], low_memory=False)
    else:
        dfs = [pd.read_csv(f, low_memory=False) for f in csv_files[:5]]
        df = pd.concat(dfs, ignore_index=True)
    print(f"Prepared {dataset}: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


def main():
    parser = argparse.ArgumentParser(description="IoT Dataset Download & Preprocessing")
    parser.add_argument("--instructions", action="store_true")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic data")
    parser.add_argument("--dataset", choices=["swat", "wadi", "ciciot2023", "edge_iiotset", "all"], default="all")
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if args.instructions:
        show_instructions()
    if args.synthetic:
        if args.dataset == "all":
            generate_all_synthetic()
        else:
            generate_synthetic(args.dataset, DATA_ROOT / "raw" / args.dataset)
    if args.prepare:
        if args.dataset == "all":
            for ds in ["swat", "wadi", "ciciot2023", "edge_iiotset"]:
                prepare_dataset(ds)
        else:
            prepare_dataset(args.dataset)


if __name__ == "__main__":
    main()
