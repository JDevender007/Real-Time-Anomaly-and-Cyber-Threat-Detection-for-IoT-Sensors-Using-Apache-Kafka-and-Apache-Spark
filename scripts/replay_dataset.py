"""Dataset replayer: streams real dataset rows into Kafka for live ML inference.

Replays rows from SWaT/WADI (physical domain) and CICIoT2023 (cyber domain)
with ground-truth labels so the streaming models can be evaluated live.

Topics:
    iot.physical.replay  - 178-feature physical rows (SWaT+WADI feature union)
    iot.cyber.replay     - 46-feature CICIoT2023 network flow rows

Env:
    KAFKA_BOOTSTRAP_SERVERS  Kafka bootstrap (default localhost:9092)
    REPLAY_PHYSICAL_RATE     physical rows/sec (default 10)
    REPLAY_CYBER_RATE        cyber rows/sec (default 10)
    REPLAY_SAMPLE_SIZE       rows sampled per domain (default 30000)
"""
from __future__ import annotations

import os
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config.loader import load_config
from src.config.logging_config import setup_logging
from src.config.settings import create_required_directories, MODEL_ROOT
from src.kafka.producer import KafkaProducer
from src.kafka.topics import ensure_topics

import logging
logger = logging.getLogger("replayer")

_running = True


def _signal_handler(sig, frame):
    global _running
    logger.info("Shutdown signal received")
    _running = False


def _read_swat_bounded(data_dir: Path, normal_rows: int) -> pd.DataFrame:
    """Read SWaT attack.csv fully + a bounded random sample of normal.csv."""
    frames: list[pd.DataFrame] = []
    att_path = data_dir / "attack.csv"
    if att_path.exists():
        att = pd.read_csv(att_path, encoding="utf-8", low_memory=False)
        att.columns = att.columns.str.strip()
        att["_src_label"] = 1
        frames.append(att)
        logger.info("SWaT attack.csv: %s rows", len(att))

    nrm_path = data_dir / "normal.csv"
    if nrm_path.exists():
        rng = np.random.default_rng(42)
        total = sum(1 for _ in open(nrm_path, encoding="utf-8", errors="ignore")) - 1
        keep = min(normal_rows, total)
        if keep < total:
            chosen = set(rng.choice(total, size=keep, replace=False).tolist())
            nrm = pd.read_csv(
                nrm_path, encoding="utf-8", low_memory=False,
                skiprows=lambda i: i > 0 and (i - 1) not in chosen,
            )
        else:
            nrm = pd.read_csv(nrm_path, encoding="utf-8", low_memory=False)
        nrm.columns = nrm.columns.str.strip()
        nrm["_src_label"] = 0
        frames.append(nrm)
        logger.info("SWaT normal.csv: sampled %s rows", len(nrm))

    if not frames:
        raise FileNotFoundError(f"No SWaT CSV files under {data_dir}")
    return pd.concat(frames, ignore_index=True)


def load_physical_rows(max_rows: int) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    import joblib

    feature_names: list[str] = joblib.load(MODEL_ROOT / "physical" / "feature_names.pkl")

    df = _read_swat_bounded(Path("data/raw/swat"), normal_rows=max_rows // 2)

    if "Normal/Attack" in df.columns:
        df["attack_label"] = df["Normal/Attack"].apply(
            lambda x: 0 if str(x).strip().lower() == "normal" else 1
        )
    elif "_src_label" in df.columns:
        df["attack_label"] = df["_src_label"]
    else:
        df["attack_label"] = 0

    for col in df.columns:
        if col in ("Timestamp", "Normal/Attack", "attack_label", "_src_label"):
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42)

    labels = df["attack_label"].astype(np.int32).values

    df = df.reindex(columns=feature_names, fill_value=0.0)
    df = df.fillna(0.0).astype(np.float32)
    logger.info(
        "Physical replay: %s rows x %s features (%.2f%% attack)",
        len(df), len(feature_names), float(labels.mean()) * 100,
    )
    return df, labels, feature_names


def load_cyber_rows(max_rows: int) -> tuple[pd.DataFrame, list[str]]:
    import joblib

    feature_names: list[str] = joblib.load(MODEL_ROOT / "cyber" / "feature_names.joblib")

    data_dir = Path("data/raw/ciciot2023")
    csv_files = sorted(
        f for f in data_dir.rglob("*.csv") if "synthetic" not in f.name.lower()
    )
    if not csv_files:
        raise FileNotFoundError(f"No CICIoT2023 CSV files under {data_dir}")

    per_file = max(1000, max_rows // max(len(csv_files), 1))
    frames = []
    for f in csv_files:
        try:
            part = pd.read_csv(f, encoding="utf-8", nrows=per_file, low_memory=False)
            frames.append(part)
        except Exception as e:
            logger.warning("Skipping %s: %s", f.name, e)
    df = pd.concat(frames, ignore_index=True)

    if "attack_label" not in df.columns:
        label_col = next(
            (c for c in df.columns if c.lower() in ("label", "attack", "class", "category")),
            None,
        )
        if label_col is None:
            raise ValueError("CICIoT2023 replay data has no label column")
        df["attack_label"] = df[label_col].apply(
            lambda x: 0 if "benign" in str(x).lower() else 1
        )

    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42)

    label_col = next(
        (c for c in df.columns if c.lower() in ("label", "attack", "class", "category")),
        None,
    )
    df = df[feature_names + ["attack_label"] + ([label_col] if label_col else [])]
    for col in feature_names:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[feature_names] = df[feature_names].fillna(0.0).astype(np.float32)
    logger.info("Cyber replay: %s rows x %s features", len(df), len(feature_names))
    return df, feature_names


def main() -> None:
    setup_logging()
    create_required_directories()
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    kafka_cfg = load_config("kafka")
    bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS") or kafka_cfg.kafka.bootstrap_servers
    phys_topic = kafka_cfg.kafka.topics.physical_replay
    cyber_topic = kafka_cfg.kafka.topics.cyber_replay

    phys_rate = float(os.getenv("REPLAY_PHYSICAL_RATE", "10"))
    cyber_rate = float(os.getenv("REPLAY_CYBER_RATE", "10"))
    sample_size = int(os.getenv("REPLAY_SAMPLE_SIZE", "30000"))

    ensure_topics(bootstrap)

    logger.info("Loading replay datasets (sample size=%s)...", sample_size)
    t0 = time.time()
    phys_df, phys_labels, phys_feats = load_physical_rows(sample_size)
    cyber_df, cyber_feats = load_cyber_rows(sample_size)
    logger.info("Datasets loaded in %.1fs", time.time() - t0)

    phys_df = phys_df.reset_index(drop=True)
    cyber_df = cyber_df.reset_index(drop=True)

    producer = KafkaProducer(bootstrap_servers=bootstrap, topic=phys_topic)
    producer.connect()
    if not producer.is_connected:
        logger.error("Kafka not available at %s - exiting", bootstrap)
        return

    logger.info(
        "Replaying: physical %.0f msg/s on %s | cyber %.0f msg/s on %s",
        phys_rate, phys_topic, cyber_rate, cyber_topic,
    )

    phys_records = phys_df.to_dict("records")
    cyber_records = cyber_df.to_dict("records")
    attack_types = None
    if "label" in cyber_df.columns:
        attack_types = cyber_df["label"].tolist()

    p_idx = c_idx = 0
    p_sent = c_sent = 0
    p_attacks = c_attacks = 0
    last_tick = time.time()
    last_log = time.time()

    phys_interval = 1.0 / max(phys_rate, 0.1)
    cyber_interval = 1.0 / max(cyber_rate, 0.1)
    next_phys = time.time()
    next_cyber = time.time()

    while _running:
        now = time.time()

        if now >= next_phys:
            row = phys_records[p_idx % len(phys_records)]
            label = int(phys_labels[p_idx % len(phys_labels)])
            payload = {
                "kind": "dataset_replay",
                "domain": "physical",
                "dataset": "swat_wadi",
                "device_id": f"replay_phys_{(p_idx % 5) + 1:03d}",
                "timestamp": pd.Timestamp.now("UTC").isoformat(),
                "features": {k: float(v) for k, v in row.items()},
                "label": label,
            }
            if producer.send(payload, topic=phys_topic):
                p_sent += 1
                p_attacks += label
            p_idx += 1
            next_phys = now + phys_interval

        if now >= next_cyber:
            row = cyber_records[c_idx % len(cyber_records)]
            label = int(row.get("attack_label", 0))
            atype = str(attack_types[c_idx % len(attack_types)]) if attack_types else str(label)
            payload = {
                "kind": "dataset_replay",
                "domain": "cyber",
                "dataset": "ciciot2023",
                "device_id": f"replay_cyber_{(c_idx % 5) + 1:03d}",
                "timestamp": pd.Timestamp.now("UTC").isoformat(),
                "features": {
                    k: float(row.get(k, 0.0) or 0.0) for k in cyber_feats
                },
                "label": label,
                "attack_type": atype,
            }
            if producer.send(payload, topic=cyber_topic):
                c_sent += 1
                c_attacks += label
            c_idx += 1
            next_cyber = now + cyber_interval

        if time.time() - last_log > 15:
            elapsed = time.time() - last_tick
            logger.info(
                "Replayed: %d physical (%.1f/s, %.1f%% attack), "
                "%d cyber (%.1f/s, %.1f%% attack)",
                p_sent, p_sent / max(elapsed, 1),
                (p_attacks / p_sent * 100) if p_sent else 0.0,
                c_sent, c_sent / max(elapsed, 1),
                (c_attacks / c_sent * 100) if c_sent else 0.0,
            )
            last_log = time.time()

        time.sleep(0.005)

    logger.info(
        "Final: %d physical (%d attack), %d cyber (%d attack) rows replayed",
        p_sent, p_attacks, c_sent, c_attacks,
    )
    producer.close()


if __name__ == "__main__":
    main()
