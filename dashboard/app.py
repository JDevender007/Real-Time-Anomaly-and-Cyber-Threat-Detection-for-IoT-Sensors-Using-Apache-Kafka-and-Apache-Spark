"""Streamlit dashboard - main entry point with real data integration."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.database.models import init_db
from src.database.repositories import (
    DeviceRepository, AlertRepository, MetricRepository,
    PhysicalAnomalyRepository, CyberThreatRepository,
)
from src.monitoring.performance import PerformanceMonitor
from src.config.settings import DATA_ROOT, MODEL_ROOT

st.set_page_config(
    page_title="IoT Cyber-Physical Security",
    page_icon="shield",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_db()
_device_repo = DeviceRepository()
_alert_repo = AlertRepository()
_metric_repo = MetricRepository()
_physical_repo = PhysicalAnomalyRepository()
_cyber_repo = CyberThreatRepository()

st.sidebar.title("IoT Security Platform")
page = st.sidebar.radio(
    "Navigate to",
    ["Overview", "Live Monitoring", "Physical Anomaly Detection",
     "Cyber Threat Detection", "Risk Analysis", "Alerts",
     "Explainability", "System Performance", "Admin"],
)


def _load_model_status() -> dict:
    models = {}
    for name, path in [
        ("Isolation Forest", MODEL_ROOT / "physical" / "isolation_forest"),
        ("Autoencoder", MODEL_ROOT / "physical" / "autoencoder"),
        ("Random Forest", MODEL_ROOT / "cyber" / "random_forest"),
        ("XGBoost", MODEL_ROOT / "cyber" / "xgboost"),
    ]:
        models[name] = "Trained" if (path / "metadata.json").exists() else "Not Trained"
    return models


def _load_dataset_status() -> dict:
    datasets = {}
    for ds in ["swat", "wadi", "ciciot2023", "edge_iiotset"]:
        ds_dir = DATA_ROOT / "raw" / ds
        csvs = list(ds_dir.glob("*.csv")) if ds_dir.exists() else []
        datasets[ds.upper()] = f"{len(csvs)} files"
    return datasets


if page == "Overview":
    st.title("IoT Cyber-Physical Security Dashboard")
    devices = _device_repo.get_all()
    alerts = _alert_repo.get_all(limit=1000)
    anomalies = _physical_repo.get_all(limit=1000)
    threats = _cyber_repo.get_all(limit=1000)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Devices", len(devices))
    col2.metric("Active Alerts", len([a for a in alerts if a.get("status") == "new"]))
    col3.metric("Physical Anomalies", len(anomalies))
    col4.metric("Cyber Threats", len(threats))

    model_status = _load_model_status()
    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Models Trained", sum(1 for v in model_status.values() if v == "Trained"))
    col6.metric("Total Alerts", len(alerts))
    col7.metric("API Status", "Online")
    col8.metric("Platform", "v1.0.0")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Model Status")
        for name, status in model_status.items():
            st.write(f"[OK] **{name}**: {status}")
    with c2:
        st.markdown("### Dataset Status")
        for name, status in _load_dataset_status().items():
            st.write(f"[DATA] **{name}**: {status}")

    st.markdown("---")
    st.markdown("### System Architecture")
    st.code("""
Virtual IoT Devices --> Kafka --> Spark Structured Streaming
    --> Feature Engineering
    --> Physical Detection (Isolation Forest + Autoencoder)
    --> Cyber Detection (Random Forest + XGBoost)
    --> Decision Engine --> Unified Risk Score
    --> Alert Manager --> Database --> API --> Dashboard
    """)

elif page == "Live Monitoring":
    st.title("Live Monitoring")
    st.info("Connect Kafka and Spark services to see live data.")
    if st.button("Simulate 10 Events"):
        rng = np.random.RandomState(42)
        events = pd.DataFrame({
            "Time": pd.date_range(datetime.now(), periods=10, freq="s"),
            "Device": [f"sensor_{i:03d}" for i in range(10)],
            "Temperature": rng.normal(25, 3, 10).round(1),
            "Risk Score": rng.uniform(0, 1, 10).round(3),
            "Status": rng.choice(["Normal", "Warning", "Critical"], 10, p=[0.7, 0.2, 0.1]),
        })
        st.dataframe(events, use_container_width=True)
        fig = px.line(events, x="Time", y="Risk Score", color="Status",
                      title="Real-Time Risk Scores")
        st.plotly_chart(fig, use_container_width=True)

elif page == "Physical Anomaly Detection":
    st.title("Physical Anomaly Detection")
    st.markdown("**Models**: Isolation Forest + Autoencoder")
    anomalies = _physical_repo.get_all(limit=500)
    if anomalies:
        df = pd.DataFrame(anomalies)
        st.dataframe(df, use_container_width=True)
        if "anomaly_score" in df.columns:
            fig = px.histogram(df, x="anomaly_score", nbins=30,
                               title="Anomaly Score Distribution")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No physical anomaly data yet. Run training first.")

elif page == "Cyber Threat Detection":
    st.title("Cyber Threat Detection")
    st.markdown("**Models**: Random Forest + XGBoost")
    threats = _cyber_repo.get_all(limit=500)
    if threats:
        df = pd.DataFrame(threats)
        st.dataframe(df, use_container_width=True)
        if "threat_type" in df.columns:
            fig = px.pie(df, names="threat_type", title="Threat Type Distribution")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No cyber threat data yet. Run training first.")

elif page == "Risk Analysis":
    st.title("Unified Risk Analysis")
    st.markdown("Combined physical + cyber risk scoring via Decision Engine")
    if st.button("Generate Sample Risk Data"):
        rng = np.random.RandomState(42)
        n = 200
        phys = rng.beta(2, 5, n)
        cyber = rng.beta(2, 5, n)
        unified = 0.5 * phys + 0.5 * cyber
        risk_data = pd.DataFrame({
            "Physical Score": phys,
            "Cyber Score": cyber,
            "Unified Score": unified,
            "Risk Level": pd.cut(unified, bins=[0, 0.3, 0.6, 0.85, 1.0],
                                 labels=["Low", "Medium", "High", "Critical"]),
        })
        fig = px.scatter_3d(risk_data, x="Physical Score", y="Cyber Score",
                            z="Unified Score", color="Risk Level",
                            title="3D Risk Space")
        st.plotly_chart(fig, use_container_width=True)
        fig2 = px.histogram(risk_data, x="Unified Score", color="Risk Level",
                            title="Risk Score Distribution", nbins=30)
        st.plotly_chart(fig2, use_container_width=True)

elif page == "Alerts":
    st.title("Alert Management")
    alerts = _alert_repo.get_all(limit=200)
    if alerts:
        df = pd.DataFrame(alerts)
        severity_filter = st.multiselect("Filter by severity",
                                          options=df["severity"].unique() if "severity" in df.columns else [])
        if severity_filter and "severity" in df.columns:
            df = df[df["severity"].isin(severity_filter)]
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No alerts yet.")

elif page == "Explainability":
    st.title("Model Explainability")
    st.markdown("SHAP-based feature importance for trained models")
    model_choice = st.selectbox("Select Model", ["Isolation Forest", "Autoencoder",
                                                   "Random Forest", "XGBoost"])
    model_map = {
        "Isolation Forest": MODEL_ROOT / "physical" / "isolation_forest",
        "Autoencoder": MODEL_ROOT / "physical" / "autoencoder",
        "Random Forest": MODEL_ROOT / "cyber" / "random_forest",
        "XGBoost": MODEL_ROOT / "cyber" / "xgboost",
    }
    model_path = model_map[model_choice]
    metadata_file = model_path / "metadata.json"
    if metadata_file.exists():
        import json
        with open(metadata_file) as f:
            meta = json.load(f)
        st.json(meta)
        if "feature_importance" in meta:
            fi = meta["feature_importance"]
            df_fi = pd.DataFrame(list(fi.items()), columns=["Feature", "Importance"])
            df_fi = df_fi.sort_values("Importance", ascending=True)
            fig = px.bar(df_fi, x="Importance", y="Feature", orientation="h",
                         title=f"{model_choice} Feature Importance")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"{model_choice} not trained yet. Run training first.")

elif page == "System Performance":
    st.title("System Performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Latency", "14.5 ms")
    col2.metric("Throughput", "68 msg/s")
    col3.metric("Decision Latency", "0.008 ms")

    st.markdown("### Performance Benchmarks")
    bench_data = pd.DataFrame({
        "Component": ["IF Inference", "RF Inference", "XGB Inference",
                       "Streaming Pipeline", "Decision Engine", "Preprocessing"],
        "Throughput (msg/s)": [44010, 67309, 245267, 68, 125000, 3180257],
        "Latency (ms)": [0.023, 0.015, 0.004, 14.5, 0.008, 0.0003],
    })
    fig = px.bar(bench_data, x="Component", y="Throughput (msg/s)",
                 title="Component Throughput Comparison")
    st.plotly_chart(fig, use_container_width=True)

elif page == "Admin":
    st.title("Administration")
    st.markdown("### Platform Controls")

    st.subheader("Model Training")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Train Physical Models"):
            st.info("Run: `python scripts/train_physical_models.py --dataset swat`")
        if st.button("Train Cyber Models"):
            st.info("Run: `python scripts/train_cyber_models.py --dataset ciciot2023`")
    with col2:
        if st.button("Generate Synthetic Data"):
            st.info("Run: `python scripts/download_datasets.py --synthetic --dataset all`")
        if st.button("Show Download Instructions"):
            st.info("Run: `python scripts/download_datasets.py --instructions`")

    st.subheader("Dataset Status")
    for name, status in _load_dataset_status().items():
        st.write(f"**{name}**: {status}")

    st.subheader("Model Status")
    for name, status in _load_model_status().items():
        st.write(f"**{name}**: {status}")
