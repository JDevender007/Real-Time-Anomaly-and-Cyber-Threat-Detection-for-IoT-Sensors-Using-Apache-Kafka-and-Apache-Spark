"""End-to-end integration test: API + Models + Streaming pipeline."""
from __future__ import annotations

import os, sys, time, json
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import joblib

def p(msg=""):
    print(msg, flush=True)

p("=" * 70)
p("  END-TO-END INTEGRATION TEST")
p("=" * 70)

errors = []
passed = 0

def check(name, condition, detail=""):
    global passed
    if condition:
        passed += 1
        p(f"  [PASS] {name}")
    else:
        errors.append(name)
        p(f"  [FAIL] {name}: {detail}")

# ============================================================
# 1. Model Loading
# ============================================================
p("\n[1/4] Model Loading")
model_root = Path("models")

# Physical models
if_dir = model_root / "physical" / "isolation_forest"
ae_dir = model_root / "physical" / "autoencoder"
cyber_dir = model_root / "cyber"
stacking_dir = model_root / "stacking"

check("IF ensemble dir exists", if_dir.exists())
check("AE dir exists", ae_dir.exists())
check("Cyber dir exists", cyber_dir.exists())
check("Stacking dir exists", stacking_dir.exists())

# List all saved models
p("\n  Saved models:")
for f in model_root.rglob("*.joblib"):
    p(f"    {f.relative_to(model_root)} ({f.stat().st_size // 1024}KB)")
for f in model_root.rglob("*.pt"):
    p(f"    {f.relative_to(model_root)} ({f.stat().st_size // 1024}KB)")

# Try loading key models
try:
    rf_cyber = joblib.load(cyber_dir / "random_forest.joblib")
    check("Load cyber RF model", True)
except Exception as e:
    check("Load cyber RF model", False, str(e))

try:
    xgb_cyber = joblib.load(cyber_dir / "xgboost.joblib")
    check("Load cyber XGBoost model", True)
except Exception as e:
    check("Load cyber XGBoost model", False, str(e))

try:
    rf_edge = joblib.load(cyber_dir / "rf_edge_iiotset.joblib")
    check("Load Edge-IIoTSet RF model", True)
except Exception as e:
    check("Load Edge-IIoTSet RF model", False, str(e))

try:
    xgb_edge = joblib.load(cyber_dir / "xgb_edge_iiotset.joblib")
    check("Load Edge-IIoTSet XGBoost model", True)
except Exception as e:
    check("Load Edge-IIoTSet XGBoost model", False, str(e))

try:
    meta_learner = joblib.load(stacking_dir / "meta_learner.joblib")
    check("Load stacking meta-learner", True)
except Exception as e:
    check("Load stacking meta-learner", False, str(e))

# ============================================================
# 2. Model Inference Test
# ============================================================
p("\n[2/4] Model Inference Test")

# Cyber RF inference on synthetic data
try:
    X_dummy = np.random.randn(100, 46).astype(np.float32)  # 46 features = CICIoT2023
    y_pred = rf_cyber.predict(X_dummy)
    y_proba = rf_cyber.predict_proba(X_dummy)
    check("Cyber RF inference", y_pred.shape == (100,) and y_proba.shape == (100, 2))
except Exception as e:
    check("Cyber RF inference", False, str(e))

# Cyber XGBoost inference
try:
    y_pred_xgb = xgb_cyber.predict(X_dummy)
    y_proba_xgb = xgb_cyber.predict_proba(X_dummy)
    check("Cyber XGBoost inference", y_pred_xgb.shape == (100,))
except Exception as e:
    check("Cyber XGBoost inference", False, str(e))

# Edge-IIoTSet RF inference (57 features)
try:
    X_edge = np.random.randn(100, 57).astype(np.float32)
    y_edge = rf_edge.predict(X_edge)
    check("Edge-IIoTSet RF inference", y_edge.shape == (100,))
except Exception as e:
    check("Edge-IIoTSet RF inference", False, str(e))

# Meta-learner inference (3 features: IF_score, AE_score, domain)
try:
    X_meta = np.random.rand(100, 3).astype(np.float32)
    y_meta = meta_learner.predict(X_meta)
    check("Stacking meta-learner inference", y_meta.shape == (100,))
except Exception as e:
    check("Stacking meta-learner inference", False, str(e))

# ============================================================
# 3. Dataset Loading Integration
# ============================================================
p("\n[3/4] Dataset Loading Integration")

try:
    from src.datasets.swat import SWaTDatasetLoader
    swat = SWaTDatasetLoader(Path("data/raw/swat"))
    df_s = swat.load()
    df_s = swat.clean(df_s)
    df_s = swat.transform(df_s)
    check("SWaT load+clean+transform", len(df_s) > 100000)
    p(f"    SWaT: {len(df_s):,} rows, {len(swat.get_features())} features")
except Exception as e:
    check("SWaT load+clean+transform", False, str(e))

try:
    from src.datasets.wadi import WADIDatasetLoader
    wadi = WADIDatasetLoader(Path("data/raw/wadi"))
    df_w = wadi.load()
    df_w = wadi.clean(df_w)
    df_w = wadi.transform(df_w)
    check("WADI load+clean+transform", len(df_w) > 100000)
    p(f"    WADI: {len(df_w):,} rows, {len(wadi.get_features())} features")
except Exception as e:
    check("WADI load+clean+transform", False, str(e))

try:
    from src.datasets.ciciot2023 import CICIoT2023DatasetLoader
    cic = CICIoT2023DatasetLoader(Path("data/raw/ciciot2023"))
    df_c = cic.load()
    df_c = cic.clean(df_c)
    df_c = cic.transform(df_c)
    check("CICIoT2023 load+clean+transform", len(df_c) > 100000)
    p(f"    CICIoT2023: {len(df_c):,} rows, {len(cic.get_features())} features")
except Exception as e:
    check("CICIoT2023 load+clean+transform", False, str(e))

try:
    from src.datasets.edge_iiotset import EdgeIIoTSetDatasetLoader
    eii = EdgeIIoTSetDatasetLoader(Path("data/raw/edge_iiotset"))
    df_e = eii.load()
    df_e = eii.clean(df_e)
    df_e = eii.transform(df_e)
    check("Edge-IIoTSet load+clean+transform", len(df_e) > 10000)
    p(f"    Edge-IIoTSet: {len(df_e):,} rows, {len(eii.get_features())} features")
except Exception as e:
    check("Edge-IIoTSet load+clean+transform", False, str(e))

# ============================================================
# 4. Streaming Pipeline Component Test
# ============================================================
p("\n[4/4] Streaming Pipeline Components")

try:
    from src.decision.engine import DecisionEngine
    de = DecisionEngine()
    result = de.compute_risk(
        physical_score=0.8,
        cyber_score=0.2,
        device_health=95.0,
        model_confidence=0.9
    )
    check("DecisionEngine computes risk", result is not None and hasattr(result, "unified_risk_score"))
    p(f"    Risk score: {result.unified_risk_score:.4f}, Level: {result.risk_level}")
except Exception as e:
    check("DecisionEngine computes risk", False, str(e))

try:
    from src.alerts.manager import AlertManager
    from src.api.schemas import RiskLevel, AlertType
    am = AlertManager()
    alert = am.create_alert(
        device_id="test-sensor-001",
        alert_type=AlertType.PHYSICAL_ANOMALY,
        severity=RiskLevel.HIGH,
        physical_score=0.8,
        cyber_score=0.2,
        risk_score=0.6,
        model_name="integration_test",
        explanation="Test alert from integration test"
    )
    check("AlertManager creates alert", alert is not None)
except Exception as e:
    check("AlertManager creates alert", False, str(e))

try:
    from src.simulation.iot_simulator import IoTSimulator
    sim = IoTSimulator(num_devices=5)
    sensors, network = sim.generate_once()
    check("IoTSimulator generate_once", len(sensors) > 0 and len(network) > 0)
    p(f"    Generated {len(sensors)} sensor msgs, {len(network)} network msgs")
except Exception as e:
    check("IoTSimulator generate_once", False, str(e))

try:
    from src.simulation.attack_simulator import CyberAttackSimulator
    atk = CyberAttackSimulator()
    event = atk.simulate_ddoS(device_id="test-device-001")
    check("CyberAttackSimulator generates attack", event is not None)
except Exception as e:
    check("CyberAttackSimulator generates attack", False, str(e))

# ============================================================
# Summary
# ============================================================
p("\n" + "=" * 70)
p(f"  RESULTS: {passed} passed, {len(errors)} failed")
if errors:
    p(f"  Failed: {errors}")
p("=" * 70)
