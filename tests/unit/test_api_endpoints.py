"""API endpoint tests using FastAPI TestClient."""
from __future__ import annotations

import os, sys

os.environ["PYTHONUNBUFFERED"] = "1"
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from fastapi.testclient import TestClient
from src.api.main import app
from src.api.auth import get_api_keys

client = TestClient(app)
API_KEY = get_api_keys()[0]

passed = 0
errors = []


def check(name, condition, detail=""):
    global passed
    if condition:
        passed += 1
        print(f"  [PASS] {name}", flush=True)
    else:
        errors.append(name)
        print(f"  [FAIL] {name}: {detail}", flush=True)


print("=" * 70)
print("  API ENDPOINT TESTS")
print("=" * 70)

# Health (no auth)
r = client.get("/api/v1/health")
check("GET /api/v1/health", r.status_code == 200 and r.json()["status"] == "ok")

# System status (no auth)
r = client.get("/api/v1/system/status")
check("GET /api/v1/system/status", r.status_code == 200 and r.json()["status"] == "running")

# Auth-required endpoints WITHOUT key -> should fail
r = client.get("/api/v1/devices")
check("GET /api/v1/devices without key -> 401/403", r.status_code in (401, 403))

# Auth-required endpoints WITH key
r = client.get("/api/v1/devices", headers={"X-API-Key": API_KEY})
check("GET /api/v1/devices with key", r.status_code == 200 and "devices" in r.json())

r = client.get("/api/v1/alerts", headers={"X-API-Key": API_KEY})
check("GET /api/v1/alerts", r.status_code == 200 and "alerts" in r.json())

r = client.get("/api/v1/anomalies", headers={"X-API-Key": API_KEY})
check("GET /api/v1/anomalies", r.status_code == 200 and "anomalies" in r.json())

r = client.get("/api/v1/threats", headers={"X-API-Key": API_KEY})
check("GET /api/v1/threats", r.status_code == 200 and "threats" in r.json())

r = client.get("/api/v1/metrics", headers={"X-API-Key": API_KEY})
check("GET /api/v1/metrics", r.status_code == 200 and "metrics" in r.json())

r = client.get("/api/v1/models", headers={"X-API-Key": API_KEY})
check("GET /api/v1/models", r.status_code == 200 and "models" in r.json())

r = client.get("/api/v1/statistics", headers={"X-API-Key": API_KEY})
check("GET /api/v1/statistics", r.status_code == 200)

# Device detail
r = client.get("/api/v1/devices/test-device", headers={"X-API-Key": API_KEY})
check("GET /api/v1/devices/{id}", r.status_code in (200, 404))

# Simulation endpoints
r = client.post("/api/v1/simulation/start", headers={"X-API-Key": API_KEY})
check("POST /api/v1/simulation/start", r.status_code == 200 and "status" in r.json())

r = client.post("/api/v1/simulation/anomaly", headers={"X-API-Key": API_KEY})
check("POST /api/v1/simulation/anomaly", r.status_code == 200)

r = client.post("/api/v1/simulation/attack", headers={"X-API-Key": API_KEY})
check("POST /api/v1/simulation/attack", r.status_code == 200)

r = client.post("/api/v1/simulation/stop", headers={"X-API-Key": API_KEY})
check("POST /api/v1/simulation/stop", r.status_code == 200)

print("\n" + "=" * 70)
print(f"  RESULTS: {passed} passed, {len(errors)} failed")
if errors:
    print(f"  Failed: {errors}")
print("=" * 70)
