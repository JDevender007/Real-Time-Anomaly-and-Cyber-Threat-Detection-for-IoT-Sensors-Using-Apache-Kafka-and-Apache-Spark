"""Run pipeline for 10 cycles."""
import sys, os
sys.path.insert(0, os.getcwd())
os.environ["PYTHONUNBUFFERED"] = "1"

from src.pipeline.orchestrator import Pipeline, PipelineConfig

config = PipelineConfig(num_devices=10, events_per_cycle=20, attack_probability=0.15, cycle_interval_s=0.5)
pipeline = Pipeline(config)
pipeline.initialize()

print("Running 10 cycles...", flush=True)
for i in range(10):
    result = pipeline.run_cycle()
    cycle = result["cycle"]
    events = result["events"]
    alerts = result["alerts"]
    ct = result["cycle_time_ms"]
    print(f"  Cycle {cycle}: {events} events, {alerts} alerts, {ct}ms", flush=True)

print(flush=True)
stats = pipeline.get_stats()
print("Pipeline Stats:", flush=True)
for k, v in stats.items():
    print(f"  {k}: {v}", flush=True)
