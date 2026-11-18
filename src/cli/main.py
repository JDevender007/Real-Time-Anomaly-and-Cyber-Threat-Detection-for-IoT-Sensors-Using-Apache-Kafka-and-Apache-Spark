"""CLI completion and enhanced commands."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def cmd_train_physical(args):
    from scripts.train_physical_models import train_physical_models
    result = train_physical_models(args.dataset)
    print(json.dumps(result, indent=2, default=str))


def cmd_train_cyber(args):
    from scripts.train_cyber_models import train_cyber_models
    result = train_cyber_models(args.dataset)
    print(json.dumps(result, indent=2, default=str))


def cmd_generate_data(args):
    from scripts.download_datasets import generate_synthetic
    if args.dataset == "all":
        from scripts.download_datasets import generate_all_synthetic
        generate_all_synthetic()
    else:
        from src.config.settings import DATA_ROOT
        generate_synthetic(args.dataset, DATA_ROOT / "raw" / args.dataset)


def cmd_show_instructions(args):
    from scripts.download_datasets import show_instructions
    show_instructions()


def cmd_evaluate(args):
    from pathlib import Path as P
    results_dir = P("experiments/results")
    if results_dir.exists():
        for f in sorted(results_dir.glob("*.json")):
            print(f"\n=== {f.stem} ===")
            with open(f) as fh:
                data = json.load(fh)
                print(json.dumps(data, indent=2, default=str))
    else:
        print("No experiment results found. Run training first.")


def cmd_serve(args):
    import uvicorn
    from src.config.settings import API_HOST, API_PORT
    host = args.host or API_HOST
    port = args.port or API_PORT
    print(f"Starting API server at http://{host}:{port}")
    uvicorn.run("src.api.main:app", host=host, port=port, reload=args.reload)


def cmd_dashboard(args):
    import subprocess
    print("Starting Streamlit dashboard...")
    subprocess.run(["streamlit", "run", "dashboard/app.py", "--server.port", str(args.port)])


def cmd_status(args):
    from src.config.settings import MODEL_ROOT, DATA_ROOT
    print("=== IoT Security Platform Status ===\n")

    print("Datasets:")
    for ds in ["swat", "wadi", "ciciot2023", "edge_iiotset"]:
        ds_dir = DATA_ROOT / "raw" / ds
        csvs = list(ds_dir.glob("*.csv")) if ds_dir.exists() else []
        status = f"{len(csvs)} files" if csvs else "not downloaded"
        print(f"  {ds}: {status}")

    print("\nModels:")
    model_dirs = {
        "Isolation Forest": MODEL_ROOT / "physical" / "isolation_forest",
        "Autoencoder": MODEL_ROOT / "physical" / "autoencoder",
        "Random Forest": MODEL_ROOT / "cyber" / "random_forest",
        "XGBoost": MODEL_ROOT / "cyber" / "xgboost",
    }
    for name, path in model_dirs.items():
        status = "trained" if (path / "metadata.json").exists() else "not trained"
        print(f"  {name}: {status}")

    print("\nResults:")
    results_dir = Path("experiments/results")
    if results_dir.exists():
        for f in sorted(results_dir.glob("*.json")):
            print(f"  {f.stem}")
    else:
        print("  No results yet")


def cmd_test(args):
    import subprocess
    cmd = ["python", "-m", "pytest", "tests/", "-v", "--tb=short"]
    if args.coverage:
        cmd.extend(["--cov=src", "--cov-report=term-missing"])
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="iot-security",
        description="IoT Cyber-Physical Security Platform CLI",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    p_train_phys = sub.add_parser("train-physical", help="Train physical anomaly models")
    p_train_phys.add_argument("--dataset", choices=["swat", "wadi"], default="swat")
    p_train_phys.set_defaults(func=cmd_train_physical)

    p_train_cyber = sub.add_parser("train-cyber", help="Train cyber threat models")
    p_train_cyber.add_argument("--dataset", choices=["ciciot2023", "edge_iiotset"], default="ciciot2023")
    p_train_cyber.set_defaults(func=cmd_train_cyber)

    p_data = sub.add_parser("generate-data", help="Generate synthetic datasets")
    p_data.add_argument("--dataset", choices=["swat", "wadi", "ciciot2023", "edge_iiotset", "all"], default="all")
    p_data.set_defaults(func=cmd_generate_data)

    p_instr = sub.add_parser("instructions", help="Show dataset download instructions")
    p_instr.set_defaults(func=cmd_show_instructions)

    p_eval = sub.add_parser("evaluate", help="Show experiment results")
    p_eval.set_defaults(func=cmd_evaluate)

    p_serve = sub.add_parser("serve", help="Start API server")
    p_serve.add_argument("--host", default=None)
    p_serve.add_argument("--port", type=int, default=None)
    p_serve.add_argument("--reload", action="store_true")
    p_serve.set_defaults(func=cmd_serve)

    p_dash = sub.add_parser("dashboard", help="Start Streamlit dashboard")
    p_dash.add_argument("--port", type=int, default=8501)
    p_dash.set_defaults(func=cmd_dashboard)

    p_status = sub.add_parser("status", help="Show platform status")
    p_status.set_defaults(func=cmd_status)

    p_test = sub.add_parser("test", help="Run test suite")
    p_test.add_argument("--coverage", action="store_true")
    p_test.set_defaults(func=cmd_test)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
