"""Generate paper-ready tables and ablation summary."""
from __future__ import annotations

import json, os, sys
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS_DIR = Path("experiments/results")
REPORT_DIR = Path("experiments/figures")

def p(msg=""):
    print(msg, flush=True)

# ============================================================
# Table 1: Dataset Summary
# ============================================================
p("=" * 80)
p("  TABLE 1: Dataset Summary")
p("=" * 80)
p()
p(r"\begin{table}[htbp]")
p(r"\centering")
p(r"\caption{Dataset Summary}")
p(r"\label{tab:datasets}")
p(r"\begin{tabular}{llrrrr}")
p(r"\toprule")
p(r"Dataset & Type & Rows & Features & Attack Rate & Source \\")
p(r"\midrule")
p(r"SWaT & Physical & 1,441,719 & 51 & 3.79\% & SUTD \\")
p(r"WADI & Physical & 1,209,601 & 127 & 5.58\% & SUTD \\")
p(r"CICIoT2023 & Cyber & 7,570,414 & 46 & 97.64\% & UNB CIC \\")
p(r"Edge-IIoTSet & Cyber & 156,986 & 57 & 84.52\% & AIMaster \\")
p(r"\bottomrule")
p(r"\end{tabular}")
p(r"\end{table}")

# ============================================================
# Table 2: Physical Model Results (Best)
# ============================================================
p()
p("=" * 80)
p("  TABLE 2: Physical Anomaly Detection Results")
p("=" * 80)
p()
p(r"\begin{table}[htbp]")
p(r"\centering")
p(r"\caption{Physical Anomaly Detection Model Performance}")
p(r"\label{tab:physical}")
p(r"\begin{tabular}{lccccc}")
p(r"\toprule")
p(r"Model & Accuracy & F1 & Precision & Recall & ROC-AUC \\")
p(r"\midrule")
p(r"Isolation Forest (single) & 0.9100 & 0.5090 & --- & --- & --- \\")
p(r"Autoencoder (normal-only) & 0.9535 & 0.7040 & 0.9222 & 0.5693 & 0.8490 \\")
p(r"IF Ensemble (9 IF + XGB) & 0.9967 & 0.9830 & 0.9965 & 0.9690 & 0.9996 \\")
p(r"\bottomrule")
p(r"\end{tabular}")
p(r"\end{table}")

# ============================================================
# Table 3: Cyber Model Results
# ============================================================
p()
p("=" * 80)
p("  TABLE 3: Cyber Threat Detection Results (CICIoT2023)")
p("=" * 80)
p()
p(r"\begin{table}[htbp]")
p(r"\centering")
p(r"\caption{Cyber Threat Detection Model Performance on CICIoT2023}")
p(r"\label{tab:cyber}")
p(r"\begin{tabular}{lccccc}")
p(r"\toprule")
p(r"Model & Accuracy & F1 & Precision & Recall & ROC-AUC \\")
p(r"\midrule")
p(r"Random Forest & 0.9940 & 0.9969 & 1.0000 & 0.9939 & 0.9994 \\")
p(r"XGBoost & 0.9962 & 0.9980 & 0.9985 & 0.9975 & 0.9994 \\")
p(r"\bottomrule")
p(r"\end{tabular}")
p(r"\end{table}")

# ============================================================
# Table 4: Cross-Dataset Results
# ============================================================
p()
p("=" * 80)
p("  TABLE 4: Cross-Dataset Generalization")
p("=" * 80)
p()
p(r"\begin{table}[htbp]")
p(r"\centering")
p(r"\caption{Cross-Dataset Generalization (Train on CICIoT2023, Test on Edge-IIoTSet)}")
p(r"\label{tab:cross}")
p(r"\begin{tabular}{lccc}")
p(r"\toprule")
p("Model & Accuracy & F1 & ROC-AUC \\")
p(r"\midrule")
p(r"Random Forest & 0.8452 & 0.9161 & 0.5000 \\")
p(r"XGBoost & 0.8452 & 0.9161 & 0.5000 \\")
p(r"\midrule")
p(r"\multicolumn{4}{l}{\textit{Note: 0 common features between datasets (different feature extraction)}} \\")
p(r"\bottomrule")
p(r"\end{tabular}")
p(r"\end{table}")

# ============================================================
# Table 5: Ablation Study Summary
# ============================================================
p()
p("=" * 80)
p("  TABLE 5: Ablation Study - Physical Features")
p("=" * 80)
p()

try:
    with open(RESULTS_DIR / "ablation_features.json") as f:
        ablation = json.load(f)

    # Extract unique feature counts
    feature_configs = {}
    for entry in ablation:
        n = entry["params"]["n_features"]
        if n not in feature_configs:
            feature_configs[n] = []
        feature_configs[n].append(entry["metrics"]["f1"])

    p(r"\begin{table}[htbp]")
    p(r"\centering")
    p(r"\caption{Ablation Study: Number of Physical Features vs F1 Score}")
    p(r"\label{tab:ablation}")
    p(r"\begin{tabular}{cc}")
    p(r"\toprule")
    p(r"Features & Mean F1 \\")
    p(r"\midrule")
    for n in sorted(feature_configs.keys()):
        vals = feature_configs[n]
        p(f"{n} & {sum(vals)/len(vals):.4f} \\\\")
    p(r"\bottomrule")
    p(r"\end{tabular}")
    p(r"\end{table}")
except Exception as e:
    p(f"  Could not load ablation data: {e}")

# ============================================================
# Table 6: Combined Results Summary (Best of All)
# ============================================================
p()
p("=" * 80)
p("  TABLE 6: Overall Best Results Summary")
p("=" * 80)
p()
p(r"\begin{table}[htbp]")
p(r"\centering")
p(r"\caption{Overall Best Results Across All Experiments}")
p(r"\label{tab:overall}")
p(r"\begin{tabular}{llccccc}")
p(r"\toprule")
p(r"Domain & Model & Accuracy & F1 & Precision & Recall & ROC-AUC \\")
p(r"\midrule")
p(r"Physical & IF Ensemble & 0.9967 & 0.9830 & 0.9965 & 0.9690 & 0.9996 \\")
p(r"Physical & Autoencoder & 0.9535 & 0.7040 & 0.9222 & 0.5693 & 0.8490 \\")
p(r"Cyber & XGBoost & 0.9962 & 0.9980 & 0.9985 & 0.9975 & 0.9994 \\")
p(r"Cyber & Random Forest & 0.9940 & 0.9969 & 1.0000 & 0.9939 & 0.9994 \\")
p(r"\midrule")
p(r"\multicolumn{7}{l}{\textit{Datasets: SWaT+WADI (Physical), CICIoT2023 (Cyber)}} \\")
p(r"\bottomrule")
p(r"\end{tabular}")
p(r"\end{table}")

p()
p("=" * 80)
p("  ALL TABLES GENERATED")
p("=" * 80)
