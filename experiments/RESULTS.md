# Experiment Results Summary

## Physical Anomaly Detection

### Best Model: IF Ensemble (9 IF + XGBoost)
- **Accuracy:** 99.67%
- **F1 Score:** 0.983
- **Precision:** 99.65%
- **Recall:** 96.90%
- **ROC-AUC:** 0.9996
- **Method:** 9 Isolation Forest models (6 raw + 3 PCA-reduced) → scores stacked + PCA-40 features = 49 features → XGBoost classifier
- **Training data:** 500K rows (SWaT+WADI combined), normal-only
- **Training time:** ~19min IF + ~19s XGBoost

### Autoencoder (Normal-Only Training)
- **Accuracy:** 95.35%
- **F1 Score:** 0.704
- **Precision:** 92.22%
- **Recall:** 56.93%
- **ROC-AUC:** 0.849
- **Architecture:** 178→[256,192,128]→64→[128,192,256]→178 (258K params)
- **Key insight:** Training on normal-only data (correct paradigm) improved from 88% to 95%+ accuracy

## Cyber Threat Detection

### XGBoost (CICIoT2023)
- **Accuracy:** 99.62%
- **F1 Score:** 0.998
- **Precision:** 99.85%
- **Recall:** 99.75%
- **ROC-AUC:** 0.9994
- **Training data:** 500K sample from 7.57M CICIoT2023 rows
- **Features:** 46 network flow features

### Random Forest (CICIoT2023)
- **Accuracy:** 99.40%
- **F1 Score:** 0.997
- **Precision:** 100.00%
- **Recall:** 99.39%
- **ROC-AUC:** 0.9994

### Cross-Dataset Generalization
- **Finding:** 0 common features between CICIoT2023 and Edge-IIoTSet
- Cross-dataset AUC = 0.50 (random) — expected due to incompatible feature spaces
- **Implication:** Feature-aligned benchmarks needed for IoT IDS generalization studies

## Datasets

| Dataset | Type | Rows | Features | Attack Rate | Source |
|---------|------|------|----------|-------------|--------|
| SWaT | Physical | 1,441,719 | 51 | 3.79% | SUTD |
| WADI | Physical | 1,209,601 | 127 | 5.58% | SUTD |
| CICIoT2023 | Cyber | 7,570,414 | 46 | 97.64% | UNB CIC |
| Edge-IIoTSet | Cyber | 156,986 | 57 | 84.52% | AIMaster |

## Generated Figures

1. `fig1_model_comparison.png` — Bar chart comparing all 4 models
2. `fig2_ae_convergence.png` — AE training loss curves
3. `fig3_confusion_matrices.png` — Confusion matrices for all models
4. `fig4_dataset_statistics.png` — Dataset sizes, features, class imbalance
5. `fig5_system_architecture.png` — System architecture diagram
6. `fig6_roc_curves.png` — ROC curves for physical and cyber models

## Training Infrastructure

- **Hardware:** NVIDIA RTX 3050 (4GB VRAM), CPU fallback
- **Stack:** Python 3.13, PyTorch, scikit-learn, XGBoost, Kafka, Spark
- **Reproducibility:** RANDOM_SEED=42, all results from actual experiments
- **Tests:** 138/138 passing
