"""Quick test of real data loaders."""
from pathlib import Path
from src.datasets.ciciot2023 import CICIoT2023DatasetLoader
from src.datasets.edge_iiotset import EdgeIIoTSetDatasetLoader

print("=== CICIoT2023 ===", flush=True)
loader = CICIoT2023DatasetLoader(Path("data/raw/ciciot2023"))
df = loader.load()
print(f"Loaded: {df.shape}", flush=True)
loader.validate(df)
df = loader.clean(df)
df = loader.transform(df)
print(f"After clean+transform: {df.shape}", flush=True)
print(f"Features: {len(loader.get_features())}", flush=True)
print(f"Label col: {loader.LABEL_COLUMN}", flush=True)
al = df["attack_label"]
print(f"Attack rate: {al.mean():.2%}", flush=True)

print("\n=== Edge-IIoTSet ===", flush=True)
loader2 = EdgeIIoTSetDatasetLoader(Path("data/raw/edge_iiotset"))
df2 = loader2.load()
print(f"Loaded: {df2.shape}", flush=True)
loader2.validate(df2)
df2 = loader2.clean(df2)
df2 = loader2.transform(df2)
print(f"After clean+transform: {df2.shape}", flush=True)
print(f"Features: {len(loader2.get_features())}", flush=True)
print(f"Label col: {loader2.LABEL_COLUMN}", flush=True)
al2 = df2["attack_label"]
print(f"Attack rate: {al2.mean():.2%}", flush=True)
