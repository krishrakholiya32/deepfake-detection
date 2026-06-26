"""
Copy the trained .onnx model (downloaded from a Kaggle training run) into models/
so Docker can mount it. Run from the project root.
"""
import os
import shutil

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_FILES = ["deepfake_effnetb0.onnx"]

for name in MODEL_FILES:
    src = os.path.join(PROJECT_ROOT, name)
    dst = os.path.join(MODELS_DIR, name)
    if os.path.exists(dst):
        print(f"  Already exists: {name}")
        continue
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"  Copied: {name}")
    else:
        print(f"  NOT FOUND: {name} — download it from your Kaggle training run "
              f"and place it in the project root or models/")

print("\nDone. Model file should now be in models/")
