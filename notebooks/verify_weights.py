"""
verify_weights.py  --  Infinity Shield O1 helper

Checks whether a saved MobileNet-SSD .pt file is the GOOD Option A model.
It does NOT train anything -- it just loads the weights and runs the
test-set evaluation, which takes a few minutes.

Expected result for the correct file:
    [test] TP=21 FP=6 FN=21  ->  recall=0.500  FP-rate=0.222

Usage (from project root, virtualenv active):
    python notebooks/verify_weights.py "C:\\Users\\snort\\Desktop\\mobilenet_ssd_plastic_v1.pt"

If you don't pass a path, it checks the file currently in models/checkpoints/.
"""

import sys
from pathlib import Path

import torch

# Reuse everything already written in the benchmark script so the
# model, dataset, and evaluation logic are guaranteed identical.
from benchmark_mobilenet import (
    build_model,
    evaluate,
    PlasticBallDataset,
    collate_fn,
    DEVICE,
    WEIGHTS_DIR,
)
from torch.utils.data import DataLoader


def main():
    # --- Decide which .pt file to check ---
    if len(sys.argv) > 1:
        weights_path = Path(sys.argv[1])
    else:
        weights_path = WEIGHTS_DIR / "mobilenet_ssd_plastic_v1.pt"

    if not weights_path.exists():
        print(f"[ERROR] File not found: {weights_path}")
        sys.exit(1)

    print(f"\n=== Verifying weights file ===")
    print(f"File: {weights_path}")
    print(f"Size: {weights_path.stat().st_size / 1e6:.1f} MB\n")

    # --- Build the model skeleton, then load the saved weights into it ---
    model = build_model()
    state = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state)
    print("[OK] Weights loaded into model successfully.\n")

    # --- Run test-set evaluation only (no training) ---
    print("Evaluating on test set (this takes a few minutes)...")
    test_loader = DataLoader(PlasticBallDataset("test"),
                             batch_size=1, collate_fn=collate_fn)
    recall, fp_rate = evaluate(model, test_loader, "test")

    # --- Verdict ---
    print("\n=== VERDICT ===")
    if abs(recall - 0.500) < 0.01 and abs(fp_rate - 0.222) < 0.01:
        print("  MATCH -- this is the good Option A model (recall 0.50).")
        print("  Safe to keep / replace into models/checkpoints/.")
    else:
        print(f"  MISMATCH -- got recall={recall:.3f}, FP-rate={fp_rate:.3f}")
        print("  This is NOT the expected Option A model.")
        print("  Do not trust this file -- you will need to re-run training.")


if __name__ == "__main__":
    main()
