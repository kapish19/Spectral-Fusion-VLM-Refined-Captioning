import os
import json
from pathlib import Path

import importlib.util
import sys

ORIG = os.path.join(os.path.dirname(__file__), "original_imgcap.py")
spec = importlib.util.spec_from_file_location("original_imgcap_module", ORIG)
orig_mod = importlib.util.module_from_spec(spec)
sys.modules["original_imgcap_module"] = orig_mod
spec.loader.exec_module(orig_mod)

# The original script sets up YOLO as `yolo` and exposes `evaluate_image`, and savedir `save_dir`.
# We'll attempt to use those; if not present, we gracefully degrade.

def refine_results_with_vlm(test_images, out_json_path):
    """
    For each image in test_images (list of paths), call the original evaluate_image to get a draft caption,
    then call the LLaVARefiner (rule-based fallback) defined in the original file (if present).
    Writes results as a JSON list.
    """
    print("Running VLM refinement wrapper...")
    results = []
    for img_path in test_images:
        try:
            # draft generation
            if hasattr(orig_mod, "evaluate_image"):
                draft = orig_mod.evaluate_image(img_path)
            elif hasattr(orig_mod, "evaluate"):
                # some versions return (result, attention)
                out = orig_mod.evaluate(img_path)
                if isinstance(out, tuple):
                    draft = ' '.join(out[0]) if isinstance(out[0], (list, tuple)) else out[0]
                else:
                    draft = out
            else:
                raise RuntimeError("Original module does not expose evaluate_image/evaluate")

            draft = str(draft).replace("<start>", "").replace("<end>", "").strip()

            # call refiner if available
            if hasattr(orig_mod, "vlm_refiner"):
                refined, grounding = orig_mod.vlm_refiner.refine(img_path, draft)
            elif hasattr(orig_mod, "LLaVARefiner") and hasattr(orig_mod, "yolo"):
                # instantiate fallback refiner from original definitions
                RF = orig_mod.LLaVARefiner(orig_mod.yolo, use_llm=False)
                refined, grounding = RF.refine(img_path, draft)
            else:
                refined, grounding = draft, {}

            results.append({
                "image_id": os.path.basename(img_path),
                "draft_caption": draft,
                "refined_caption": refined,
                "grounding": grounding
            })
        except Exception as e:
            print(f"Skipping {img_path} due to error: {e}")
            continue

    # save
    os.makedirs(os.path.dirname(out_json_path) or ".", exist_ok=True)
    with open(out_json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved refined results to {out_json_path}")
    return results
