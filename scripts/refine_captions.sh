#!/bin/bash
set -e
PY=python3

echo "Running refinement wrapper..."
$PY - <<'PYCODE'
from src.vlm_refiner_wrapper import refine_results_with_vlm
import os, glob
# change this to where your test images are
test_folder = os.environ.get("TEST_FOLDER", "/content/drive/MyDrive/image_captioning/datasets/coco2014/val2014")
test_images = sorted([os.path.join(test_folder, f) for f in os.listdir(test_folder) if f.endswith(".jpg")])[:100]
out_json = "results/results_refined.json"
refine_results_with_vlm(test_images, out_json)
PYCODE
