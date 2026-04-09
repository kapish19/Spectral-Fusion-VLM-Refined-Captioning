#!/bin/bash
set -e
PY=python3

echo "Starting training via original script..."
$PY - <<'PYCODE'
import importlib.util, os
spec = importlib.util.spec_from_file_location("orig", os.path.join("src","original_imgcap.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print("Original script executed for training.")
PYCODE
