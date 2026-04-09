set -e
PY=python3

echo "Running feature extraction using original script..."
$PY - <<'PYCODE'
# execute the original script with a flag to only run feature extraction if it supports it.
# If original script runs top-to-bottom, it will perform extraction as part of its flow.
import importlib.util, sys, os
spec = importlib.util.spec_from_file_location("orig", os.path.join("src","original_imgcap.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print("Original script executed for feature extraction.")
PYCODE
