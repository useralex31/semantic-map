#!/bin/bash
set -euo pipefail

echo "=== Precompute Neighbors Job ==="
echo "Date: $(date -Iseconds)"
echo "Host: $(hostname)"
echo "Python: $(python3 --version 2>&1)"

# Discover mount path
SCRIPT_DIR=""
for candidate in \
    "/work/TBIC-MA-NO/scripts" \
    "/1097013/TBIC-MA-NO/scripts" \
    "/work/scripts"; do
    if [ -f "$candidate/precompute_neighbors.py" ]; then
        SCRIPT_DIR="$candidate"
        break
    fi
done

if [ -z "$SCRIPT_DIR" ]; then
    echo "ERROR: Cannot find precompute_neighbors.py"
    find /work -maxdepth 3 -type f -name "*.py" 2>/dev/null || true
    exit 1
fi

echo "Scripts found at: $SCRIPT_DIR"

# Install dependencies (Python 3.9 compatible)
echo "Installing dependencies..."
pip install --quiet --no-cache-dir \
    "pandas==2.0.3" \
    "pyarrow==17.0.0" \
    "numpy==1.26.4"

echo "Dependencies installed."

# Run
cd "$SCRIPT_DIR"
python3 precompute_neighbors.py
RC=$?

echo "Exit code: $RC"
echo "=== Job Complete ==="
exit $RC
