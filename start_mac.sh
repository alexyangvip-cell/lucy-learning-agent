#!/bin/bash

set -u

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
export PYTHONUTF8=1

for PYTHON_COMMAND in python3.14 python3.13 python3.12 python3.11 python3; do
    if ! command -v "$PYTHON_COMMAND" >/dev/null 2>&1; then
        continue
    fi
    if "$PYTHON_COMMAND" -c \
        'import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] < (3, 15) else 1)' \
        >/dev/null 2>&1; then
        exec "$PYTHON_COMMAND" "$SCRIPT_DIR/scripts/launch_app.py" "$@"
    fi
done

echo "启动失败：未找到 Python 3.11.x 至 3.14.x。"
echo "建议安装 Python 3.14，然后重新双击 start_mac.command。"
exit 1
