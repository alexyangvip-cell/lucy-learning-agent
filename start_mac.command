#!/bin/bash

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

/bin/bash "$SCRIPT_DIR/start_mac.sh" "$@"
EXIT_CODE=$?

if [ "$EXIT_CODE" -ne 0 ] && [ "$EXIT_CODE" -ne 130 ]; then
    echo
    read -r -p "启动未完成。按回车键关闭此窗口..."
fi

exit "$EXIT_CODE"
