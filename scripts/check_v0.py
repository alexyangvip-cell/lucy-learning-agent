"""无需启动 JupyterLab 的 V0 模型连通性检查。"""

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.facade import invoke


def main() -> int:
    parser = argparse.ArgumentParser(description="检查 V0 所选模型的连通性。")
    parser.add_argument(
        "message",
        nargs="?",
        default="请用一句简短的中文介绍你自己。",
        help="发送给 V0 的测试消息。",
    )
    args = parser.parse_args()

    result = invoke("V0", args.message)
    if result["error"]:
        print(f"V0 连通失败：{result['error']}", file=sys.stderr)
        return 1

    print("V0 连通成功。")
    print(result["text"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
