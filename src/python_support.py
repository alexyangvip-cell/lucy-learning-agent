"""课程支持的 Python 版本范围。"""

REQUIRES_PYTHON = ">=3.11,<3.15"
MINIMUM_PYTHON_SERIES = (3, 11)
MAXIMUM_PYTHON_SERIES = (3, 15)
SUPPORTED_PYTHON_LABEL = "Python 3.11.x 至 3.14.x"


def parse_python_series(version: str) -> tuple[int, int] | None:
    """将 Python 版本解析为 major/minor，无法解析时返回 None。"""

    parts = version.strip().split(".")
    if len(parts) < 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def is_supported_python_series(series: tuple[int, int]) -> bool:
    """判断 major/minor 是否位于课程支持范围内。"""

    return MINIMUM_PYTHON_SERIES <= series < MAXIMUM_PYTHON_SERIES
