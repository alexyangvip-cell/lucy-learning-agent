"""课程进度的严格 JSON 读取与原子写入。"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Literal, TypedDict, cast

from src.storage import STUDENT_ROOT


PROGRESS_PATH = STUDENT_ROOT / "progress.json"
MAX_PROGRESS_BYTES = 64 * 1024
CourseLesson = Literal["lesson_1", "lesson_2"]
CourseModule = Literal["v0", "v1", "v2", "v3", "v4"]
CourseLocation = Literal["home", "v0", "v1", "v2", "v3", "v4"]
LESSON_IDS = ("lesson_1", "lesson_2")
MODULE_IDS = ("v0", "v1", "v2", "v3", "v4")
LOCATION_IDS = ("home", *MODULE_IDS)
_PROGRESS_FIELDS = {
    "schema_version",
    "current_lesson",
    "completed_modules",
    "last_location",
    "updated_at",
}


class ProgressDataError(ValueError):
    """课程进度文件不符合稳定数据格式。"""


class LearningProgress(TypedDict):
    """只保存课程位置，不保存聊天或工作流状态。"""

    schema_version: int
    current_lesson: CourseLesson
    completed_modules: list[CourseModule]
    last_location: CourseLocation
    updated_at: str | None


def default_progress() -> LearningProgress:
    """返回尚未开始课程时的默认进度。"""

    return {
        "schema_version": 1,
        "current_lesson": "lesson_1",
        "completed_modules": [],
        "last_location": "home",
        "updated_at": None,
    }


def _validate_timestamp(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ProgressDataError(
            "updated_at 必须是带时区的 ISO 时间或 null。"
        )
    clean_value = value.strip()
    try:
        parsed = datetime.fromisoformat(clean_value)
    except ValueError as exc:
        raise ProgressDataError("updated_at 不是有效的 ISO 时间。") from exc
    if parsed.tzinfo is None:
        raise ProgressDataError("updated_at 必须包含时区。")
    return clean_value


def validate_progress(data: object) -> LearningProgress:
    """严格校验反序列化后的课程进度。"""

    if not isinstance(data, dict):
        raise ProgressDataError("课程进度必须是 JSON 对象。")
    fields = set(data)
    missing = sorted(_PROGRESS_FIELDS - fields)
    unknown = sorted(str(field) for field in fields - _PROGRESS_FIELDS)
    if missing:
        raise ProgressDataError(f"课程进度缺少字段：{'、'.join(missing)}。")
    if unknown:
        raise ProgressDataError(
            f"课程进度包含未知字段：{'、'.join(unknown)}。"
        )
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ProgressDataError("课程进度仅支持 schema_version: 1。")

    current_lesson = data["current_lesson"]
    if current_lesson not in LESSON_IDS:
        raise ProgressDataError("current_lesson 必须是 lesson_1 或 lesson_2。")

    completed_modules = data["completed_modules"]
    if not isinstance(completed_modules, list) or any(
        module not in MODULE_IDS for module in completed_modules
    ):
        raise ProgressDataError("completed_modules 只能包含 v0 到 v4。")
    if len(completed_modules) != len(set(completed_modules)):
        raise ProgressDataError("completed_modules 不能包含重复模块。")

    last_location = data["last_location"]
    if last_location not in LOCATION_IDS:
        raise ProgressDataError("last_location 必须是 home 或 v0 到 v4。")

    return {
        "schema_version": 1,
        "current_lesson": cast(CourseLesson, current_lesson),
        "completed_modules": [
            cast(CourseModule, module) for module in completed_modules
        ],
        "last_location": cast(CourseLocation, last_location),
        "updated_at": _validate_timestamp(data["updated_at"]),
    }


def load_progress(path: str | Path = PROGRESS_PATH) -> LearningProgress:
    """读取课程进度；文件尚不存在时返回默认值且不落盘。"""

    progress_path = Path(path)
    if not progress_path.exists():
        return default_progress()
    if progress_path.is_symlink() or not progress_path.is_file():
        raise ProgressDataError(f"{progress_path} 必须是普通 JSON 文件。")
    try:
        payload = progress_path.read_bytes()
    except OSError as exc:
        raise ProgressDataError(f"无法读取 {progress_path}。") from exc
    if len(payload) > MAX_PROGRESS_BYTES:
        raise ProgressDataError("课程进度文件超过 64 KB。")
    try:
        data = json.loads(payload.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ProgressDataError("课程进度文件不是 UTF-8 JSON。") from exc
    except json.JSONDecodeError as exc:
        raise ProgressDataError("课程进度文件不是有效 JSON。") from exc
    return validate_progress(data)


def complete_module(
    progress: LearningProgress,
    module: CourseModule,
) -> LearningProgress:
    """返回完成指定模块后的新进度，不直接写入磁盘。"""

    clean_progress = validate_progress(progress)
    if module not in MODULE_IDS:
        raise ProgressDataError("完成的模块只能是 v0 到 v4。")

    completed = set(clean_progress["completed_modules"])
    completed.add(module)
    return {
        "schema_version": 1,
        "current_lesson": (
            "lesson_1" if module in {"v0", "v1", "v2"} else "lesson_2"
        ),
        "completed_modules": [
            cast(CourseModule, module_id)
            for module_id in MODULE_IDS
            if module_id in completed
        ],
        "last_location": cast(CourseLocation, module),
        "updated_at": clean_progress["updated_at"],
    }


def save_progress(
    progress: LearningProgress,
    path: str | Path = PROGRESS_PATH,
    *,
    allowed_root: str | Path = STUDENT_ROOT,
) -> LearningProgress:
    """校验进度后，以同目录临时文件原子替换目标 JSON。"""

    clean_progress = validate_progress(progress)
    clean_progress["updated_at"] = datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )

    root_path = Path(allowed_root).resolve()
    progress_path = Path(path)
    target_path = progress_path.resolve()
    try:
        target_path.relative_to(root_path)
    except ValueError as exc:
        raise ProgressDataError(
            f"只能写入 {root_path} 目录及其子目录。"
        ) from exc
    if progress_path.is_symlink():
        raise ProgressDataError(f"{progress_path} 不能是符号链接。")
    if progress_path.exists() and not progress_path.is_file():
        raise ProgressDataError(f"{progress_path} 必须是普通 JSON 文件。")

    payload = json.dumps(clean_progress, ensure_ascii=False, indent=2) + "\n"
    temp_path: Path | None = None
    try:
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.parent.resolve().relative_to(root_path)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=progress_path.parent,
            prefix=f".{progress_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
            temp_path = Path(temporary.name)
        os.replace(temp_path, progress_path)
    except (OSError, ValueError) as exc:
        raise ProgressDataError(
            f"无法写入 {progress_path}，原课程进度保持不变。"
        ) from exc
    finally:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass

    return clean_progress
