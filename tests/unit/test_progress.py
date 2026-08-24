import json
from pathlib import Path

import pytest

import src.progress as progress_module


def test_load_progress_returns_default_without_creating_file(tmp_path: Path) -> None:
    progress_path = tmp_path / "student" / "progress.json"

    progress = progress_module.load_progress(progress_path)

    assert progress == progress_module.default_progress()
    assert not progress_path.exists()


def test_save_progress_round_trips_with_atomic_timestamp(tmp_path: Path) -> None:
    student_root = tmp_path / "student"
    progress_path = student_root / "progress.json"
    progress = progress_module.default_progress()
    progress["current_lesson"] = "lesson_2"
    progress["completed_modules"] = ["v0", "v1", "v2"]
    progress["last_location"] = "v3"

    saved = progress_module.save_progress(
        progress,
        progress_path,
        allowed_root=student_root,
    )

    assert saved["updated_at"] is not None
    assert progress["updated_at"] is None
    assert progress_module.load_progress(progress_path) == saved
    assert not list(student_root.glob(".progress.json.*.tmp"))


def test_complete_module_orders_modules_and_updates_lesson_location() -> None:
    progress = progress_module.default_progress()
    progress["completed_modules"] = ["v1"]

    completed_v0 = progress_module.complete_module(progress, "v0")
    completed_v3 = progress_module.complete_module(completed_v0, "v3")

    assert progress["completed_modules"] == ["v1"]
    assert completed_v0["completed_modules"] == ["v0", "v1"]
    assert completed_v0["current_lesson"] == "lesson_1"
    assert completed_v0["last_location"] == "v0"
    assert completed_v3["completed_modules"] == ["v0", "v1", "v3"]
    assert completed_v3["current_lesson"] == "lesson_2"
    assert completed_v3["last_location"] == "v3"


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"schema_version": True}, "schema_version"),
        ({"completed_modules": ["v0", "v0"]}, "重复"),
        ({"last_location": "lesson_1"}, "last_location"),
        ({"updated_at": "2026-08-15T12:00:00"}, "时区"),
        ({"unexpected": "value"}, "未知字段"),
    ],
)
def test_validate_progress_rejects_invalid_schema(update, message: str) -> None:
    data = dict(progress_module.default_progress())
    data.update(update)

    with pytest.raises(progress_module.ProgressDataError, match=message):
        progress_module.validate_progress(data)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"not-json", "有效 JSON"),
        (b"\xff\xfe", "UTF-8"),
        (json.dumps([]).encode("utf-8"), "JSON 对象"),
    ],
)
def test_load_progress_rejects_corrupt_file(
    tmp_path: Path,
    payload: bytes,
    message: str,
) -> None:
    progress_path = tmp_path / "progress.json"
    progress_path.write_bytes(payload)

    with pytest.raises(progress_module.ProgressDataError, match=message):
        progress_module.load_progress(progress_path)


def test_save_progress_rejects_target_outside_allowed_root(tmp_path: Path) -> None:
    student_root = tmp_path / "student"

    with pytest.raises(progress_module.ProgressDataError, match="只能写入"):
        progress_module.save_progress(
            progress_module.default_progress(),
            tmp_path / "outside" / "progress.json",
            allowed_root=student_root,
        )


def test_failed_atomic_replace_preserves_previous_progress(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    student_root = tmp_path / "student"
    progress_path = student_root / "progress.json"
    progress_module.save_progress(
        progress_module.default_progress(),
        progress_path,
        allowed_root=student_root,
    )
    previous_payload = progress_path.read_bytes()
    updated = progress_module.load_progress(progress_path)
    updated["completed_modules"] = ["v0"]

    def reject_replace(source: Path, target: Path) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(progress_module.os, "replace", reject_replace)

    with pytest.raises(
        progress_module.ProgressDataError,
        match="原课程进度保持不变",
    ):
        progress_module.save_progress(
            updated,
            progress_path,
            allowed_root=student_root,
        )

    assert progress_path.read_bytes() == previous_payload
    assert not list(student_root.glob(".progress.json.*.tmp"))
