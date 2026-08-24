from pathlib import Path

import pytest

from src.storage import (
    MAX_MARKDOWN_BYTES,
    MISTAKES_INBOX_PATH,
    MISTAKES_RECORDS_PATH,
    MISTAKES_ROOT,
    STUDENT_ROOT,
    FileAlreadyExistsError,
    StorageError,
    load_markdown,
    save_markdown,
)


def test_mistake_paths_separate_inbox_and_normalized_records() -> None:
    assert MISTAKES_ROOT == STUDENT_ROOT / "mistakes"
    assert MISTAKES_INBOX_PATH == MISTAKES_ROOT / "inbox"
    assert MISTAKES_RECORDS_PATH == MISTAKES_ROOT / "records"


def test_load_markdown_reads_absolute_utf8_file_inside_allowed_root(
    tmp_path: Path,
) -> None:
    mistakes_path = tmp_path / "student" / "mistakes"
    mistakes_path.mkdir(parents=True)
    source_path = mistakes_path / "english.md"
    source_path.write_text("错题1\n原题：测试题\n", encoding="utf-8")

    content = load_markdown(source_path, allowed_root=mistakes_path)

    assert content == "错题1\n原题：测试题"


def test_load_markdown_resolves_relative_path_from_allowed_root(
    tmp_path: Path,
) -> None:
    mistakes_path = tmp_path / "student" / "mistakes"
    mistakes_path.mkdir(parents=True)
    (mistakes_path / "english.md").write_text("两道错题", encoding="utf-8")

    content = load_markdown("english.md", allowed_root=mistakes_path)

    assert content == "两道错题"


@pytest.mark.parametrize("requested_path", ["../outside.md", "notes.txt"])
def test_load_markdown_rejects_unsafe_path_or_extension(
    tmp_path: Path,
    requested_path: str,
) -> None:
    mistakes_path = tmp_path / "student" / "mistakes"
    mistakes_path.mkdir(parents=True)
    (tmp_path / "student" / "outside.md").write_text("越界", encoding="utf-8")
    (mistakes_path / "notes.txt").write_text("不是 Markdown", encoding="utf-8")

    with pytest.raises(StorageError):
        load_markdown(requested_path, allowed_root=mistakes_path)


def test_load_markdown_rejects_absolute_path_outside_allowed_root(
    tmp_path: Path,
) -> None:
    mistakes_path = tmp_path / "student" / "mistakes"
    mistakes_path.mkdir(parents=True)
    outside_path = tmp_path / "outside.md"
    outside_path.write_text("越界", encoding="utf-8")

    with pytest.raises(StorageError, match="只能读取"):
        load_markdown(outside_path, allowed_root=mistakes_path)


def test_load_markdown_rejects_symlink_that_escapes_allowed_root(
    tmp_path: Path,
) -> None:
    mistakes_path = tmp_path / "student" / "mistakes"
    mistakes_path.mkdir(parents=True)
    outside_path = tmp_path / "outside.md"
    outside_path.write_text("越界", encoding="utf-8")
    linked_path = mistakes_path / "linked.md"
    linked_path.symlink_to(outside_path)

    with pytest.raises(StorageError, match="只能读取"):
        load_markdown(linked_path, allowed_root=mistakes_path)


def test_load_markdown_rejects_directory_missing_file_and_empty_file(
    tmp_path: Path,
) -> None:
    mistakes_path = tmp_path / "student" / "mistakes"
    mistakes_path.mkdir(parents=True)
    empty_path = mistakes_path / "empty.md"
    empty_path.write_text("", encoding="utf-8")

    with pytest.raises(StorageError, match="普通文件"):
        load_markdown(mistakes_path, allowed_root=mistakes_path)
    with pytest.raises(StorageError, match="普通文件"):
        load_markdown("missing.md", allowed_root=mistakes_path)
    with pytest.raises(StorageError, match="内容为空"):
        load_markdown(empty_path, allowed_root=mistakes_path)


def test_load_markdown_rejects_oversized_and_non_utf8_files(
    tmp_path: Path,
) -> None:
    mistakes_path = tmp_path / "student" / "mistakes"
    mistakes_path.mkdir(parents=True)
    oversized_path = mistakes_path / "oversized.md"
    oversized_path.write_bytes(b"x" * (MAX_MARKDOWN_BYTES + 1))
    invalid_utf8_path = mistakes_path / "invalid.md"
    invalid_utf8_path.write_bytes(b"\xff\xfe")

    with pytest.raises(StorageError, match="过大"):
        load_markdown(oversized_path, allowed_root=mistakes_path)
    with pytest.raises(StorageError, match="UTF-8"):
        load_markdown(invalid_utf8_path, allowed_root=mistakes_path)


def test_save_markdown_writes_utf8_file_inside_allowed_root(tmp_path: Path) -> None:
    student_root = tmp_path / "student"
    output_dir = student_root / "reports"

    saved_path = save_markdown(
        output_dir,
        "report.md",
        "\n# 学习报告\n\n今天掌握了现在完成时。\n",
        allowed_root=student_root,
    )

    assert saved_path == output_dir / "report.md"
    assert saved_path.read_text(encoding="utf-8") == (
        "# 学习报告\n\n今天掌握了现在完成时。\n"
    )


@pytest.mark.parametrize(
    "filename",
    ["../escape.md", "nested/report.md", "/tmp/report.md", "report.txt"],
)
def test_save_markdown_rejects_unsafe_filename(
    tmp_path: Path,
    filename: str,
) -> None:
    student_root = tmp_path / "student"

    with pytest.raises(StorageError):
        save_markdown(
            student_root / "reports",
            filename,
            "内容",
            allowed_root=student_root,
        )


def test_save_markdown_rejects_output_directory_outside_allowed_root(
    tmp_path: Path,
) -> None:
    student_root = tmp_path / "student"

    with pytest.raises(StorageError, match="只能写入"):
        save_markdown(
            tmp_path / "outside",
            "report.md",
            "内容",
            allowed_root=student_root,
        )


def test_save_markdown_rejects_symlink_that_escapes_allowed_root(
    tmp_path: Path,
) -> None:
    student_root = tmp_path / "student"
    outside = tmp_path / "outside"
    student_root.mkdir()
    outside.mkdir()
    linked_output = student_root / "linked"
    linked_output.symlink_to(outside, target_is_directory=True)

    with pytest.raises(StorageError, match="只能写入"):
        save_markdown(
            linked_output,
            "report.md",
            "内容",
            allowed_root=student_root,
        )

    assert not (outside / "report.md").exists()


def test_save_markdown_never_overwrites_existing_file(tmp_path: Path) -> None:
    student_root = tmp_path / "student"
    output_dir = student_root / "reports"
    saved_path = save_markdown(
        output_dir,
        "report.md",
        "第一版",
        allowed_root=student_root,
    )

    with pytest.raises(FileAlreadyExistsError, match="已存在"):
        save_markdown(
            output_dir,
            "report.md",
            "第二版",
            allowed_root=student_root,
        )

    assert saved_path.read_text(encoding="utf-8") == "第一版\n"
