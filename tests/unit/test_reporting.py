from pathlib import Path

import pytest

from src.reporting import (
    NoMistakeRecordsError,
    ReportConflictError,
    ReportDataError,
    discover_mistake_records,
    read_report_snapshot,
    render_learning_report,
    save_report_atomic,
)
from src.schemas import PracticeItem


def _write_mistake(path: Path, *, mistake_id: str = "mistake-test") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\n"
        "schema_version: 1\n"
        f"id: {mistake_id}\n"
        "subject: english\n"
        "topic: present-perfect\n"
        "status: needs-review\n"
        'created_at: "2026-08-15"\n'
        "review_count: 0\n"
        "next_review_at: null\n"
        'source: "chat"\n'
        "---\n\n"
        "# 错题记录\n\n"
        "- 学科：英语\n"
        "- 题型：语法填空\n"
        "- 原题：I ____ (see) the film three times.\n"
        "- 我的答案：saw\n"
        "- 正确答案：have seen\n"
        "- 正确思路：累计次数使用现在完成时。\n"
        "- 错因：忽略了 three times。\n"
        "- 知识点：现在完成时\n"
        "- 下次提醒：先圈出次数线索。\n",
        encoding="utf-8",
    )
    return path


def test_discover_mistake_records_reads_strict_formal_records(tmp_path: Path) -> None:
    records_root = tmp_path / "records"
    _write_mistake(records_root / "english" / "mistake-test.md")

    records = discover_mistake_records(records_root)

    assert len(records) == 1
    assert records[0].mistake_id == "mistake-test"
    assert records[0].subject == "english"
    assert records[0].topic == "present-perfect"
    assert records[0].student_answer == "saw"


def test_discover_mistake_records_rejects_empty_or_malformed_data(
    tmp_path: Path,
) -> None:
    records_root = tmp_path / "records"

    with pytest.raises(NoMistakeRecordsError, match="没有正式错题"):
        discover_mistake_records(records_root)

    bad_path = records_root / "english" / "mistake-bad.md"
    bad_path.parent.mkdir(parents=True)
    bad_path.write_text("# 缺少 front matter\n", encoding="utf-8")

    with pytest.raises(ReportDataError, match="mistake-bad.md"):
        discover_mistake_records(records_root)


def test_report_hides_reference_answer_and_updates_atomically(tmp_path: Path) -> None:
    student_root = tmp_path / "student"
    records_root = student_root / "mistakes" / "records"
    report_path = student_root / "reports" / "learning-review.md"
    records = discover_mistake_records(
        _write_mistake(records_root / "english" / "mistake-test.md").parents[1]
    )
    practice: PracticeItem = {
        "question": "I ____ (visit) Ningbo four times.",
        "expected_answer": "have visited",
        "reasoning": "four times 表示累计次数。",
        "subject": "english",
        "topic": "present-perfect",
        "source_record_ids": ["mistake-test"],
    }
    markdown = render_learning_report(
        records,
        version=1,
        request_id="review-1",
        generated_at="2026-08-15T12:00:00+08:00",
        summary="本次应优先复习现在完成时。",
        patterns=["容易忽略累计次数线索。"],
        action_steps=["做题前先圈出 times。"],
        practice_item=practice,
    )

    assert "mistake-test" in markdown
    assert practice["question"] in markdown
    assert practice["expected_answer"] not in markdown
    assert practice["reasoning"] not in markdown

    save_report_atomic(
        report_path,
        markdown,
        expected_digest=None,
        allowed_root=student_root,
    )
    snapshot = read_report_snapshot(report_path)
    assert snapshot.version == 1
    assert snapshot.request_id == "review-1"

    with pytest.raises(ReportConflictError, match="已被其他操作修改"):
        save_report_atomic(
            report_path,
            markdown.replace("version: 1", "version: 2"),
            expected_digest=None,
            allowed_root=student_root,
        )
