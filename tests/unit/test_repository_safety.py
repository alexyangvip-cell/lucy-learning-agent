import ast
import json
import subprocess
import sys
import tomllib
from pathlib import Path

from src.retrieval import discover_knowledge_cards
from src.python_support import REQUIRES_PYTHON


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_project_metadata_declares_supported_python_series() -> None:
    metadata = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]

    assert metadata["name"] == "agent-course-teen-v1"
    assert metadata["version"] == "1.0.0"
    assert metadata["requires-python"] == ">=3.11,<3.15"
    assert metadata["requires-python"] == REQUIRES_PYTHON


def test_sensitive_files_are_ignored() -> None:
    ignored_entries = {
        line.strip()
        for line in (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert ".env" in ignored_entries
    assert "*.log" in ignored_entries
    assert ".ipynb_checkpoints/" in ignored_entries
    assert ".streamlit/secrets.toml" in ignored_entries
    assert "student/reports/" in ignored_entries
    assert "/student/SOUL.md" in ignored_entries
    assert "/student/OWNER.md" in ignored_entries


def test_personalization_templates_are_tracked_but_live_files_are_ignored() -> None:
    for live_file in ("student/SOUL.md", "student/OWNER.md"):
        result = subprocess.run(
            ["git", "check-ignore", "--quiet", live_file],
            cwd=PROJECT_ROOT,
            check=False,
        )
        assert result.returncode == 0, live_file

    for template in (
        "student/templates/SOUL.md",
        "student/templates/OWNER.md",
    ):
        assert (PROJECT_ROOT / template).is_file()
        result = subprocess.run(
            ["git", "check-ignore", "--quiet", template],
            cwd=PROJECT_ROOT,
            check=False,
        )
        assert result.returncode == 1, template


def test_teacher_environment_uses_jupyterlab_only() -> None:
    dependencies = {
        line.strip().split("==", maxsplit=1)[0].lower()
        for line in (PROJECT_ROOT / "requirements-dev.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.lstrip().startswith(("#", "-r"))
    }

    assert "jupyterlab" in dependencies
    assert "notebook" not in dependencies


def test_teacher_notebooks_have_no_saved_outputs() -> None:
    notebook_paths = sorted((PROJECT_ROOT / "teacher").glob("lesson_*.ipynb"))

    assert notebook_paths
    for notebook_path in notebook_paths:
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        code_cells = [
            cell for cell in notebook["cells"] if cell["cell_type"] == "code"
        ]
        assert code_cells, notebook_path
        assert all(cell["execution_count"] is None for cell in code_cells), notebook_path
        assert all(cell["outputs"] == [] for cell in code_cells), notebook_path


def test_teacher_notebook_code_cells_have_valid_python_syntax() -> None:
    notebook_paths = sorted((PROJECT_ROOT / "teacher").glob("lesson_*.ipynb"))

    assert notebook_paths
    for notebook_path in notebook_paths:
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            ast.parse(
                "".join(cell["source"]),
                filename=f"{notebook_path} cell {cell.get('id', 'unknown')}",
            )


def test_mistake_inbox_and_normalized_records_are_separated() -> None:
    mistakes_root = PROJECT_ROOT / "student" / "mistakes"
    inbox_files = sorted((mistakes_root / "inbox").glob("*.md"))
    record_files = sorted((mistakes_root / "records").glob("*/*.md"))

    assert not list(mistakes_root.glob("*.md"))
    assert inbox_files
    assert all(path.name.startswith("mistake-") for path in record_files)
    for path in record_files:
        content = path.read_text(encoding="utf-8")
        assert content.startswith("---\n")
        assert "schema_version: 1" in content
        assert f"id: {path.stem}" in content
        assert f"subject: {path.parent.name}" in content
        assert "topic: " in content
        assert "status: needs-review" in content
        assert "created_at: \"" in content
        assert "review_count: 0" in content
        assert "next_review_at: null" in content
        assert 'source: "chat"' in content or 'source: "inbox/' in content
        assert "# 错题记录" in content


def test_teacher_v1_dialogue_starts_from_live_student_input() -> None:
    notebook_path = PROJECT_ROOT / "teacher" / "lesson_1_prompt.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    definition = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if "def run_v1_dialogue" in "".join(cell["source"])
    )
    launcher = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if "v1_dialogue_history = run_v1_dialogue()" in "".join(cell["source"])
    )

    assert 'student_message = input("你：")' in definition
    assert 'invoke("V1", student_message, history=history)' in definition
    assert "ask_v1(question)" not in definition
    assert "START_INTERACTIVE_V1" not in launcher
    assert launcher.rstrip().endswith("v1_dialogue_history = run_v1_dialogue()")


def test_teacher_v1_dialogue_passes_each_student_input_with_history() -> None:
    notebook_path = PROJECT_ROOT / "teacher" / "lesson_1_prompt.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    definition_cell = next(
        cell
        for cell in notebook["cells"]
        if "def run_v1_dialogue" in "".join(cell["source"])
    )
    student_inputs = iter(["我的题目", "我的回答", "/exit"])
    calls = []

    def fake_input(prompt: str) -> str:
        assert prompt == "你："
        return next(student_inputs)

    def fake_invoke(stage: str, message: str, *, history: list[dict]) -> dict:
        calls.append((stage, message, [item.copy() for item in history]))
        return {"text": f"针对{message}的一个问题？", "error": None}

    namespace = {"input": fake_input, "invoke": fake_invoke}
    exec("".join(definition_cell["source"]), namespace)

    history = namespace["v1_dialogue_history"]

    assert calls == [
        ("V1", "我的题目", []),
        (
            "V1",
            "我的回答",
            [
                {"role": "user", "content": "我的题目"},
                {"role": "assistant", "content": "针对我的题目的一个问题？"},
            ],
        ),
    ]
    assert history[-2:] == [
        {"role": "user", "content": "我的回答"},
        {"role": "assistant", "content": "针对我的回答的一个问题？"},
    ]


def test_teacher_v2_dialogue_uses_live_input_and_displays_loaded_skill() -> None:
    notebook_path = PROJECT_ROOT / "teacher" / "lesson_2_skill.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    definition_cell = next(
        cell
        for cell in notebook["cells"]
        if cell["id"] == "v2-dialogue-definition"
    )
    definition = "".join(definition_cell["source"])
    assert 'call.get("name") == "load_skill"' in definition
    assert 'call.get("name") == "load_mistake_file"' in definition
    assert 'call.get("name") == "save_mistake"' in definition
    student_inputs = iter(["请帮我整理错题", "/exit"])
    calls = []

    def fake_input(prompt: str) -> str:
        assert prompt == "学生："
        return next(student_inputs)

    def fake_invoke(stage: str, message: str, *, history: list[dict]) -> dict:
        calls.append((stage, message, [item.copy() for item in history]))
        return {
            "text": "错题已经保存。",
            "tool_calls": [
                {
                    "name": "load_skill",
                    "args": {"skill_name": "sorting-out-mistakes"},
                },
                {
                    "name": "load_mistake_file",
                    "args": {"path": "student/mistakes/inbox/english.md"},
                },
                {
                    "name": "save_mistake",
                    "args": {"original_question": "测试题"},
                },
            ],
            "error": None,
        }

    namespace = {"input": fake_input, "invoke": fake_invoke}
    exec("".join(definition_cell["source"]), namespace)

    session = namespace["run_v2_dialogue"]()

    assert calls == [("V2", "请帮我整理错题", [])]
    assert session["history"] == [
        {"role": "user", "content": "请帮我整理错题"},
        {"role": "assistant", "content": "错题已经保存。"},
    ]
    assert session["tool_calls"][0]["name"] == "load_skill"
    assert session["tool_calls"][1]["name"] == "load_mistake_file"
    assert session["tool_calls"][2]["name"] == "save_mistake"


def test_teacher_v2_notebook_compares_skill_routing_scenarios() -> None:
    notebook_path = PROJECT_ROOT / "teacher" / "lesson_2_skill.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    markdown = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )

    assert "什么是现在完成时？" in markdown
    assert "我们玩一个侦探闯关游戏练现在完成时。" in markdown
    assert "把刚才答错的题整理进错题本。" in markdown
    assert "english-quest" in markdown
    assert "sorting-out-mistakes" in markdown
    assert "专属闯关页面" in markdown
    assert "scripts/" in markdown
    assert "assets/" in markdown


def test_english_quest_page_and_bundled_resources_are_wired() -> None:
    app_source = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
    lesson_sources = "\n".join(
        (PROJECT_ROOT / "app_pages" / filename).read_text(encoding="utf-8")
        for filename in ("lesson_1.py", "lesson_2.py")
    )
    skill_root = PROJECT_ROOT / "student" / "skill" / "english-quest"

    assert '"app_pages/english_quest.py"' in app_source
    assert "result_loaded_skill" in lesson_sources
    assert "st.switch_page(ENGLISH_QUEST_PAGE)" in lesson_sources
    assert (skill_root / "scripts" / "quest_state.py").is_file()
    assert (skill_root / "assets" / "detective-board.svg").is_file()


def test_default_knowledge_cards_use_unique_yaml_ids_and_required_sections() -> None:
    knowledge_path = PROJECT_ROOT / "student" / "knowledge"

    cards = discover_knowledge_cards(knowledge_path)

    assert cards
    assert len({card.card_id for card in cards}) == len(cards)
    assert cards[0].card_id == "english-grammar-present-perfect"
    assert cards[0].subject == "english"
    assert cards[0].category == "grammar"
    assert cards[0].source == (
        "student/knowledge/english/grammar/present-perfect.md"
    )
    assert "three times" in cards[0].keywords
    assert "times" in cards[0].aliases
    assert cards[0].core_rule
    assert cards[0].example
    assert cards[0].common_mistake


def test_teacher_v3_dialogue_displays_citations_and_trace() -> None:
    notebook_path = PROJECT_ROOT / "teacher" / "lesson_3_knowledge.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    display_cell = next(
        cell for cell in notebook["cells"] if cell["id"] == "v3-result-display"
    )
    definition_cell = next(
        cell for cell in notebook["cells"] if cell["id"] == "v3-dialogue-definition"
    )
    definition = "".join(definition_cell["source"])
    assert 'invoke("V3", student_message, history=history)' in definition
    assert 'turn_result["citations"]' in definition
    assert 'turn_result["trace"]' in definition

    student_inputs = iter(["three times 是什么线索？", "/exit"])
    calls = []

    def fake_input(prompt: str) -> str:
        assert prompt == "学生："
        return next(student_inputs)

    def fake_invoke(stage: str, message: str, *, history: list[dict]) -> dict:
        calls.append((stage, message, [item.copy() for item in history]))
        return {
            "text": (
                "先观察次数线索？\n\n"
                "知识依据：[english-grammar-present-perfect] 现在完成时"
            ),
            "tool_calls": [],
            "citations": [
                {
                    "id": "english-grammar-present-perfect",
                    "source": (
                        "student/knowledge/english/grammar/present-perfect.md"
                    ),
                    "title": "现在完成时",
                    "matches": [
                        {
                            "field": "例句",
                            "terms": [],
                            "excerpt": "I have read this book three times.",
                            "method": "semantic",
                        }
                    ],
                }
            ],
            "trace": [
                {
                    "status": "hit",
                    "candidates": [{"id": "english-grammar-present-perfect"}],
                }
            ],
            "error": None,
        }

    namespace = {"input": fake_input, "invoke": fake_invoke}
    exec("".join(display_cell["source"]), namespace)
    exec(definition, namespace)

    session = namespace["run_v3_dialogue"]()

    assert calls == [("V3", "three times 是什么线索？", [])]
    assert session["history"][-1]["role"] == "assistant"
    assert session["citations"][0]["id"] == "english-grammar-present-perfect"
    assert session["trace"][0]["status"] == "hit"


def test_teacher_v4_dialogue_uses_one_thread_and_public_facade() -> None:
    notebook_path = PROJECT_ROOT / "teacher" / "lesson_4_workflow.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    assert "from src.facade import chat_v4" in source
    assert 'if project_root.name == "teacher":' in source
    assert "sys.path.insert(0, str(project_root))" in source
    assert 'THREAD_ID = "lesson-4-demo"' in source
    assert "chat_v4(student_message, THREAD_ID)" in source
    assert 'input("你：")' in source
    assert "result['waiting_for']" in source
    assert "StateGraph" not in source
    assert "get_llm" not in source


def test_teacher_v4_import_cell_runs_from_teacher_directory() -> None:
    notebook_path = PROJECT_ROOT / "teacher" / "lesson_4_workflow.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    import_cell = next(
        cell for cell in notebook["cells"] if cell.get("id") == "v4-import"
    )

    completed = subprocess.run(
        [sys.executable, "-c", "".join(import_cell["source"])],
        cwd=notebook_path.parent,
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_streamlit_home_does_not_assemble_or_call_models() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            PROJECT_ROOT / "app.py",
            PROJECT_ROOT / "app_pages" / "home.py",
        )
    )

    assert "get_llm" not in source
    assert "StateGraph" not in source
    assert "create_agent" not in source
    assert "chat_v4" not in source
    assert '"app_pages/lesson_1.py"' in source
    assert '"app_pages/lesson_2.py"' in source
    assert '"pages/1_prompt_and_skill.py"' not in source


def test_streamlit_lesson_1_uses_facade_without_agent_assembly() -> None:
    source = (PROJECT_ROOT / "app_pages" / "lesson_1.py").read_text(
        encoding="utf-8"
    )

    assert "from src.facade import" in source
    assert "get_llm" not in source
    assert "StateGraph" not in source
    assert "create_agent" not in source
    assert "from src.agents" not in source
    assert "write_text(" not in source
    assert "st.chat_input(" in source
    assert "st.form(" not in source


def test_streamlit_lesson_2_uses_facade_without_agent_assembly() -> None:
    source = (PROJECT_ROOT / "app_pages" / "lesson_2.py").read_text(
        encoding="utf-8"
    )

    assert "from src.facade import" in source
    assert "get_llm" not in source
    assert "StateGraph" not in source
    assert "create_agent" not in source
    assert "from src.agents" not in source
    assert "write_text(" not in source
    assert "st.chat_input(" in source
    assert "st.form(" not in source
