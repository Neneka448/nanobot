from pathlib import Path

from nanobot.agent.memory import LongShortTermMemory, MemoryStore
from nanobot.utils.helpers import sync_workspace_templates


def test_sync_workspace_templates_initializes_memory_store_files_and_skill(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    memory = MemoryStore(workspace)

    added = sync_workspace_templates(workspace, memory=memory, silent=True)

    assert "memory/MEMORY.md" in added
    assert "memory/HISTORY.md" in added
    assert "skills/memory/SKILL.md" in added
    assert (workspace / "memory" / "MEMORY.md").exists()
    assert (workspace / "skills" / "memory" / "SKILL.md").exists()


def test_sync_workspace_templates_initializes_long_short_term_files_and_skill(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    memory = LongShortTermMemory(workspace)

    added = sync_workspace_templates(workspace, memory=memory, silent=True)

    assert "memory/LONG_TERM_MEMORY.md" in added
    assert "memory/SHORT_TERM_MEMORY.md" in added
    assert "memory/HISTORY.md" in added
    assert "skills/memory/SKILL.md" in added
    assert (workspace / "memory" / "LONG_TERM_MEMORY.md").exists()
    assert (workspace / "memory" / "SHORT_TERM_MEMORY.md").exists()


def test_sync_workspace_templates_preserves_existing_memory_files(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    memory = MemoryStore(workspace)
    existing_memory = workspace / "memory" / "MEMORY.md"
    existing_skill = workspace / "skills" / "memory" / "SKILL.md"
    existing_memory.parent.mkdir(parents=True, exist_ok=True)
    existing_skill.parent.mkdir(parents=True, exist_ok=True)
    existing_memory.write_text("custom-memory", encoding="utf-8")
    existing_skill.write_text("custom-skill", encoding="utf-8")

    added = sync_workspace_templates(workspace, memory=memory, silent=True)

    assert "memory/MEMORY.md" not in added
    assert "skills/memory/SKILL.md" not in added
    assert existing_memory.read_text(encoding="utf-8") == "custom-memory"
    assert existing_skill.read_text(encoding="utf-8") == "custom-skill"