"""
Cross-session memory extraction.

At the end of each turn (when the model stops without pending tool calls),
this module examines the recent conversation and extracts durable facts
— user preferences, project conventions, error fixes, workflow patterns —
into persistent files under  ~/.codeclaw/projects/<hash>/memory/.

These files are loaded by MemoryFileManager on subsequent sessions, giving
the agent cross-session continuity.
"""

import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Optional

EXTRACT_MEMORY_PROMPT = """\
You are a memory extraction subagent. Analyse the most recent conversation \
messages and extract durable facts worth remembering for future sessions.

Types of memories to save:
1. **User Preferences** — coding style, language, framework choices, response format
2. **Project Conventions** — directory layout, naming patterns, build/test commands
3. **Error Fixes** — specific errors encountered and their solutions
4. **Workflow Patterns** — commands run frequently, deployment procedures

Do NOT save:
- Transient task details that won't matter next session
- Obvious or universal coding best practices
- Anything the user explicitly said to forget
- Secrets, tokens, passwords, or credentials

{existing_section}

Output a JSON array of memory objects. Each object:
  {{"title": "short descriptive title", "content": "detailed memory content", "type": "preference|convention|error_fix|workflow"}}

If there is nothing worth saving, output an empty array: []
Output ONLY the JSON array, no other text.
"""


class MemoryExtractor:
    """
    Extracts durable memories from the conversation and persists them
    to disk so they can be loaded in future sessions.

    Memory directory:  ~/.codeclaw/projects/<project_hash>/memory/
    """

    def __init__(self, compactor=None, project_dir: str = None):
        self.compactor = compactor
        self.project_dir = project_dir or os.getcwd()
        self._last_extracted_msg_uuid: Optional[str] = None
        self._turns_since_last_extraction = 0
        self._extraction_in_progress = False
        self._extraction_lock = asyncio.Lock()
        self._min_turns_between_extractions = 1
        self._memory_dir: Optional[str] = None

    @property
    def memory_dir(self) -> str:
        if self._memory_dir is None:
            project_hash = hashlib.sha256(
                os.path.abspath(self.project_dir).encode("utf-8")
            ).hexdigest()[:16]
            home = str(Path.home())
            self._memory_dir = os.path.join(
                home, ".codeclaw", "projects", project_hash, "memory"
            )
        return self._memory_dir

    def record_turn(self):
        self._turns_since_last_extraction += 1

    def should_extract(self) -> bool:
        return self._turns_since_last_extraction >= self._min_turns_between_extractions

    async def maybe_extract(self, messages: list):
        """Fire-and-forget extraction at turn end."""
        if not self.should_extract():
            return

        if self._extraction_in_progress:
            return

        asyncio.ensure_future(self._run_extraction(messages))

    async def _run_extraction(self, messages: list):
        if self._extraction_lock.locked():
            return

        async with self._extraction_lock:
            self._extraction_in_progress = True
            try:
                new_messages = self._get_new_messages(messages)
                if not new_messages:
                    return

                conversation = self._build_conversation_context(new_messages)
                existing_manifest = self._scan_existing_memories()

                memories = await self._call_llm_for_extraction(
                    conversation, existing_manifest,
                )

                if memories:
                    self._write_memories(memories)

                self._turns_since_last_extraction = 0
                if messages:
                    last = messages[-1]
                    if isinstance(last, dict) and last.get("_uuid"):
                        self._last_extracted_msg_uuid = last["_uuid"]

            except Exception:
                pass
            finally:
                self._extraction_in_progress = False

    def _get_new_messages(self, messages: list) -> list:
        if self._last_extracted_msg_uuid is None:
            return messages

        found = False
        new_msgs = []
        for msg in messages:
            if not found:
                if isinstance(msg, dict) and msg.get("_uuid") == self._last_extracted_msg_uuid:
                    found = True
                continue
            new_msgs.append(msg)

        return new_msgs if found else messages

    def _build_conversation_context(self, messages: list, max_chars: int = 30000) -> str:
        lines = []
        total = 0
        for msg in reversed(messages):
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            if isinstance(content, str):
                line = f"[{role}]: {content[:3000]}"
            elif isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict):
                        btype = block.get("type", "")
                        if btype == "text":
                            parts.append(block.get("text", "")[:2000])
                        elif btype == "tool_use":
                            parts.append(f"[tool_use:{block.get('name', '?')}]")
                        elif btype == "tool_result":
                            parts.append(f"[tool_result: {str(block.get('content', ''))[:300]}]")
                line = f"[{role}]: {' '.join(parts)}"
            else:
                line = f"[{role}]: {str(content)[:2000]}"

            if total + len(line) > max_chars:
                break
            lines.insert(0, line)
            total += len(line)

        return "\n".join(lines)

    def _scan_existing_memories(self) -> str:
        if not os.path.isdir(self.memory_dir):
            return ""

        entries = []
        try:
            for fname in sorted(os.listdir(self.memory_dir)):
                if not fname.endswith(".md"):
                    continue
                fpath = os.path.join(self.memory_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        first_line = f.readline().strip()
                    entries.append(f"- {fname}: {first_line}")
                except Exception:
                    entries.append(f"- {fname}")
        except Exception:
            pass

        if not entries:
            return ""
        return "Existing memories:\n" + "\n".join(entries)

    async def _call_llm_for_extraction(
        self, conversation: str, existing_manifest: str,
    ) -> list:
        if self.compactor is None or self.compactor.client is None:
            return []

        existing_section = ""
        if existing_manifest:
            existing_section = (
                f"\nCheck these existing memories before writing duplicates:\n"
                f"{existing_manifest}\n"
                f"Update existing files rather than creating duplicates.\n"
            )

        prompt = EXTRACT_MEMORY_PROMPT.format(existing_section=existing_section)
        full_prompt = f"<conversation>\n{conversation}\n</conversation>\n\n{prompt}"

        try:
            from codeclaw.core.config import get_model_max_output_tokens, safe_llm_call
            default_tokens, _ = get_model_max_output_tokens(self.compactor.model)

            raw = (await safe_llm_call(
                self.compactor.client, self.compactor.model,
                self.compactor.provider,
                [{"role": "user", "content": full_prompt}],
                default_tokens,
            )).strip()

            if not raw:
                return []

            if raw.startswith("```"):
                lines = raw.split("\n")
                raw = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

            parsed = json.loads(raw)
            if not isinstance(parsed, list):
                return []

            valid = []
            for item in parsed:
                if isinstance(item, dict) and item.get("title") and item.get("content"):
                    valid.append(item)
            return valid

        except Exception:
            return []

    def _write_memories(self, memories: list):
        os.makedirs(self.memory_dir, exist_ok=True)

        for mem in memories:
            title = str(mem.get("title", "untitled"))
            content = str(mem.get("content", ""))
            mem_type = str(mem.get("type", "general"))

            slug = self._slugify(title)
            filename = f"{slug}.md"
            filepath = os.path.join(self.memory_dir, filename)

            counter = 1
            while os.path.exists(filepath) and not self._should_update(filepath, title):
                filename = f"{slug}_{counter}.md"
                filepath = os.path.join(self.memory_dir, filename)
                counter += 1

            md_content = (
                f"---\n"
                f"title: {title}\n"
                f"type: {mem_type}\n"
                f"created: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"project: {os.path.basename(self.project_dir)}\n"
                f"---\n\n"
                f"{content}\n"
            )

            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(md_content)
            except Exception:
                pass

        self._update_index()

    def _should_update(self, filepath: str, title: str) -> bool:
        """Check if an existing file matches the title (update in place)."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("title:"):
                        existing_title = line[6:].strip()
                        return existing_title.lower() == title.lower()
                    if line == "---" and f.tell() > 5:
                        break
        except Exception:
            pass
        return False

    def _update_index(self):
        """Update MEMORY.md index in the memory directory."""
        if not os.path.isdir(self.memory_dir):
            return

        entries = []
        for fname in sorted(os.listdir(self.memory_dir)):
            if not fname.endswith(".md") or fname == "MEMORY.md":
                continue
            fpath = os.path.join(self.memory_dir, fname)
            title = fname[:-3].replace("_", " ").title()
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip().startswith("title:"):
                            title = line.strip()[6:].strip()
                            break
            except Exception:
                pass
            entries.append(f"- [{title}]({fname})")

        if not entries:
            return

        index_path = os.path.join(self.memory_dir, "MEMORY.md")
        try:
            with open(index_path, "w", encoding="utf-8") as f:
                f.write("# Persistent Memories\n\n")
                f.write("\n".join(entries))
                f.write("\n")
        except Exception:
            pass

    @staticmethod
    def _slugify(text: str) -> str:
        import re
        slug = text.lower().strip()
        slug = re.sub(r"[^\w\s-]", "", slug)
        slug = re.sub(r"[\s_-]+", "_", slug)
        return slug[:60].strip("_") or "memory"

    def export_state(self) -> dict:
        return {
            "last_extracted_msg_uuid": self._last_extracted_msg_uuid,
            "turns_since_last_extraction": self._turns_since_last_extraction,
            "memory_dir": self.memory_dir,
        }

    def load_state(self, state: dict):
        if not state:
            return
        self._last_extracted_msg_uuid = state.get("last_extracted_msg_uuid")
        self._turns_since_last_extraction = state.get("turns_since_last_extraction", 0)
