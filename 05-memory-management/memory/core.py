import re
from collections.abc import Mapping, MutableMapping
from pathlib import Path
import time
from typing import Any
import json
import uuid

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam


def _get_field(obj: Any, key: str, default: Any = None) -> Any:
    """兼容 dict 和 OpenAI SDK 对象的字段读取。"""
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _set_field(obj: Any, key: str, value: Any) -> None:
    """兼容 dict 和 OpenAI SDK 对象的字段设置。"""
    if isinstance(obj, MutableMapping):
        obj[key] = value
    elif isinstance(obj, Mapping):
        raise TypeError(
            f"Cannot set field {key!r}: mapping is immutable"
        )
    else:
        setattr(obj, key, value)


class AgentMemory:
    RECALL_CHAR_LIMIT = 20000
    CONSOLIDATE_THRESHOLD = 10
    CONSOLIDATE_INPUT_CHAR_LIMIT = 20000

    def __init__(self, work_dir: Path, llm_client: OpenAI, model: str):
        self.work_dir = work_dir
        self.memory_dir = self.work_dir / ".memory"
        self.memory_index = self.memory_dir / "MEMORY.md"
        self.client = llm_client
        self.model = model

    def _parse_frontmatter(self, text: str) -> tuple[dict, str]:
        if not text.startswith("---"):
            return {}, text
        parts = text.split("---", 2)
        if len(parts) < 3:
            return {}, text
        meta = {}
        for line in parts[1].strip().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip().strip('"').strip("'")
        return meta, parts[2].strip()
    
    def _read_memory_file(self, filename: str) -> str | None:
        """Read a single memory file's full content."""
        path = self.memory_dir / filename
        if not path.exists():
            return None
        return path.read_text()

    def _list_memory_files(self) -> list[dict]:
        """List all memory files with metadata."""
        result = []
        for f in sorted(self.memory_dir.glob("*.md")):
            if f.name == "MEMORY.md":
                continue
            raw = f.read_text()
            meta, body = self._parse_frontmatter(raw)
            result.append({
                "filename": f.name,
                "name": meta.get("name", f.stem),
                "description": meta.get("description", ""),
                "type": meta.get("type", "user"),
                "body": body,
            })
        return result

    def _select_relevant_memories_llm(self, files: list[dict], recent: str, max_items: int = 5) -> list[str]:
        """Select relevant memory filenames by LLM call."""
        # Build catalog of name + description for LLM to choose from
        catalog_lines = []
        for i, f in enumerate(files):
            catalog_lines.append(f"{i}: {f['name']} — {f['description']}")
        catalog = "\n".join(catalog_lines)
        prompt = (
            "Given the recent conversation and the memory catalog below, "
            "select the indices of memories that are clearly relevant. "
            "Return ONLY a JSON array of integers, e.g. [0, 3]. "
            "If none are relevant, return [].\n\n"
            f"Recent conversation:\n{recent}\n\n"
            f"Memory catalog:\n{catalog}"
        )

        try:
            resonse = self.client.chat.completions.create(model=self.model, messages=[{"role": "user", "content": prompt}], max_tokens=2000)
            assistant_message = resonse.choices[0].message
            text = _get_field(assistant_message, "content", "").strip()
            # Extract JSON array from response
            match = re.search(r'\[.*?\]', text, re.DOTALL)
            if match:
                indices = json.loads(match.group())
                selected = []
                for idx in indices:
                    if isinstance(idx, int) and 0 <= idx < len(files):
                        selected.append(files[idx]["filename"])
                        if len(selected) >= max_items:
                            break
                return selected
            return []
        except Exception:
            print(f"Error in LLM call for memory selection: {prompt}")
            return []

    def _select_relevant_memories(self, messages: list[ChatCompletionMessageParam], max_items: int = 5) -> list[str]:
        """
        Select relevant memory filenames by matching recent conversation against
        memory names/descriptions. Uses a simple LLM call (or falls back to keyword
        matching on name+description).
        """
        # List all memory files
        files = list(self._list_memory_files())
        if not files:
            return []
        
        # Collect recent user text for context
        recent_texts = []
        for msg in reversed(messages):
            if len(recent_texts) >= 3:
                break
            if _get_field(msg, "role") == "user":
                content = _get_field(msg, "content", "")
                recent_texts.append(content)

        recent = " ".join(reversed(recent_texts))[:2000]
        if not recent.strip():
            return []

        # Try LLM selection first
        relevant_file_names = self._select_relevant_memories_llm(files, recent, max_items)
        if relevant_file_names and len(relevant_file_names) > 0:
            return relevant_file_names
        
        # Fallback: keyword matching on name + description
        keywords = [w.lower() for w in recent.split() if len(w) > 3]
        selected = []
        for f in files:
            text = (f["name"] + " " + f["description"]).lower()
            if any(kw in text for kw in keywords):
                selected.append(f["filename"])
                if len(selected) >= max_items:
                    break
        return selected

    def _rebuild_index(self):
        """Rebuild MEMORY.md index from all memory files."""
        lines = []
        for f in sorted(self.memory_dir.glob("*.md")):
            if f.name == "MEMORY.md":
                continue
            raw = f.read_text()
            meta, body = self._parse_frontmatter(raw)
            name = meta.get("name", f.stem)
            desc = meta.get("description", body.split("\n")[0][:80])
            lines.append(f"- [{name}]({f.name}) — {desc}")
        self.memory_index.write_text("\n".join(lines) + "\n" if lines else "")

    def _write_memory_file(self, name: str, mem_type: str, description: str, body: str):
        """Write a single memory file with YAML frontmatter."""
        slug = name.lower().replace(" ", "-").replace("/", "-")
        filename = f"{slug}.md"
        filepath = self.memory_dir / filename
        filepath.write_text(
            f"---\nname: {name}\ndescription: {description}\ntype: {mem_type}\n---\n\n{body}\n"
        )
        self._rebuild_index()
        return filepath

    def read_memory_index(self) -> str:
        """Read MEMORY.md index (injected into SYSTEM every turn)."""
        if not self.memory_index.exists():
            return ""
        text = self.memory_index.read_text().strip()
        return text if text else ""

    # Agent loop 之前系统提示词中注入 relevant_memories 标签，包含相关记忆的文本
    def load_memories(self, messages: list[ChatCompletionMessageParam]) -> str:
        """Load relevant memory content for injection into context."""
        selected_files = self._select_relevant_memories(messages)
        if not selected_files:
            return ""
        parts = ["<relevant_memories>"]
        for filename in selected_files:
            content = self._read_memory_file(filename)
            if content:
                parts.append(content)
        parts.append("</relevant_memories>")
        return "\n\n".join(parts)

    # Agent loop 之后运行，提取新记忆
    def extract_memories(self, messages: list[ChatCompletionMessageParam]) -> int:
        """Extract new memories from recent dialogue. Runs after each turn."""
        # Collect recent conversation text
        dialogue_parts = []
        for msg in messages[-10:]:
            role = _get_field(msg, "role", "?")
            content = _get_field(msg, "content", "")
            if content.strip():
                dialogue_parts.append(f"{role}: {content}")
        dialogue = "\n".join(dialogue_parts)
        # if dialogue.strip():
        #     return
        # Check existing memories to avoid duplicates
        existing = self._list_memory_files()
        existing_desc = "\n".join(f"- {m['name']}: {m['description']}" for m in existing) if existing else "(none)"
        prompt = (
            "Extract user preferences, constraints, or project facts from this dialogue.\n"
            "Return a JSON array. Each item: {name, type, description, body}.\n"
            "- name: short kebab-case identifier (e.g. 'user-preference-tabs')\n"
            "- type: one of 'user' (user preference), 'feedback' (guidance), "
            "'project' (project fact), 'reference' (external pointer)\n"
            "- description: one-line summary for index lookup\n"
            "- body: full detail in markdown\n"
            "If nothing new or already covered by existing memories, return [].\n\n"
            f"Existing memories:\n{existing_desc}\n\n"
            f"Dialogue:\n{dialogue[:4000]}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=[{"role": "user", "content": prompt}], max_tokens=800
            )
            assistant_message = response.choices[0].message
            text = _get_field(assistant_message, "content", "").strip()
            # Extract JSON array from response
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if not match:
                return 0
            items = json.loads(match.group())
            if not items:
                return 0
            count = 0
            for mem in items:
                name = mem.get("name", f"memory_{int(time.time())}")
                mem_type = mem.get("type", "user")
                desc = mem.get("description", "")
                body = mem.get("body", "")
                if desc and body:
                    self._write_memory_file(name, mem_type, desc, body)
                    count += 1
            if count:
                print(f"\n\033[33m[Memory: extracted {count} new memories]\033[0m")
            return count
        except Exception:
            print("Error extracting memories")
            return 0
            
    def consolidate_memories(self):
        """Merge duplicate/stale memories. Triggered when file count ≥ threshold."""
        files = self._list_memory_files()
        if len(files) < self.CONSOLIDATE_THRESHOLD:
            return
        catalog = "\n\n".join(
            f"## {f['filename']}\nname: {f['name']}\ndescription: {f['description']}\n{f['body']}"
            for f in files
        )
        prompt = (
            "Consolidate the following memory files. Rules:\n"
            "1. Merge duplicates into one\n"
            "2. Remove outdated/contradicted memories\n"
            "3. Keep the total under 30 memories\n"
            "4. Preserve important user preferences above all\n"
            "Return a JSON array. Each item: {name, type, description, body}.\n\n"
            f"{catalog[:16000]}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=[{"role": "user", "content": prompt}], max_tokens=3000
            )
            assistant_message = response.choices[0].message
            text = _get_field(assistant_message, "content", "").strip()
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if not match:
                return
            items = json.loads(match.group())
            if not items or len(items) == 0:
                return
            # Remove old memory files (keep MEMORY.md)
            for f in self.memory_dir.glob("*.md"):
                if f.name != "MEMORY.md":
                    try:
                        f.unlink()
                    except ValueError:
                        continue
            for mem in items:
                name = mem.get("name", f"memory_{int(time.time())}")
                mem_type = mem.get("type", "user")
                desc = mem.get("description", "")
                body = mem.get("body", "")
                if desc and body:
                    self._write_memory_file(name, mem_type, desc, body)
            self._rebuild_index()
            print(f"\n\033[33m[Memory: consolidated {len(files)} → {len(items)} memories]\033[0m")
        except Exception:
            print("Error consolidating memories")