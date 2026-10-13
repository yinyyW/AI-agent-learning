"""Shared helpers for the 04-planning-coordination module.

The module is deliberately split into two layers:

* **Pure helpers** -- :func:`extract_json`, :func:`render_task_list`,
  :func:`truncate`, :func:`normalize_whitespace` and
  :func:`unique_preserving_order` only use the standard library, so they can be
  unit tested offline and reused by the later planning/coordination scripts.
* **Configuration helpers** -- :func:`get_llm_config`, :func:`load_env` and
  :func:`create_client` read ``.env`` and import ``openai`` lazily, which keeps
  ``import planning_coordination`` cheap and dependency-free.

Typical usage::

    from planning_coordination import create_client, extract_json, get_llm_config

    client = create_client()
    response = client.chat.completions.create(
        model=get_llm_config().model,
        messages=[{"role": "user", "content": "Plan the refactor as JSON."}],
    )
    plan = extract_json(response.choices[0].message.content)
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

__all__ = [
    "ConfigError",
    "JsonExtractionError",
    "LLMConfig",
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL_NAME",
    "ENV_FILE_NAME",
    "PROJECT_MARKERS",
    "create_client",
    "extract_json",
    "find_project_root",
    "get_llm_config",
    "load_env",
    "normalize_whitespace",
    "render_task_list",
    "truncate",
    "unique_preserving_order",
]

#: Default endpoint used by this course module (DeepSeek exposes an
#: OpenAI-compatible API).
DEFAULT_BASE_URL = "https://api.deepseek.com"

#: Default model name, overridable with ``AI_MODEL_NAME``.
DEFAULT_MODEL_NAME = "deepseek-flash"

#: Name of the dotenv file this module looks for.
ENV_FILE_NAME = ".env"

#: Files that mark the root of a project when walking up the tree.
PROJECT_MARKERS: tuple[str, ...] = (".env", "requirements.txt", "pyproject.toml", ".git")

#: Environment variables checked, in priority order, for each setting.
_API_KEY_VARS = ("OPENAI_API_KEY", "AI_API_KEY", "DEEPSEEK_API_KEY")
_BASE_URL_VARS = ("AI_BASE_URL", "OPENAI_BASE_URL", "OPENAI_API_BASE")
_MODEL_VARS = ("AI_MODEL_NAME", "AI_MODEL", "OPENAI_MODEL", "MODEL_NAME")

_FENCE_RE = re.compile(r"```[ \t]*(?:json|json5)?[ \t]*\r?\n(.*?)```", re.DOTALL | re.IGNORECASE)

_SIMPLE_FENCE_RE = re.compile(r"```[ \t]*(?:json|json5)?[ \t]*(.*?)```", re.DOTALL | re.IGNORECASE)

_OPENING_TO_CLOSING = {"{": "}", "[": "]"}

_STATUS_MARKERS = {
    "pending": " ",
    "not_started": " ",
    "todo": " ",
    "in_progress": "~",
    "in-progress": "~",
    "active": "~",
    "doing": "~",
    "completed": "x",
    "complete": "x",
    "done": "x",
}

_CONTENT_KEYS = ("content", "text", "task", "title", "description", "name")


class ConfigError(RuntimeError):
    """Raised when the LLM configuration cannot be resolved from the environment."""


class JsonExtractionError(ValueError):
    """Raised when no valid JSON value can be found inside a text blob."""


@dataclass(frozen=True, repr=False)
class LLMConfig:
    """Resolved connection settings for an OpenAI-compatible endpoint."""

    api_key: str = ""
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL_NAME

    @property
    def masked_api_key(self) -> str:
        """Return the API key with everything but the last four characters hidden."""
        if not self.api_key:
            return "<unset>"
        if len(self.api_key) <= 4:
            return "*" * len(self.api_key)
        return f"***{self.api_key[-4:]}"

    def __repr__(self) -> str:  # pragma: no cover - trivial formatting
        return (
            f"LLMConfig(api_key={self.masked_api_key!r}, "
            f"base_url={self.base_url!r}, model={self.model!r})"
        )

    def as_dict(self, *, include_secret: bool = False) -> dict[str, str]:
        """Return the configuration as a plain dict (secrets masked by default)."""
        return {
            "api_key": self.api_key if include_secret else self.masked_api_key,
            "base_url": self.base_url,
            "model": self.model,
        }


# --------------------------------------------------------------------------- #
# Environment / configuration helpers
# --------------------------------------------------------------------------- #
def find_project_root(start: str | os.PathLike[str] | None = None) -> Path:
    """Walk up from *start* until a directory containing a project marker is found.

    Falls back to the starting directory when no marker exists anywhere above.
    """
    current = Path(start).expanduser().resolve() if start is not None else Path(__file__).resolve().parent
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        if any((candidate / marker).exists() for marker in PROJECT_MARKERS):
            return candidate
    return current


def load_env(
    env_file: str | os.PathLike[str] | None = None,
    *,
    override: bool = False,
) -> Path | None:
    """Load a ``.env`` file into :data:`os.environ`.

    Returns the path that was loaded, or ``None`` when there is no such file
    (so callers can run with plain environment variables).
    """
    path = Path(env_file).expanduser() if env_file is not None else find_project_root() / ENV_FILE_NAME
    if not path.is_file():
        return None
    try:
        from dotenv import load_dotenv
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ConfigError("python-dotenv is required to load a .env file") from exc

    load_dotenv(dotenv_path=path, override=override)
    return path


def get_llm_config(
    env: Mapping[str, str] | None = None,
    *,
    env_file: str | os.PathLike[str] | None = None,
    load_env_file: bool = True,
    require_api_key: bool = True,
) -> LLMConfig:
    """Resolve :class:`LLMConfig` from *env* or from the process environment.

    Passing *env* explicitly makes the function pure and easy to test; otherwise
    ``.env`` is loaded first (unless *load_env_file* is ``False``).
    """
    if env is not None:
        source: Mapping[str, str] = env
    else:
        if load_env_file:
            load_env(env_file)
        source = os.environ

    api_key = _first_value(source, _API_KEY_VARS, default="")
    base_url = _first_value(source, _BASE_URL_VARS, default=DEFAULT_BASE_URL)
    model = _first_value(source, _MODEL_VARS, default=DEFAULT_MODEL_NAME)

    if require_api_key and not api_key:
        looked_for = ", ".join(_API_KEY_VARS)
        raise ConfigError(f"no API key found; set one of: {looked_for}")

    return LLMConfig(api_key=api_key, base_url=base_url.rstrip("/"), model=model)


def create_client(
    config: LLMConfig | None = None,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    **client_kwargs: Any,
) -> Any:
    """Build an ``openai.OpenAI`` client from *config* (or the environment).

    ``openai`` is imported lazily so that importing this package never requires
    the dependency to be installed.
    """
    resolved = config if config is not None else get_llm_config()
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ConfigError("the 'openai' package is required to create a client") from exc

    return OpenAI(
        api_key=api_key or resolved.api_key,
        base_url=base_url or resolved.base_url,
        **client_kwargs,
    )


def _first_value(source: Mapping[str, str], names: Sequence[str], *, default: str) -> str:
    """Return the first non-empty value among *names*, else *default*."""
    for name in names:
        value = source.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


# --------------------------------------------------------------------------- #
# Text / JSON helpers
# --------------------------------------------------------------------------- #
def extract_json(text: str) -> Any:
    """Extract the first JSON value from a chunk of LLM output.

    Handles the three shapes seen in practice: bare JSON, JSON inside a fenced
    code block, and JSON embedded in prose. Raises :class:`JsonExtractionError`
    when nothing parses.
    """
    if not isinstance(text, str):
        raise TypeError(f"extract_json() expects str, got {type(text).__name__}")

    for candidate in _json_candidates(text):
        try:
            return json.loads(candidate)
        except ValueError:
            continue

    preview = normalize_whitespace(text)[:120]
    raise JsonExtractionError(f"no valid JSON value found in: {preview!r}")


def normalize_whitespace(text: str) -> str:
    """Collapse all runs of whitespace (including newlines) into single spaces."""
    return " ".join(text.split())


def truncate(text: str, max_chars: int, *, suffix: str = "...") -> str:
    """Shorten *text* to at most *max_chars* characters, appending *suffix*.

    The result never exceeds *max_chars*: when the budget is smaller than the
    suffix itself, the suffix is trimmed to fit.
    """
    if max_chars < 0:
        raise ValueError("max_chars must be non-negative")
    if len(text) <= max_chars:
        return text
    if max_chars <= len(suffix):
        return suffix[:max_chars]
    return text[: max_chars - len(suffix)] + suffix


def unique_preserving_order(items: Iterable[Any]) -> list[Any]:
    """Deduplicate *items* while keeping their original order (hashable items)."""
    seen: set[Any] = set()
    result: list[Any] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def render_task_list(tasks: Iterable[Any]) -> str:
    """Render planning tasks as a numbered markdown checklist.

    Each task may be a mapping (``{"content": ..., "status": ...}``) or any
    object exposing those attributes. Status is mapped to ``[ ]`` / ``[~]`` /
    ``[x]``; unknown statuses fall back to ``[ ]``.
    """
    lines: list[str] = []
    for index, task in enumerate(tasks, start=1):
        content = normalize_whitespace(str(_task_content(task)))
        status = normalize_whitespace(str(_field(task, ("status",), default="pending"))).lower()
        status = status.replace("-", "_").replace(" ", "_")
        marker = _STATUS_MARKERS.get(status, " ")
        lines.append(f"{index}. [{marker}] {content}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _json_candidates(text: str) -> Iterator[str]:
    """Yield trimmed candidate substrings that might contain JSON."""
    stripped = text.strip()
    if stripped:
        yield stripped

    for match in _FENCE_RE.finditer(text):
        fenced = match.group(1).strip()
        if fenced:
            yield fenced

    for match in _SIMPLE_FENCE_RE.finditer(text):
        fenced = match.group(1).strip()
        if fenced:
            yield fenced

    yield from _balanced_candidates(text)
    yield from _balanced_candidates(text, limit_per_char=True)


def _balanced_candidates(text: str, *, limit: int = 32, limit_per_char: bool = False) -> Iterator[str]:
    """Yield balanced ``{...}`` / ``[...]`` substrings found in *text*.

    With ``limit_per_char=True`` the scan restarts *inside* the previously found
    block, which recovers nested payloads such as ``{"plan": {...}}``.
    """
    count = 0
    index = 0
    length = len(text)
    while index < length:
        if text[index] not in _OPENING_TO_CLOSING:
            index += 1
            continue
        end = _find_matching_bracket(text, index)
        if end is None:
            index += 1
            continue
        yield text[index : end + 1]
        count += 1
        if count >= limit:
            return
        index = index + 1 if limit_per_char else end + 1


def _find_matching_bracket(text: str, start: int) -> int | None:
    """Return the index closing the bracket at *start*, ignoring string contents."""
    opening = text[start]
    closing = _OPENING_TO_CLOSING[opening]
    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return index
    return None


def _field(task: Any, names: Sequence[str], *, default: Any = None) -> Any:
    """Read *names* from a mapping or an object, returning *default* if absent."""
    if isinstance(task, Mapping):
        for name in names:
            if name in task and task[name] is not None:
                return task[name]
        return default
    for name in names:
        value = getattr(task, name, None)
        if value is not None:
            return value
    return default


def _task_content(task: Any) -> Any:
    """Best-effort extraction of a task's human readable text."""
    content = _field(task, _CONTENT_KEYS)
    return content if content is not None else task
