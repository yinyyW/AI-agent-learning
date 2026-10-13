"""Helper package for the 04-planning-coordination module.

The module explores how an agent plans and coordinates work: keeping a task
list in sync with reality, deciding the next step, and delegating to sub-agents.
This package holds the pieces that every script in the module shares, so the
demos can stay focused on the planning logic itself.

Public API (re-exported from :mod:`planning_coordination.utils`)::

    from planning_coordination import (
        create_client,      # build an OpenAI-compatible client from .env
        extract_json,       # pull JSON out of an LLM reply
        get_llm_config,     # resolved api_key / base_url / model
        render_task_list,   # task list -> markdown checklist
    )
"""

from __future__ import annotations

from .utils import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL_NAME,
    ENV_FILE_NAME,
    PROJECT_MARKERS,
    ConfigError,
    JsonExtractionError,
    LLMConfig,
    create_client,
    extract_json,
    find_project_root,
    get_llm_config,
    load_env,
    normalize_whitespace,
    render_task_list,
    truncate,
    unique_preserving_order,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "ConfigError",
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL_NAME",
    "ENV_FILE_NAME",
    "JsonExtractionError",
    "LLMConfig",
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
