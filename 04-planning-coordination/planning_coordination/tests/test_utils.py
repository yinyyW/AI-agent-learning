"""Tests for :mod:`planning_coordination.utils`.

Run with either runner::

    python planning_coordination/tests/test_utils.py     # stdlib, no extra deps
    python -m pytest planning_coordination/tests -v      # if pytest is installed

The tests only cover behaviour that is free of network calls: the pure text
helpers, and the configuration helpers driven by an injected ``env`` mapping or
a temporary ``.env`` file.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Make the package importable when the tests are run from anywhere.
# tests/ -> planning_coordination/ -> 04-planning-coordination/ (package parent)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import planning_coordination  # noqa: E402
from planning_coordination import utils  # noqa: E402
from planning_coordination.utils import (  # noqa: E402
    DEFAULT_BASE_URL,
    DEFAULT_MODEL_NAME,
    ConfigError,
    JsonExtractionError,
    LLMConfig,
    extract_json,
    find_project_root,
    get_llm_config,
    load_env,
    normalize_whitespace,
    render_task_list,
    truncate,
    unique_preserving_order,
)


class Task:
    """Minimal attribute-based task, to prove non-mapping inputs work."""

    def __init__(self, content: str, status: str = "pending") -> None:
        self.content = content
        self.status = status


class ExtractJsonTests(unittest.TestCase):
    def test_bare_object(self) -> None:
        self.assertEqual(extract_json('{"step": "plan", "count": 2}'), {"step": "plan", "count": 2})

    def test_bare_array(self) -> None:
        self.assertEqual(extract_json('[{"id": 1}, {"id": 2}]'), [{"id": 1}, {"id": 2}])

    def test_scalar_values(self) -> None:
        self.assertEqual(extract_json("42"), 42)
        self.assertIs(extract_json("true"), True)
        self.assertIsNone(extract_json("null"))

    def test_fenced_json_block(self) -> None:
        text = 'Here you go:\n```json\n{"tasks": ["a", "b"]}\n```\nHope that helps!'
        self.assertEqual(extract_json(text), {"tasks": ["a", "b"]})

    def test_fenced_block_without_language_tag(self) -> None:
        self.assertEqual(extract_json('```\n{"ok": true}\n```'), {"ok": True})

    def test_json_embedded_in_prose(self) -> None:
        text = 'Sure. {"plan": {"step": "first"}, "confidence": 0.9} — anything else?'
        self.assertEqual(extract_json(text), {"plan": {"step": "first"}, "confidence": 0.9})

    def test_first_valid_value_wins(self) -> None:
        text = 'Ignore {this is not json} but keep {"keep": 1}'
        self.assertEqual(extract_json(text), {"keep": 1})

    def test_brackets_inside_strings_do_not_confuse_scanner(self) -> None:
        payload = {"note": "uses {braces} and [brackets]"}
        self.assertEqual(extract_json(f"prefix {json.dumps(payload)} suffix"), payload)

    def test_escaped_quote_inside_string(self) -> None:
        payload = {"quote": 'he said "hi"'}
        self.assertEqual(extract_json(json.dumps(payload)), payload)

    def test_nested_object(self) -> None:
        payload = {"outer": {"inner": {"value": 1}}}
        self.assertEqual(extract_json(json.dumps(payload)), payload)

    def test_missing_json_raises(self) -> None:
        with self.assertRaises(JsonExtractionError) as ctx:
            extract_json("I would rather describe the plan in plain English.")
        self.assertIn("no valid JSON value", str(ctx.exception))

    def test_unbalanced_json_raises(self) -> None:
        with self.assertRaises(JsonExtractionError):
            extract_json('{"unterminated": true')

    def test_non_string_input_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            extract_json({"already": "parsed"})  # type: ignore[arg-type]


class TextHelperTests(unittest.TestCase):
    def test_normalize_whitespace(self) -> None:
        self.assertEqual(normalize_whitespace("  a\n\tb \r\n c  "), "a b c")
        self.assertEqual(normalize_whitespace("\n\n"), "")

    def test_truncate_keeps_short_text(self) -> None:
        self.assertEqual(truncate("hello", 10), "hello")
        self.assertEqual(truncate("hello", 5), "hello")

    def test_truncate_appends_suffix(self) -> None:
        result = truncate("abcdefghij", 7)
        self.assertEqual(result, "abcd...")
        self.assertEqual(len(result), 7)

    def test_truncate_custom_suffix(self) -> None:
        result = truncate("abcdefghij", 6, suffix=" ->")
        self.assertEqual(result, "abc ->")
        self.assertEqual(len(result), 6)

    def test_truncate_budget_smaller_than_suffix(self) -> None:
        self.assertEqual(truncate("abcdefghij", 2), "..")
        self.assertEqual(truncate("abcdefghij", 0), "")

    def test_truncate_negative_budget_raises(self) -> None:
        with self.assertRaises(ValueError):
            truncate("abc", -1)

    def test_unique_preserving_order(self) -> None:
        self.assertEqual(unique_preserving_order(["a", "b", "a", "c", "b"]), ["a", "b", "c"])
        self.assertEqual(unique_preserving_order([]), [])


class RenderTaskListTests(unittest.TestCase):
    def test_status_markers(self) -> None:
        tasks = [
            {"content": "Scope the feature", "status": "completed"},
            {"content": "Write utils.py", "status": "in_progress"},
            {"content": "Write tests", "status": "pending"},
        ]
        self.assertEqual(
            render_task_list(tasks),
            "1. [x] Scope the feature\n2. [~] Write utils.py\n3. [ ] Write tests",
        )

    def test_missing_status_defaults_to_pending(self) -> None:
        self.assertEqual(render_task_list([{"content": "Read the docs"}]), "1. [ ] Read the docs")

    def test_unknown_status_falls_back_to_pending(self) -> None:
        self.assertEqual(
            render_task_list([{"content": "Mystery", "status": "sideways"}]),
            "1. [ ] Mystery",
        )

    def test_hyphenated_and_capitalised_statuses(self) -> None:
        tasks = [
            {"content": "A", "status": "In-Progress"},
            {"content": "B", "status": "done"},
        ]
        self.assertEqual(render_task_list(tasks), "1. [~] A\n2. [x] B")

    def test_alternative_content_keys(self) -> None:
        tasks = [{"text": "from text"}, {"title": "from title"}, {"description": "from description"}]
        self.assertEqual(
            render_task_list(tasks),
            "1. [ ] from text\n2. [ ] from title\n3. [ ] from description",
        )

    def test_attribute_based_tasks(self) -> None:
        tasks = [Task("object task", "completed"), Task("other task")]
        self.assertEqual(render_task_list(tasks), "1. [x] object task\n2. [ ] other task")

    def test_content_whitespace_is_collapsed(self) -> None:
        self.assertEqual(
            render_task_list([{"content": "  multi\n  line  "}]),
            "1. [ ] multi line",
        )

    def test_empty_input(self) -> None:
        self.assertEqual(render_task_list([]), "")


class LLMConfigTests(unittest.TestCase):
    def test_defaults(self) -> None:
        config = get_llm_config({}, require_api_key=False)
        self.assertEqual(config.base_url, DEFAULT_BASE_URL)
        self.assertEqual(config.model, DEFAULT_MODEL_NAME)
        self.assertEqual(config.api_key, "")

    def test_reads_this_projects_variable_names(self) -> None:
        config = get_llm_config(
            {
                "AI_API_KEY": "sk-ai",
                "AI_BASE_URL": "https://api.deepseek.com",
                "AI_MODEL_NAME": "deepseek-flash",
            }
        )
        self.assertEqual(config.api_key, "sk-ai")
        self.assertEqual(config.model, "deepseek-flash")

    def test_openai_variables_take_priority(self) -> None:
        config = get_llm_config({"OPENAI_API_KEY": "sk-openai", "AI_API_KEY": "sk-ai"})
        self.assertEqual(config.api_key, "sk-openai")

    def test_blank_values_are_skipped(self) -> None:
        config = get_llm_config({"AI_API_KEY": "   ", "OPENAI_API_KEY": "sk-real"})
        self.assertEqual(config.api_key, "sk-real")

    def test_trailing_slash_is_stripped_from_base_url(self) -> None:
        config = get_llm_config({"AI_API_KEY": "k", "AI_BASE_URL": "https://example.com/v1/"})
        self.assertEqual(config.base_url, "https://example.com/v1")

    def test_values_are_stripped(self) -> None:
        config = get_llm_config(
            {"AI_API_KEY": " svc-key ", "AI_BASE_URL": " https://x.test ", "AI_MODEL_NAME": " m "}
        )
        self.assertEqual((config.api_key, config.base_url, config.model), ("svc-key", "https://x.test", "m"))

    def test_missing_api_key_raises(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            get_llm_config({})
        self.assertIn("no API key found", str(ctx.exception))

    def test_repr_masks_api_key(self) -> None:
        config = LLMConfig(api_key="sk-abcdefgh1234")
        self.assertIn("***1234", repr(config))
        self.assertNotIn("sk-abcdefgh1234", repr(config))
        self.assertEqual(config.masked_api_key, "***1234")

    def test_masking_edge_cases(self) -> None:
        self.assertEqual(LLMConfig(api_key="").masked_api_key, "<unset>")
        self.assertEqual(LLMConfig(api_key="ab").masked_api_key, "**")

    def test_as_dict_masks_secret_by_default(self) -> None:
        config = LLMConfig(api_key="sk-secret9876")
        self.assertEqual(config.as_dict()["api_key"], "***9876")
        self.assertEqual(config.as_dict(include_secret=True)["api_key"], "sk-secret9876")

    def test_config_is_frozen(self) -> None:
        config = LLMConfig(api_key="k")
        with self.assertRaises(Exception):
            config.api_key = "other"  # type: ignore[misc]


class EnvironmentFileTests(unittest.TestCase):
    def test_find_project_root_locates_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "requirements.txt").write_text("openai\n", encoding="utf-8")
            nested = root / "a" / "b" / "c"
            nested.mkdir(parents=True)
            self.assertEqual(find_project_root(nested), root.resolve())

    def test_find_project_root_falls_back_to_start_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lonely = Path(tmp) / "nowhere"
            lonely.mkdir()
            self.assertEqual(find_project_root(lonely), lonely.resolve())

    def test_find_project_root_accepts_a_file_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pyproject.toml").write_text("", encoding="utf-8")
            script = root / "script.py"
            script.write_text("", encoding="utf-8")
            self.assertEqual(find_project_root(script), root.resolve())

    def test_load_env_reads_file_into_environment(self) -> None:
        variable = "PLANNING_COORDINATION_TEST_ENV"
        os.environ.pop(variable, None)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                env_file = Path(tmp) / ".env"
                env_file.write_text(f"{variable}=loaded-ok\n", encoding="utf-8")
                loaded = load_env(env_file)
                self.assertEqual(loaded, env_file)
                self.assertEqual(os.environ.get(variable), "loaded-ok")
        finally:
            os.environ.pop(variable, None)

    def test_load_env_returns_none_when_file_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(load_env(Path(tmp) / "does-not-exist.env"))


class PackageSurfaceTests(unittest.TestCase):
    def test_version_is_exposed(self) -> None:
        self.assertRegex(planning_coordination.__version__, r"^\d+\.\d+\.\d+$")

    def test_star_exports_all_exist(self) -> None:
        for name in planning_coordination.__all__:
            self.assertTrue(hasattr(planning_coordination, name), f"missing export: {name}")

    def test_utils_all_exports_exist(self) -> None:
        for name in utils.__all__:
            self.assertTrue(hasattr(utils, name), f"missing attribute: {name}")

    def test_re_exports_are_the_same_objects(self) -> None:
        self.assertIs(planning_coordination.extract_json, utils.extract_json)
        self.assertIs(planning_coordination.get_llm_config, utils.get_llm_config)


class CreateClientTests(unittest.TestCase):
    def test_client_uses_config_endpoint(self) -> None:
        try:
            from openai import OpenAI  # noqa: F401
        except ImportError:  # pragma: no cover - openai is in requirements.txt
            self.skipTest("openai is not installed")

        from planning_coordination import create_client

        config = LLMConfig(api_key="sk-test-only", base_url="https://example.test/v1", model="m")
        client = create_client(config)
        self.assertEqual(str(client.base_url).rstrip("/"), "https://example.test/v1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
