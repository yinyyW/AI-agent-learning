"""ContextCompact 公有方法的单元测试（当前仅覆盖 tool_result_budget）。

运行方式（在 05-memory-management 目录下）：
    python -m unittest test_context_compact -v
"""

import sys
import tempfile
import unittest
from pathlib import Path
from openai.types.chat import ChatCompletionMessageParam

# 允许从任意目录运行：把测试文件所在目录加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# 包导入无副作用（不创建 OpenAI 客户端、不读环境变量），可直接导入
from context_compact.core import ContextCompact, _get_field, _set_field


def make_tool_msg(tool_call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


class ToolResultBudgetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        tmp = Path(self.tmp.name)
        self.tool_results_dir = tmp / "tool_results"
        self.compact = ContextCompact(
            llm_client=None,
            model="test-model",
            transcript_dir=tmp / "transcripts",
            tool_results_dir=self.tool_results_dir,
        )
        # 用实例属性遮蔽类属性，缩小阈值以便用小数据测试
        _set_field(self.compact, "LARGE_RESULT_CHAR_LIMIT", 100)
        _set_field(self.compact, "TOOL_RESULT_MAX_CHARS", 1000)

    def persisted_files(self):
        if not self.tool_results_dir.exists():
            return []
        return sorted(p.name for p in self.tool_results_dir.glob("tool_*.txt"))

    # ---------- 不触发压缩的场景 ----------

    def test_empty_messages(self):
        result = self.compact.tool_result_budget([], max_chars=10)
        self.assertEqual(result, [])

    def test_no_trailing_tool_messages(self):
        messages: list[ChatCompletionMessageParam] = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        result = self.compact.tool_result_budget(messages, max_chars=10)
        self.assertIs(result, messages)  # 原列表直接返回
        self.assertEqual(_get_field(messages[1], "content"), "hello")
        self.assertEqual(self.persisted_files(), [])

    def test_under_budget_returns_unchanged(self):
        messages = [
            {"role": "assistant", "content": "calling tools"},
            make_tool_msg("call_a", "a" * 50),
            make_tool_msg("call_b", "b" * 50),
        ]
        result = self.compact.tool_result_budget(messages, max_chars=1000)
        self.assertIs(result, messages)
        self.assertEqual(messages[1]["content"], "a" * 50)
        self.assertEqual(messages[2]["content"], "b" * 50)
        self.assertEqual(self.persisted_files(), [])

    def test_messages_below_large_limit_never_persisted(self):
        # 总量超预算，但每条都小于 LARGE_RESULT_CHAR_LIMIT → 不转存
        _set_field(self.compact, "LARGE_RESULT_CHAR_LIMIT", 1000)
        messages = [
            {"role": "assistant", "content": "calling tools"},
            make_tool_msg("call_a", "a" * 150),
            make_tool_msg("call_b", "b" * 150),
            make_tool_msg("call_c", "c" * 150),
        ]
        result = self.compact.tool_result_budget(messages, max_chars=100)

        self.assertIs(result, messages)
        self.assertEqual(messages[1]["content"], "a" * 150)
        self.assertEqual(messages[2]["content"], "b" * 150)
        self.assertEqual(messages[3]["content"], "c" * 150)
        self.assertEqual(self.persisted_files(), [])

    # # ---------- 触发压缩的场景 ----------

    def test_over_budget_persists_large_result(self):
        original = "x" * 2000
        messages = [{"role": "assistant", "content": ""}, make_tool_msg("call_big", original)]
        result = self.compact.tool_result_budget(messages)

        self.assertIs(result, messages)

        new_content = messages[1]["content"]
        # 内容被替换为 persisted 占位符
        self.assertTrue(new_content.startswith("<persisted-output>"))
        self.assertIn("Full output: tool_call_big.txt", new_content)
        self.assertIn(original[:1000], new_content)  # 保留前 1000 字符预览
        self.assertLessEqual(len(new_content), 2500)  # 总量回落到预算内
        # # 完整输出已落盘
        self.assertEqual(self.persisted_files(), ["tool_call_big.txt"])
        saved = (self.tool_results_dir / "tool_call_big.txt").read_text(encoding="utf-8")
        self.assertEqual(saved, original)

    def test_multiple_large_results_persisted_largest_first(self):
        big = "b" * 5000
        mid = "m" * 4000
        tiny = "t" * 50  # 小于 LARGE_RESULT_CHAR_LIMIT，永不转存
        messages = [
            {"role": "assistant", "content": "calling tools"},
            make_tool_msg("call_big", big),
            make_tool_msg("call_mid", mid),
            make_tool_msg("call_tiny", tiny),
        ]
        self.compact.tool_result_budget(messages, max_chars=2500)

        self.assertTrue(messages[1]["content"].startswith("<persisted-output>"))
        self.assertTrue(messages[2]["content"].startswith("<persisted-output>"))
        self.assertEqual(messages[3]["content"], tiny)  # 小结果保持原样
        self.assertEqual(self.persisted_files(), ["tool_call_big.txt", "tool_call_mid.txt"])
        self.assertEqual(
            (self.tool_results_dir / "tool_call_mid.txt").read_text(encoding="utf-8"), mid
        )

    def test_only_last_tool_round_is_considered(self):
        old = "o" * 5000  # 历史轮次的大结果，不在统计范围内
        new_big = "n" * 5000
        new_small = "s" * 50
        messages = [
            make_tool_msg("call_old", old),
            {"role": "assistant", "content": "intermediate reply"},
            make_tool_msg("call_new_big", new_big),
            make_tool_msg("call_new_small", new_small),
        ]
        self.compact.tool_result_budget(messages, max_chars=2500)

        # 历史大结果未被处理
        self.assertEqual(messages[0]["content"], old)
        # 只有最后一轮的大结果被转存（转存 new_big 后总量已回落，new_small 不动）
        self.assertTrue(messages[2]["content"].startswith("<persisted-output>"))
        self.assertEqual(messages[3]["content"], new_small)
        self.assertEqual(self.persisted_files(), ["tool_call_new_big.txt"])

    def test_default_budget_used_when_max_chars_is_none(self):
        # 不传 max_chars 时使用 TOOL_RESULT_MAX_CHARS
        _set_field(self.compact, "TOOL_RESULT_MAX_CHARS", 2500)
        original = "y" * 5000
        messages = [{"role": "assistant", "content": ""}, make_tool_msg("call_default", original)]
        self.compact.tool_result_budget(messages)

        self.assertTrue(messages[1]["content"].startswith("<persisted-output>"))
        self.assertEqual(self.persisted_files(), ["tool_call_default.txt"])

    def test_tool_call_id_sanitized_in_filename(self):
        # tool_call_id 含特殊字符时文件名会被清洗
        original = "z" * 5000
        messages = [{"role": "assistant", "content": ""}, make_tool_msg("call/../../etc", original)]
        self.compact.tool_result_budget(messages, max_chars=2500)

        files = self.persisted_files()
        self.assertEqual(len(files), 1)
        self.assertNotIn("/", files[0])
        saved = (self.tool_results_dir / files[0]).read_text(encoding="utf-8")
        self.assertEqual(saved, original)


if __name__ == "__main__":
    unittest.main()
