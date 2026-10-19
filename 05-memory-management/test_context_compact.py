"""ContextCompact 公有方法的单元测试（覆盖 tool_result_budget、snip_compact 与 micro_compact）。

运行方式（在 05-memory-management 目录下）：
    python -m unittest test_context_compact -v
"""

import json
import re
import sys
import tempfile
from typing import cast
import unittest
from pathlib import Path
from openai.types.chat import ChatCompletionMessageParam

# 允许从任意目录运行：把测试文件所在目录加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# 包导入无副作用（不创建 OpenAI 客户端、不读环境变量），可直接导入
from context_compact.core import ContextCompact, _get_field, _set_field


def make_tool_msg(tool_call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


def make_assistant_tool_calls(call_ids: list[str]) -> dict:
    return {
        "role": "assistant",
        "content": "using tools",
        "tool_calls": [
            {"id": cid, "type": "function",
             "function": {"name": "bash", "arguments": "{}"}}
            for cid in call_ids
        ],
    }


def plain_msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


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
        self.assertIn("Full output: ", new_content)
        self.assertIn("tool_call_big.txt", new_content)  # 占位符引用完整输出路径
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


class SnipCompactTest(unittest.TestCase):
    """snip_compact：头尾保留 + 中间剪断归档（L1）。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        tmp = Path(self.tmp.name)
        self.transcript_dir = tmp / "transcripts"
        self.compact = ContextCompact(
            llm_client=None,
            model="test-model",
            transcript_dir=self.transcript_dir,
            tool_results_dir=tmp / "tool_results",
        )

    def transcript_files(self):
        if not self.transcript_dir.exists():
            return []
        return sorted(p.name for p in self.transcript_dir.glob("*.jsonl"))

    # ---------- 不触发压缩 ----------

    def test_short_list_passthrough(self):
        messages = [plain_msg("user", f"m{i}") for i in range(5)]
        result: list[ChatCompletionMessageParam] = self.compact.snip_compact(
            cast(list[ChatCompletionMessageParam], messages), max_messages=50
        )
        self.assertIs(result, messages)
        self.assertEqual(self.transcript_files(), [])

    # ---------- 基本压缩 ----------

    def test_compacts_middle_and_keeps_head_tail(self):
        messages = [plain_msg("user", f"m{i}") for i in range(20)]
        result: list[ChatCompletionMessageParam] = self.compact.snip_compact(
            cast(list[ChatCompletionMessageParam], messages), max_messages=10
        )

        self.assertIsNot(result, messages)           # 返回新列表
        self.assertIs(result[0], messages[0])        # 消息对象被复用
        self.assertEqual(len(result), 10)            # 3 head + 1 marker + 6 tail

        marker = result[3]
        self.assertEqual(marker["role"], "user")
        match = re.fullmatch(r"\[(\d+) messages archived at (.+)\]", _get_field(marker, "content"))
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.group(1), "11")       # 剪掉中间 11 条
        archived = Path(match.group(2))
        self.assertTrue(archived.is_file())
        self.assertTrue(archived.resolve().is_relative_to(self.transcript_dir.resolve()))

        self.assertIs(result[4], messages[14])       # 尾部从 m14 开始
        self.assertEqual(result[-1], messages[19])

        # transcript 落盘：当前实现归档全量 messages（含头尾，共 20 行）
        files = self.transcript_files()
        self.assertEqual(len(files), 1)
        lines = (self.transcript_dir / files[0]).read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 20)

    def test_default_max_messages_is_50(self):
        messages = [plain_msg("user", f"m{i}") for i in range(60)]
        result: list[ChatCompletionMessageParam] = self.compact.snip_compact(
            cast(list[ChatCompletionMessageParam], messages)
        )
        self.assertEqual(len(result), 50)
        self.assertIs(result[4], messages[14])
        match = re.fullmatch(r"\[(\d+) messages archived at (.+)\]", _get_field(result[3], "content"))
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.group(1), "11")

    def test_boundary_max_messages_5(self):
        messages = [plain_msg("user", f"m{i}") for i in range(6)]
        result: list[ChatCompletionMessageParam] = self.compact.snip_compact(
            cast(list[ChatCompletionMessageParam], messages), max_messages=5
        )
        self.assertEqual(len(result), 5)             # 3 head + 1 marker + 1 tail
        match = re.fullmatch(r"\[(\d+) messages archived at (.+)\]", _get_field(result[3], "content"))
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.group(1), "2")
        self.assertIs(result[4], messages[5])

    # ---------- 工具调用对保持 ----------

    def test_head_tool_pair_fully_kept(self):
        messages = (
            [plain_msg("system", "s"), plain_msg("user", "u"),
             make_assistant_tool_calls(["call_h1", "call_h2", "call_h3"])]
            + [make_tool_msg(f"call_h{i}", f"out{i}") for i in (1, 2, 3)]
            + [plain_msg("user", f"m{i}") for i in range(14)]
        )
        result: list[ChatCompletionMessageParam] = self.compact.snip_compact(
            cast(list[ChatCompletionMessageParam], messages), max_messages=10
        )

        # head 扩展保留完整工具对 → 保留区超出 max_messages（设计取舍）
        self.assertEqual(len(result), 13)            # 6 head + 1 marker + 6 tail
        self.assertIs(result[2], messages[2])
        self.assertEqual(_get_field(result[2], "tool_calls")[0]["id"], "call_h1")
        self.assertEqual(_get_field(result[2], "tool_calls")[1]["id"], "call_h2")
        self.assertEqual(_get_field(result[2], "tool_calls")[2]["id"], "call_h3")
        self.assertEqual(_get_field(result[3], "tool_call_id"), "call_h1")
        self.assertEqual(_get_field(result[4], "tool_call_id"), "call_h2")
        self.assertEqual(_get_field(result[5], "tool_call_id"), "call_h3")
        
        match = re.fullmatch(r"\[(\d+) messages archived at (.+)\]", _get_field(result[6], "content"))
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.group(1), "8")
        self.assertIs(result[7], messages[14])

    def test_tail_tool_pair_fully_kept(self):
        messages = (
            [plain_msg("user", f"m{i}") for i in range(13)]
            + [make_assistant_tool_calls(["call_t1", "call_t2"])]
            + [make_tool_msg("call_t1", "out1"), make_tool_msg("call_t2", "out2")]
            + [plain_msg("user", f"e{i}") for i in range(4)]
        )
        result: list[ChatCompletionMessageParam] = self.compact.snip_compact(
            cast(list[ChatCompletionMessageParam], messages), max_messages=10
        )

        # tail_start 从 14 回退到 13，assistant(tool_calls) 一并留在尾部
        self.assertEqual(len(result), 11)            # 3 head + 1 marker + 7 tail
        self.assertEqual(_get_field(result[4], "role"), "assistant")
        self.assertIsNotNone(_get_field(result[4], "tool_calls"))
        self.assertEqual(_get_field(result[4], "tool_calls")[0]["id"], "call_t1")
        self.assertEqual(_get_field(result[4], "tool_calls")[1]["id"], "call_t2")
        result: list[ChatCompletionMessageParam] = self.compact.snip_compact(
            cast(list[ChatCompletionMessageParam], messages), max_messages=10
        )
        self.assertEqual(_get_field(result[3], "role"), "user")
        match = re.fullmatch(r"\[(\d+) messages archived at (.+)\]", _get_field(result[3], "content"))
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.group(1), "10")

    # # ---------- 幂等：单条 archive marker 不重复归档 ----------

    def test_single_archive_marker_not_rearchived(self):
        # 构造"上次压缩结果 + 新消息"形态，使 middle 恰好只剩旧 marker
        real_path = self.compact._write_transcript([plain_msg("user", "old")])
        messages = (
            [plain_msg("system", "s"), plain_msg("user", "u"),
             make_assistant_tool_calls(["call_h1", "call_h2", "call_h3"])]
            + [make_tool_msg(f"call_h{i}", f"out{i}") for i in (1, 2, 3)]
            + [{"role": "user", "content": f"[1 messages archived at {real_path}]"}]
            + [plain_msg("user", f"m{i}") for i in range(6)]
        )
        result: list[ChatCompletionMessageParam] = self.compact.snip_compact(
            cast(list[ChatCompletionMessageParam], messages), max_messages=10
        )

        self.assertIs(result, messages)                     # 原样返回，不再归档
        self.assertEqual(len(self.transcript_files()), 1)   # 未新增 transcript

    def test_marker_outside_transcript_dir_is_rearchived(self):
        # marker 指向 transcript_dir 之外的文件 → 视为伪造，重新归档
        fake = Path(self.tmp.name) / "outside.json"
        fake.write_text("not a transcript", encoding="utf-8")
        messages = (
            [plain_msg("system", "s"), plain_msg("user", "u"),
             make_assistant_tool_calls(["call_h1", "call_h2", "call_h3"])]
            + [make_tool_msg(f"call_h{i}", f"out{i}") for i in (1, 2, 3)]
            + [{"role": "user", "content": f"[1 messages archived at {fake}]"}]
            + [plain_msg("user", f"m{i}") for i in range(6)]
        )
        result: list[ChatCompletionMessageParam] = self.compact.snip_compact(
            cast(list[ChatCompletionMessageParam], messages), max_messages=10
        )

        self.assertIsNot(result, messages)          # 伪造 marker 未被认可 → 重新归档
        match = re.fullmatch(r"\[(\d+) messages archived at (.+)\]", _get_field(result[6], "content"))
        self.assertIsNotNone(match)
        assert match is not None
        self.assertTrue(Path(match.group(2)).resolve()
                        .is_relative_to(self.transcript_dir.resolve()))
        self.assertEqual(len(self.transcript_files()), 1)


class MicroCompactTest(unittest.TestCase):
    """micro_compact：持久化过旧的工具调用结果，上下文只留指针（L2）。"""

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
        # 缩小保留窗口，便于用小数据测试
        _set_field(self.compact, "KEEP_RECENT_MESSAGES", 1)

    def persisted_files(self):
        if not self.tool_results_dir.exists():
            return []
        return sorted(p.name for p in self.tool_results_dir.glob("tool_*.txt"))

    @staticmethod
    def tool_round(call_id: str, size: int = 200) -> list:
        """一轮完整工具调用：assistant(tool_calls) + tool 结果。"""
        return [
            make_assistant_tool_calls([call_id]),
            make_tool_msg(call_id, "x" * size),
        ]

    # ---------- 不触发持久化的场景 ----------

    def test_under_keep_window_noop(self):
        _set_field(self.compact, "KEEP_RECENT_MESSAGES", 5)
        messages = (
            self.tool_round("c1") + self.tool_round("c2")
            + [plain_msg("assistant", "done")]
        )
        result = self.compact.micro_compact(messages)
        self.assertIs(result, messages)
        self.assertEqual(messages[1]["content"], "x" * 200)
        self.assertEqual(messages[3]["content"], "x" * 200)
        self.assertEqual(_get_field(messages[1], "tool_call_id"), "c1")
        self.assertEqual(self.persisted_files(), [])

    def test_target_chars_already_under_noop(self):
        # 当前估算已低于阈值 → 立即停止，什么都不转存
        messages = (
            self.tool_round("c1") + self.tool_round("c2")
            + [plain_msg("assistant", "done")]
        )
        self.compact.micro_compact(messages, target_chars=10**9)
        self.assertEqual(self.persisted_files(), [])
        self.assertEqual(messages[1]["content"], "x" * 200)

    def test_small_results_skipped(self):
        # 结果内容 <= 120 字符 → 即使在持久化范围内也跳过
        messages = (
            [make_assistant_tool_calls(["c0"]), make_tool_msg("c0", "short output")]
            + self.tool_round("c1") + self.tool_round("c2")
            + [plain_msg("assistant", "done")]
        )
        self.compact.micro_compact(messages)   # consumed=[1,3,5]，[:-1]=[1,3]
        self.assertEqual(messages[1]["content"], "short output")   # 过短跳过
        self.assertEqual(self.persisted_files(), ["tool_c1.txt"])  # 只有 c1 被转存

    # ---------- 触发持久化的场景 ----------

    def test_old_results_persisted_to_files(self):
        messages = (
            self.tool_round("c1") + self.tool_round("c2") + self.tool_round("c3")
            + [plain_msg("assistant", "done")]
        )
        result = self.compact.micro_compact(messages)   # KEEP=1 → 只保留 c3

        self.assertIs(result, messages)
        self.assertEqual(self.persisted_files(), ["tool_c1.txt", "tool_c2.txt"])
        self.assertEqual(
            (self.tool_results_dir / "tool_c1.txt").read_text(encoding="utf-8"),
            "x" * 200,
        )
        self.assertEqual(
            (self.tool_results_dir / "tool_c2.txt").read_text(encoding="utf-8"),
            "x" * 200,
        )
        self.assertEqual(messages[5]["content"], "x" * 200)   # 窗口内 c3 原样

    def test_current_turn_tool_results_untouched(self):
        # 位于最后一条 assistant 之后的工具结果属于当前轮次，永不持久化
        messages = (
            self.tool_round("c1") + self.tool_round("c2") + self.tool_round("c3")
            + [make_assistant_tool_calls(["c4"]), make_tool_msg("c4", "x" * 200)]
        )
        self.compact.micro_compact(messages)
        self.assertEqual(self.persisted_files(), ["tool_c1.txt", "tool_c2.txt"])
        self.assertEqual(messages[5]["content"], "x" * 200)   # 窗口内保留
        self.assertEqual(messages[7]["content"], "x" * 200)   # 当前轮次不处理

    def test_target_chars_stops_early(self):
        # 回归：转存后 content 变为单行指针，估算应下降，达标即停
        # （此前 content 膨胀导致估算不降反升，多余的 c2 也被转存）
        messages = (
            self.tool_round("c1") + self.tool_round("c2") + self.tool_round("c3")
            + [plain_msg("assistant", "done")]
        )
        estimate = len(json.dumps(messages, default=str, ensure_ascii=False))
        # 阈值略低于当前估算：转存 1 条后即达标 → 只处理最旧的 c1
        self.compact.micro_compact(messages, target_chars=estimate - 10)
        self.assertEqual(self.persisted_files(), ["tool_c1.txt"])
        self.assertEqual(messages[3]["content"], "x" * 200)   # c2 未处理
        self.assertEqual(messages[5]["content"], "x" * 200)   # c3 在窗口内

    def test_already_saved_marker_not_repersisted(self):
        # 已是指针格式且指向真实文件 → 不应重复落盘
        self.tool_results_dir.mkdir(parents=True)
        real = self.tool_results_dir / "tool_c0.txt"
        real.write_text("saved data", encoding="utf-8")
        messages = (
            [make_assistant_tool_calls(["c0"]),
             make_tool_msg("c0", f"[Earlier tool result saved at {real}]")]
            + self.tool_round("c1")
            + [plain_msg("assistant", "done")]
        )
        self.compact.micro_compact(messages)   # consumed=[1,3]，[:-1]=[1]
        self.assertEqual(self.persisted_files(), ["tool_c0.txt"])   # 未新增文件
        self.assertEqual(real.read_text(encoding="utf-8"), "saved data")
        self.assertEqual(_get_field(messages[1], "content"),
                         f"[Earlier tool result saved at {real}]")

    def test_persisted_output_path_parses_relative_name(self):
        # 回归：占位符中的相对文件名应按 tool_results_dir 解析（修复恒返回 None）
        self.tool_results_dir.mkdir(parents=True)
        real = self.tool_results_dir / "tool_c0.txt"
        real.write_text("data", encoding="utf-8")
        wrapper = (f"<persisted-output>\nFull output: {real.name}\n"
                   f"Preview:\ndata\n</persisted-output>")
        self.assertEqual(self.compact._persisted_output_path(wrapper), str(real))
        # 指向 tool_results_dir 之外的绝对路径仍被拒绝
        outside = Path(self.tmp.name) / "outside.txt"
        outside.write_text("evil", encoding="utf-8")
        self.assertIsNone(
            self.compact._persisted_output_path(f"[Earlier tool result saved at {outside}]")
        )

    def test_persisted_wrapper_round_trip(self):
        # <persisted-output> 占位符应能解析回真实路径，规范化为单行指针且不重复落盘
        self.tool_results_dir.mkdir(parents=True)
        real = self.tool_results_dir / "tool_c0.txt"
        real.write_text("saved data", encoding="utf-8")
        wrapper = (f"<persisted-output>\nFull output: {real.name}\n"
                   f"Preview:\n{'x' * 200}\n</persisted-output>")
        messages = (
            [make_assistant_tool_calls(["c0"]), make_tool_msg("c0", wrapper)]
            + self.tool_round("c1")
            + [plain_msg("assistant", "done")]
        )
        self.compact.micro_compact(messages)   # consumed=[1,3]，[:-1]=[1]
        self.assertEqual(self.persisted_files(), ["tool_c0.txt"])   # 未新增文件
        self.assertEqual(_get_field(messages[1], "content"),
                         f"[Earlier tool result saved at {real}]")

    # ---------- 回归：已修复的缺陷 ----------

    def test_pointer_format_and_tool_call_id_preserved(self):
        # 修复回归：指针为单行格式且指向真实文件，tool_call_id 保持原值
        # （此前 _persist_tool_result 返回包装文本导致 id 被覆写、content 混杂）
        messages = (
            self.tool_round("c1") + self.tool_round("c2")
            + [plain_msg("assistant", "done")]
        )
        self.compact.micro_compact(messages)   # KEEP=1 → c1 被转存

        match = re.fullmatch(r"\[Earlier tool result saved at (.+)\]",
                             _get_field(messages[1], "content"))
        self.assertIsNotNone(match)   # 当前 content 混入多行 <persisted-output> 文本
        assert match is not None
        pointed = Path(match.group(1))
        self.assertTrue(pointed.is_file())
        self.assertTrue(pointed.resolve().is_relative_to(self.tool_results_dir.resolve()))
        self.assertEqual(_get_field(messages[1], "tool_call_id"), "c1")   # id 不应被改写

    def test_no_assistant_message_does_not_crash(self):
        # 修复回归：无 assistant 消息时原样返回，不再抛 StopIteration
        messages = [plain_msg("user", "hi"), plain_msg("user", "again")]
        result = self.compact.micro_compact(messages)
        self.assertEqual(result, messages)


if __name__ == "__main__":
    unittest.main()
