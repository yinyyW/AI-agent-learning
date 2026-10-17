"""上下文压缩（Context Compact）：archive, reduce, then summarize。

当前实现 L3 策略：将过长的工具调用结果持久化到文件，控制上下文预算。
"""

import re
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any

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


class ContextCompact:
    TOOL_RESULT_MAX_CHARS = 200000
    LARGE_RESULT_CHAR_LIMIT = 30000

    def __init__(self, llm_client, model: str, transcript_dir: Path, tool_results_dir: Path):
        self.client = llm_client
        self.model = model
        self.transcript_dir = transcript_dir
        self.tool_results_dir = tool_results_dir

    def _persist_tool_result(self, tool_call_id: str, content: str):
        # 持久化工具调用结果到文件
        self.tool_results_dir.mkdir(parents=True, exist_ok=True)
        safe_id = re.sub(r"[^A-Za-z0-9._-]", "_", str(tool_call_id))[:120] or "unknown"
        file_path = self.tool_results_dir / f"tool_{safe_id}.txt"
        file_path.write_text(content, encoding="utf-8")

        preview = content[:1000]
        return (f"<persisted-output>\nFull output: {file_path.name}\n"
                f"Preview:\n{preview}\n</persisted-output>")

    # L3: 压缩工具调用结果，将过长的输出持久化到文件
    def tool_result_budget(self, messages: list[ChatCompletionMessageParam], max_chars: int | None = None) -> list[ChatCompletionMessageParam]:
        budget = max_chars if max_chars is not None else self.TOOL_RESULT_MAX_CHARS
        # 定位尾部连续的最后一轮 tool 消息
        end = len(messages)
        start = end
        while start > 0 and _get_field(messages[start - 1], "role") == "tool":
            start -= 1
        if start == end:
            return messages

        # 计算工具调用输出长度
        tool_call_messages = messages[start:end]
        total_len = sum(len(_get_field(msg, "content")) for msg in tool_call_messages)
        if total_len <= budget:
            return messages

        # 工具调用输出结果大小排序
        ranked_messages = sorted(tool_call_messages, key=lambda msg: len(_get_field(msg, "content")), reverse=True)

        # 保存过长的工具调用结果到文件
        for _, msg in enumerate(ranked_messages):
            if total_len <= budget:
                break
            if not _get_field(msg, "content") or len(_get_field(msg, "content")) < self.LARGE_RESULT_CHAR_LIMIT:
                continue
            new_content = self._persist_tool_result(_get_field(msg, "tool_call_id"), _get_field(msg, "content"))
            _set_field(msg, "content", new_content)
            total_len = sum(len(_get_field(msg, "content")) for msg in tool_call_messages)
        return messages
