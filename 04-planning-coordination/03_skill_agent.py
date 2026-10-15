import ast
import json
import os, subprocess
from pathlib import Path
from typing import cast
from openai.types.chat import ChatCompletion, ChatCompletionFunctionToolParam, ChatCompletionMessageParam, ChatCompletionMessageFunctionToolCallParam
import yaml

from dotenv import load_dotenv
from openai import OpenAI

# ═══════════════════════════════════════════════════════════
#  INITIALIZE
# ═══════════════════════════════════════════════════════════
WORKDIR = Path.cwd()
SKILLS_DIR = WORKDIR / "skills"
CURRENT_TODOS: list[dict] = []
SKILL_REGISTRY: dict[str, dict] = {}
MAX_TURNS = 30

# Skill catalog scan (used by build_system below)
def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from SKILL.md. Returns (meta, body)."""
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    try:
        meta = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        meta = {}
    return meta, parts[2].strip()

def _scan_skills():
    """Scan skills/ dir, populate SKILL_REGISTRY with name/description/content."""
    if not SKILLS_DIR.exists():
        return
    for d in sorted(SKILLS_DIR.iterdir()):
        if not d.is_dir():
            continue
        manifest = d / "SKILL.md"
        if manifest.exists():
            raw = manifest.read_text()
            meta, body = _parse_frontmatter(raw)
            name = meta.get("name", d.name)
            desc = meta.get("description", raw.split("\n")[0].lstrip("#").strip())
            SKILL_REGISTRY[name] = {"name": name, "description": desc, "content": raw}

_scan_skills()

## system prompt
system_prompt = """
You are a ReAct-style AI Agent at {WORKDIR}. 

Available skills:
{skills}

Your task is to solve the user's request by following this loop:

Steps:
1. REASONING

   * Understand the user's request.
   * Determine what information or action is needed.
   * Decide whether you need to use a tool.

2. ACTION

   * If a tool is needed, call the appropriate tool.
   * Provide the correct arguments required by the tool.
   * Do not invent tool results.

3. OBSERVATION

   * After a tool is executed, carefully examine the returned result.
   * Use the observation as evidence for the next reasoning step.
   * If more information or another action is needed, continue the loop.

4. FINAL ANSWER

   * When you have enough information to answer the user's request, stop using tools.
   * Provide a concise and helpful final answer to the user.

Rules:
* Before starting any multi-step task, use todo_write to plan your steps.
* Always use the available tools when they are necessary to obtain accurate information.
* Never pretend that a tool has been executed when it has not.
* Never invent or assume tool results.
* Use the result of a tool call as the observation for the next step.
* You may call multiple tools when necessary.
* If no tool is required, answer the user directly.
* Once the task is complete, provide the final answer instead of making unnecessary tool calls.

The interaction follows this pattern:

REASONING
→ ACTION
→ OBSERVATION
→ REASONING
→ ACTION
→ OBSERVATION
→ ...
→ FINAL ANSWER

""".strip()

# s06: subagent gets its own system prompt — no task, no recursion
sub_system_prompt = """
    "You are a forked worker process at {WORKDIR}. You are NOT the main agent.  "
    Available skills:
    {skills}

    "Complete the task you were given, then return a concise summary. "
    "Do not delegate further."
"""

def list_skills() -> str:
    return "\n".join(f"- **{s['name']}**: {s['description']}" for s in SKILL_REGISTRY.values())

def build_system_prompt(is_subagent: bool = False) -> str:
    if is_subagent:
        return sub_system_prompt.format(WORKDIR=WORKDIR, skills=list_skills())
    return system_prompt.format(WORKDIR=WORKDIR, skills=list_skills())

# ═══════════════════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════════════════

from collections.abc import Mapping
from typing import Any
def _get_field(obj: Any, key: str, default: Any = None) -> Any:
    """兼容 dict 和 OpenAI SDK 对象的字段读取。"""
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)

def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path

def run_bash(command: str) -> str:
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"

def run_read(path, limit=None):
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"

def run_write(path: str, content: str) -> str:
    try:
        file_path = safe_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"

def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        file_path = safe_path(path)
        text = file_path.read_text()
        if old_text not in text:
            return f"Error: text not found in {path}"
        file_path.write_text(text.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"
    
def run_glob(pattern: str) -> str:
    import glob as g
    try:
        results = []
        for match in g.glob(pattern, root_dir=WORKDIR):
            if (WORKDIR / match).resolve().is_relative_to(WORKDIR):
                results.append(match)
        return "\n".join(results) if results else "(no matches)"
    except Exception as e:
        return f"Error: {e}"
    
#  todo_write tool — plan only, no execution
def _normalize_todos(todos: list | str | None) -> tuple[list[dict] | None, str | None]:
    if isinstance(todos, str):
        try:
            todos = json.loads(todos)
        except json.JSONDecodeError:
            try:
                todos = ast.literal_eval(todos)
            except (SyntaxError, ValueError):
                return None, "Error: todos must be a list or JSON array string"
    if not isinstance(todos, list):
        return None, "Error: todos must be a list"
    for i, t in enumerate(todos):
        if not isinstance(t, dict):
            return None, f"Error: todos[{i}] must be an object"
        if "content" not in t or "status" not in t:
            return None, f"Error: todos[{i}] missing 'content' or 'status'"
        if t["status"] not in ("pending", "in_progress", "completed"):
            return None, f"Error: todos[{i}] has invalid status '{t['status']}'"
    return todos, None

def run_todo_write(todos: list | str | None) -> str:
    global CURRENT_TODOS
    todos, error = _normalize_todos(todos)
    if not todos or error:
        return error or "Error: todos must be a list"
    CURRENT_TODOS = todos
    lines = ["\n\033[33m## Current Tasks\033[0m"]
    for t in CURRENT_TODOS:
        icon = {"pending": " ", "in_progress": "\033[36m▸\033[0m", "completed": "\033[32m✓\033[0m"}[t["status"]]
        lines.append(f"  [{icon}] {t['content']}")
    print("\n".join(lines))
    return f"Updated {len(CURRENT_TODOS)} tasks"

#  load_skill tool
def load_skill(name: str) -> str:
    skill = SKILL_REGISTRY.get(name)
    if not skill:
        return f"Skill not found: {name}"
    return skill["content"]

#  spawn_subagent tool
def spawn_subagent(description: str) -> str:
    print(f"\n\033[35m[Subagent spawned]\033[0m")
    messages: list[ChatCompletionMessageParam] = [{ "role": "system", "content": build_system_prompt(is_subagent=True) },
                                                  { "role": "user", "content": description }]  # fresh context
    for _ in range(0, MAX_TURNS):
        response = client.chat.completions.create(model=model_name, messages=messages, tools=TOOLS)
        assistant_message = response.choices[0].message
        messages.append(cast(ChatCompletionMessageParam, assistant_message))
        print(f"\n{'=' * 20} Subagent Loop {_} {'=' * 20}")
        # 1. 输出 AI 文本回复
        if assistant_message.content:
            print(f"[AI]\n{assistant_message.content}")

        # 2. 输出工具调用
        if assistant_message.tool_calls:
            for i, tool_call in enumerate(assistant_message.tool_calls, start=1):
                if tool_call.type != "function":
                    continue
                print(f"\n[Tool Call {i}]")
                print(f"  Name: {tool_call.function.name}")
                print(f"  Arguments: {tool_call.function.arguments}")

        # 3. 输出拒绝信息（如果有）
        if assistant_message.refusal:
            print(f"\n[Refusal]\n{assistant_message.refusal}")

        print(f"{'=' * 50}\n")
        if not assistant_message.tool_calls:
            force = trigger_hooks("Stop", messages)
            if force:
                messages.append({ "role": "assistant", "content": force })
                continue
            break
        for tc in assistant_message.tool_calls:
            if tc.type == "function":
                blocked = trigger_hooks("PreToolUse", tc)
                if blocked:
                    messages.append({
                                        "role": "tool",
                                        "tool_call_id": tc.id,
                                        "content": "Permission denied"
                                    })
                    continue
                tool_name = tc.function.name
                tool_args = json.loads(tc.function.arguments)
                tool = TOOL_HANDLERS[tool_name]
                output = tool(**tool_args) if tool else f"Unknown: {tool_name}"
                print(f"Tool {tool_name}({tool_args}) -> {str(output)[:200]}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(output)
                })
    # 4. 提取最终回复
    result = ""
    for m in reversed(messages):
        if _get_field(m, "role") == "assistant":
            result = _get_field(m, "content")
    result = result or "Subagent stopped without final answer."
    print(f"\033[35m[Subagent done]\033[0m")
    return result

# ═══════════════════════════════════════════════════════════
#  工具定义
# ═══════════════════════════════════════════════════════════
TOOLS: list[ChatCompletionFunctionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object", 
                "properties": {
                    "command": {"type": "string"},
                },
                "required": ["command"]
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read file contents.",
            "parameters": {
                "type": "object", 
                "properties": {
                    "path": {"type": "string"},
                    "limit": {"type": "integer"}
                },
                "required": ["path"]
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file.",
            "parameters": {
                "type": "object", 
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["path", "content"]
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace exact text in a file once.",
            "parameters": {
                "type": "object", 
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"}
                },
                "required": ["path", "old_text", "new_text"]
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find files matching a glob pattern.",
            "parameters": {
                "type": "object", 
                "properties": {
                    "pattern": {"type": "string"}
                },
                "required": ["pattern"]
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "todo_write",
            "description": "Create and manage a task list for your current session.",
            "parameters": {
                "type": "object", 
                "properties": {
                    "todos": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string"},
                                "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}
                            },
                            "required": ["content", "status"]
                        }
                    }
                },
                "required": ["todos"]
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task",
            "description": "Launch a subagent to handle a complex subtask. Returns only the final conclusion.",
            "parameters": {
                "type": "object", 
                "properties": {
                    "description": {"type": "string"}
                },
                "required": ["description"]
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_skill",
            "description": "Load the full content of a skill by name.",
            "parameters": {
                "type": "object", 
                "properties": {
                    "name": {"type": "string"}
                },
                "required": ["name"]
            }
        },
    }
]

SUB_TOOLS: list[ChatCompletionFunctionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object", 
                "properties": {
                    "command": {"type": "string"},
                },
                "required": ["command"]
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read file contents.",
            "parameters": {
                "type": "object", 
                "properties": {
                    "path": {"type": "string"},
                    "limit": {"type": "integer"}
                },
                "required": ["path"]
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file.",
            "parameters": {
                "type": "object", 
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["path", "content"]
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace exact text in a file once.",
            "parameters": {
                "type": "object", 
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"}
                },
                "required": ["path", "old_text", "new_text"]
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find files matching a glob pattern.",
            "parameters": {
                "type": "object", 
                "properties": {
                    "pattern": {"type": "string"}
                },
                "required": ["pattern"]
            }
        },
    },
]

# ═══════════════════════════════════════════════════════════
#  工具分发映射
# ═══════════════════════════════════════════════════════════
TOOL_HANDLERS = {
    "bash": run_bash, "read_file": run_read, "write_file": run_write,
    "edit_file": run_edit, "glob": run_glob, "todo_write": run_todo_write,
    "task": spawn_subagent, "load_skill": load_skill,
}

SUB_HANDLERS = {
    "bash": run_bash, "read_file": run_read, "write_file": run_write,
    "edit_file": run_edit, "glob": run_glob,  "load_skill": load_skill,
}

# ═══════════════════════════════════════════════════════════
#  定义Hooks
# ═══════════════════════════════════════════════════════════

# UserPromptSubmit: inject context
def context_inject_hook(query: str) -> str | None:
    """Inject current working directory info into every prompt."""
    print(f"\033[90m[HOOK] UserPromptSubmit: working in {WORKDIR}\033[0m")
    return None   # return None = no modification, let prompt through

# 明确禁止执行的命令
DENY_LIST = [
    # Linux / Unix
    "rm -rf /",
    "sudo",
    "shutdown",
    "reboot",
    "mkfs",
    "dd if=",

    # Windows CMD
    "format ",
    "diskpart",
    "bcdedit",
    "reg delete",
    "reg add",
    "shutdown /s",
    "shutdown /r",
    "shutdown /p",
    "taskkill /f",
    "del /f /s /q",
    "rmdir /s /q",

    # Windows PowerShell
    "remove-item",
    "clear-disk",
    "format-volume",
    "format-disk",
    "stop-computer",
    "restart-computer",
    "remove-partition",
    "remove-itemproperty",
    "disable-computerrestore",

    # 高风险文件及权限操作
    "cipher /w",
    "takeown /f",
    "icacls",
]

# 需要进一步检查上下文的危险操作
DESTRUCTIVE = [
    # Linux / Unix
    "rm ",
    "> /etc/",
    "chmod 777",
    "rmdir",

    # Windows CMD
    "del ",
    "erase ",
    "rd ",
    "rmdir ",
    "move ",
    "ren ",
    "copy ",
    "robocopy ",
    "xcopy ",

    # Windows PowerShell
    "remove-item ",
    "move-item ",
    "rename-item ",
    "clear-content ",
    "set-content ",
    "add-content ",
    "out-file ",
    "set-acl ",
    "remove-acl ",

    # 危险的重定向及磁盘操作
    "> nul",
    "> con",
    "format ",
    "diskpart",
    "reg delete",
]

def permission_hook(tool_call):
    if tool_call.type != "function":
        return True
    tool_args = json.loads(tool_call.function.arguments)
    if tool_call.function.name == "bash":
        for pattern in DENY_LIST:
            if pattern in tool_args.get("command", ""):
                print(f"\n\033[31m⛔ Blocked: '{pattern}'\033[0m")
                return "Permission denied by deny list"
        for kw in DESTRUCTIVE:
            if kw in tool_args.get("command", ""):
                print(f"\n\033[33m⚠  Potentially destructive command\033[0m")
                print(f"   Tool: {tool_call.function.name}({tool_args})")
                choice = input("   Allow? [y/N] ").strip().lower()
                if choice not in ("y", "yes"):
                    return "Permission denied by user"
    if tool_call.function.name in ("write_file", "edit_file"):
        path = tool_args.get("path", "")
        if not (WORKDIR / path).resolve().is_relative_to(WORKDIR):
            print(f"\n\033[33m⚠  Writing outside workspace\033[0m")
            print(f"   Tool: {tool_call.function.name}({tool_args})")
            choice = input("   Allow? [y/N] ").strip().lower()
            if choice not in ("y", "yes"):
                return "Permission denied by user"
    return None

# PreToolUse: logging
def log_hook(tool_call):
    if tool_call.type != "function":
        return
    print(f"[HOOK] {tool_call.function.name}(...)")

# PostToolUse: large output reminder
def large_output_hook(tool_call, output):
    if len(str(output)) > 100000:
        print(f"[HOOK] ⚠ Large output from {tool_call.function.name}")

# Stop: print session summary
def summary_hook(messages: list) -> str | None:
    """Print a summary when the agent loop is about to stop."""

    print("===== Session Summary ==========")
    tool_count = 0
    for message in messages:
        # 只统计 assistant 发起的工具调用
        role = _get_field(message, "role")
        if role != "assistant":
            continue
        # OpenAI Chat Completions API:
        # 一条 assistant 消息可能发起多个工具调用
        tool_calls = _get_field(message, "tool_calls")
        if tool_calls:
            tool_count += len(tool_calls)
            continue
        # 兼容旧格式的 function_call
        function_call = _get_field(message, "function_call")
        if function_call:
            if isinstance(function_call, (list, tuple)):
                tool_count += len(function_call)
            else:
                tool_count += 1
    print(
        f"\033[90m[HOOK] Stop: "
        f"session used {tool_count} tool calls\033[0m"
    )
    return None  # None = allow stop; string = force continuation

# ═══════════════════════════════════════════════════════════
#  注册Hooks
# ═══════════════════════════════════════════════════════════

HOOKS = {
    "UserPromptSubmit": [],
    "PreToolUse": [],
    "PostToolUse": [],
    "Stop": [],
}

def register_hook(event: str, callback):
    HOOKS[event].append(callback)

def trigger_hooks(event: str, *args):
    for callback in HOOKS[event]:
        result = callback(*args)
        if result is not None:   # return value ≠ None → hook says "stop"
            return result
    return None

register_hook("UserPromptSubmit", context_inject_hook)
register_hook("PreToolUse", permission_hook)
register_hook("PreToolUse", log_hook)
register_hook("PostToolUse", large_output_hook)
register_hook("Stop", summary_hook)

# ═══════════════════════════════════════════════════════════
#  定义模型
# ═══════════════════════════════════════════════════════════
load_dotenv()
api_key = os.getenv("AI_API_KEY")
base_url = os.getenv("AI_BASE_URL")
model_name = os.getenv("AI_MODEL_NAME") or "deepseek-flash"
client = OpenAI(api_key=api_key, base_url=base_url)

def agent_loop(messages: list[ChatCompletionMessageParam]):
    rounds_since_todo = 0
    for _ in range(0, MAX_TURNS):
        if rounds_since_todo >= 3:
            messages.append({"role": "user",
                             "content": "<reminder>Update your todos.</reminder>"})
            rounds_since_todo = 0
        response = client.chat.completions.create(model=model_name, messages=messages, tools=TOOLS, )
        assistant_message = response.choices[0].message
        messages.append(cast(ChatCompletionMessageParam, assistant_message))
        print(f"\n{'=' * 20} Loop {_} {'=' * 20}")
        # 1. 输出 AI 文本回复
        if assistant_message.content:
            print(f"[AI]\n{assistant_message.content}")

        # 2. 输出工具调用
        if assistant_message.tool_calls:
            for i, tool_call in enumerate(assistant_message.tool_calls, start=1):
                if tool_call.type != "function":
                    continue
                print(f"\n[Tool Call {i}]")
                print(f"  Name: {tool_call.function.name}")
                print(f"  Arguments: {tool_call.function.arguments}")

        # 3. 输出拒绝信息（如果有）
        if assistant_message.refusal:
            print(f"\n[Refusal]\n{assistant_message.refusal}")

        print(f"{'=' * 50}\n")
        if not assistant_message.tool_calls:
            force = trigger_hooks("Stop", messages)
            if force:
                messages.append({ "role": "assistant", "content": force })
                continue
            return
        rounds_since_todo += 1
        for tc in assistant_message.tool_calls:
            if tc.type == "function":
                blocked = trigger_hooks("PreToolUse", tc)
                if blocked:
                    messages.append({
                                        "role": "tool",
                                        "tool_call_id": tc.id,
                                        "content": "Permission denied"
                                    })
                    continue
                tool_name = tc.function.name
                tool_args = json.loads(tc.function.arguments)
                tool = TOOL_HANDLERS[tool_name]
                output = tool(**tool_args) if tool else f"Unknown: {tool_name}"
                print(f"Tool {tool_name}({tool_args}) -> {str(output)[:200]}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(output)
                })

if __name__ == "__main__":
    print("输入问题，回车发送。输入 q 退出。\n")
    print("Skill Loading — catalog in SYSTEM, content on demand")
    messages: list[ChatCompletionMessageParam] = [{ "role": "system", "content": build_system_prompt() }]
    while True:
        try:
            query = input("\033[36ms00 >> \033[0m")
            trigger_hooks("UserPromptSubmit", query)
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "quit", "exit", ""):
            break
        messages.append({ "role": "user", "content": query })
        agent_loop(messages=messages)

        last_message = messages[-1] if messages else None
        print(f"\n{'=' * 20} Test {'=' * 20}\n")
        if last_message and _get_field(last_message, "role") == "assistant" and _get_field(last_message, "content", ""):
            print(_get_field(last_message, "content"))
        print()
