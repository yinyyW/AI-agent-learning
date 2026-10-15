import json
import os, subprocess
from pathlib import Path
from typing import cast
from openai.types.chat import ChatCompletion, ChatCompletionFunctionToolParam, ChatCompletionMessageParam, ChatCompletionMessageFunctionToolCallParam

from dotenv import load_dotenv
from openai import OpenAI

# system prompt
system_prompt = """
You are a ReAct-style AI Agent.

Your task is to solve the user's request by following this loop:

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

WORKDIR = Path.cwd()

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
    }
]

# ═══════════════════════════════════════════════════════════
#  工具分发映射
# ═══════════════════════════════════════════════════════════
TOOL_HANDLERS = {
    "bash": run_bash, "read_file": run_read, "write_file": run_write,
    "edit_file": run_edit, "glob": run_glob,
}

# ═══════════════════════════════════════════════════════════
#  定义模型
# ═══════════════════════════════════════════════════════════
load_dotenv()
api_key = os.getenv("AI_API_KEY")
base_url = os.getenv("AI_BASE_URL")
model_name = os.getenv("AI_MODEL_NAME") or "deepseek-flash"
client = OpenAI(api_key=api_key, base_url=base_url)

def agent_loop(messages: list[ChatCompletionMessageParam]):
    MAX_TURNS = 20
    for _ in range(0, MAX_TURNS):
        response = client.chat.completions.create(model=model_name, messages=messages, tools=TOOLS)
        assistant_message = response.choices[0].message
        messages.append(cast(ChatCompletionMessageParam, assistant_message))
        print(f"Loop {_ + 1}, AI response: {assistant_message}\n")
        if not assistant_message.tool_calls:
            break
        results = []
        for tc in assistant_message.tool_calls:
            if tc.type == "function":
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

    messages: list[ChatCompletionMessageParam] = [{"role": "system", "content": system_prompt}]

    while True:
        try:
            query = input("\033[36ms00 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        messages.append({ "role": "user", "content": query })
        agent_loop(messages=messages)
        if messages[-1]:
            print(messages[-1])
        print()
