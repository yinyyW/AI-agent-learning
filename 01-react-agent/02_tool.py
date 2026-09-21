from typing import List, cast

from openai import OpenAI
from openai.types.chat import ChatCompletionFunctionToolParam, ChatCompletionMessageParam, ChatCompletionMessageFunctionToolCallParam
from dotenv import load_dotenv
import os
import json

# define tools
def add(a: int, b: int) -> int:
    return a + b

def main():
    tools_map = { "add": add }
    tools: list[ChatCompletionFunctionToolParam] = [
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Add two numbers",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"}
                    },
                    "required": ["a", "b"]
                },
            }
        }
    ]

    # get api key from .env
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")

    # create LLM client
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    # chat with LLM
    messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": "Calculate 123 + 345"}]
    response = client.chat.completions.create(
        model="deepseek-flash",
        tools=tools,
        messages=messages
    )

    # tools calling
    print("tool call response content:")
    tool_calls = response.choices[0].message.tool_calls
    # update messages
    messages.append(cast(ChatCompletionMessageParam, response.choices[0].message))
    if tool_calls:
        for tool_call in tool_calls:
            if tool_call.type == "function":
                # get tool
                tool = tools_map[tool_call.function.name]
                # get tool arguments
                arguments = json.loads(tool_call.function.arguments)
                tool_result = tool(**arguments)
                # update messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(tool_result)
                })

    # send tool calling result to llm
    response = client.chat.completions.create(
        model="deepseek-flash",
        tools=tools,
        messages=messages
    )
    print("response result: ")
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()