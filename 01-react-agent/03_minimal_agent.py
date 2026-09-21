from typing import Any, Callable, cast

import requests
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionFunctionToolParam, ChatCompletionMessageParam, ChatCompletionMessageFunctionToolCallParam
from dotenv import load_dotenv
import os
import json

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

# define tools
def get_coordinates(city: str) -> tuple[float, float, str]:
    """Get latitude, longitude and timezone for a city."""

    url = "https://geocoding-api.open-meteo.com/v1/search"

    params = {
        "name": city,
        "count": 1,
        "language": "en",
        "format": "json",
    }

    response = requests.get(url, params=params)
    data = response.json()

    results = data.get("results")

    if not results:
        raise ValueError(f"City not found: {city}")

    location = results[0]

    return (
        location["latitude"],
        location["longitude"],
        location["timezone"],
    )

def get_weather(city: str) -> str:
    """Get current weather for a city."""

    latitude, longitude, timezone = get_coordinates(city)

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": (
            "temperature_2m,"
            "apparent_temperature,"
            "weather_code,"
            "wind_speed_10m"
        ),
        "timezone": timezone,
    }

    response = requests.get(url, params=params)
    data = response.json()

    current = data["current"]

    return (
        f"Temperature: {current['temperature_2m']}°C, "
        f"Feels like: {current['apparent_temperature']}°C, "
        f"Wind speed: {current['wind_speed_10m']} km/h, "
        f"Weather code: {current['weather_code']}"
    )

def add(a: int, b: int) -> int:
    return a + b

def multiply(a: int, b: int) -> int:
    return a * b

class MinimalAgent:
    def __init__(self, system_prompt: str, model_name: str, tools: list[ChatCompletionFunctionToolParam], tools_registry: dict[str, Callable[..., Any]] , max_turns=5):
        self.max_turns = max_turns
        # init llm client
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.model_name = model_name

        # init messages
        self.messages: list[ChatCompletionMessageParam] = []
        if system_prompt:
            self.messages.append({ "role": "system", "content": system_prompt })

        # init tools
        if tools:
            self.tools = tools
        if tools_registry:
            self.tool_registry = tools_registry

    def __call__(self, message: str) -> str:
        # add user message into messages list
        self.messages.append({ "role": "user", "content": message })

        # execute with agent loop
        response = self.loop_execute()
        if response and response.choices[0].message.content:
            return response.choices[0].message.content
        return ""
    
    def loop_execute(self) -> ChatCompletion | None:
        for i in range(0, self.max_turns):
            # call llm
            print(f"Loop {i + 1}")
            response = self.client.chat.completions.create(
                messages=self.messages,
                model=self.model_name,
                tools=self.tools)

            # trace response
            assistant_message = response.choices[0].message

            # update messages
            self.messages.append(cast(ChatCompletionMessageParam, assistant_message))
            
            # call tools
            tool_calls = response.choices[0].message.tool_calls
            if tool_calls:
                for tc in tool_calls:
                    if tc.type == "function":
                        print(f"### calling tool {tc.function.name} ###")
                        tool = self.tool_registry[tc.function.name]
                        arguments = json.loads(tc.function.arguments)
                        tool_result = tool(**arguments)
                        print(f"### calling tool result: {tool_result} ###")
                        # update messages
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": str(tool_result)
                        })
            else:
                return response
            
def main():
    tool_registry = {
        "add": add,
        "multiply": multiply,
        "get_weather": get_weather
    }
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
        },
        {
            "type": "function",
            "function": {
                "name": "multiply",
                "description": "multiply two numbers",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"}
                    },
                    "required": ["a", "b"]
                },
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "get city's weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                    },
                    "required": ["city"]
                },
            }
        }
    ]
    agent = MinimalAgent(system_prompt=system_prompt, model_name="deepseek-flash", tools=tools, tools_registry=tool_registry)

    user_query = "I'll travel to Shanghai tomorrow, give me advice about what clothes should I bring."
    response = agent(user_query)
    print(f"agent response: {response}")

if __name__ == "__main__":
    main()