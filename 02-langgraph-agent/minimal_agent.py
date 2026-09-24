import os

from langchain_openai import ChatOpenAI
from openai import OpenAI
from pydantic import SecretStr
import requests
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from typing import Annotated, TypedDict
import operator

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

@tool
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

class MinimalAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

class MinimalAgent:
    def __init__(self, model: ChatOpenAI, tools: list[BaseTool], system_prompt: str = ""):
        self.llm = model.bind_tools(tools)
        self.system_prompt = system_prompt
        self.tools_map = { tool.name: tool for tool in tools }

        # init graph
        graph = StateGraph(MinimalAgentState)
        graph.add_node("llm", self.llm_node)
        graph.add_node("action", self.action_node)
        graph.add_edge("action", "llm")
        graph.add_conditional_edges("llm", self.should_take_action, { "action": "action", "end": END })
        graph.set_entry_point("llm")
        self.graph = graph.compile()

    def should_take_action(self, state: MinimalAgentState) -> str:
        if (len(state["messages"]) > 0):
            last_message = state["messages"][-1]
            if isinstance(last_message, AIMessage) and len(last_message.tool_calls) > 0:
                return "action"
        return "end"

    def llm_node(self, state: MinimalAgentState) -> MinimalAgentState:
        messages = state["messages"]
        if self.system_prompt:
            messages = [self.system_prompt] + messages
        result = self.llm.invoke(messages)
        return { "messages": [result] }

    def action_node(self, state: MinimalAgentState) -> MinimalAgentState:
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            return { "messages": [] }

        # call tool iteratively
        tool_calls = last_message.tool_calls
        tc_messages: list[AnyMessage] = []
        for tc in tool_calls:
            tool = self.tools_map[tc["name"]]
            if not tool:
                print(f"Call Tool Error: Tool {tc['name']} not exist")
            tc_result = tool.invoke(tc["args"])
            print(f"Call Tool {tc["name"]}, result: {tc_result}")
            tc_messages.append(ToolMessage(tool_call_id=tc["id"], content=tc_result))
        return { "messages": tc_messages }
        

def main():
    # load llm config from env
    load_dotenv()
    base_url = os.getenv("AI_BASE_URL")
    model_name = os.getenv("AI_MODEL_NAME") or ""
    env_api_key = os.getenv("AI_API_KEY")
    api_key = SecretStr(env_api_key) if env_api_key else None

    # create llm model
    ai_model = ChatOpenAI(base_url=base_url, api_key=api_key, model=model_name)
    agent = MinimalAgent(model=ai_model, tools=[get_weather], system_prompt=system_prompt)
    result = agent.graph.invoke({ "messages": [HumanMessage(content="What is the weather like in Tokyo right now?")] })
    print("llm response:")
    print(result["messages"][-1].content)

if __name__ == "__main__":
    main()
