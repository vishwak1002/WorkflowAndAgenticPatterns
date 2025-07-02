import os 
import json 
from dotenv import load_dotenv
from typing import Annotated,Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langchain_core.messages import ToolMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from tools import get_weather_forecast
from langgraph.graph.message import add_messages 
from langchain_google_genai import ChatGoogleGenerativeAI # helper function to add messages to the state
# Read your API key from the environment variable or set it manually


 
 
class AgentState(TypedDict):
    """The state of the agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    number_of_steps: int

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY","xxx")
tools = [get_weather_forecast]
tools_by_name = {tool.name: tool for tool in tools}
llm = ChatGoogleGenerativeAI(
    model= "gemini-2.0-flash", # replace with "gemini-2.0-flash"
    temperature=1.0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    google_api_key=api_key,
)
 
# Bind tools to the model
model = llm.bind_tools([get_weather_forecast])
# this is similar to customizing the create_react_agent with 'prompt' parameter, but is more flexible
# system_prompt = SystemMessage(
#     "You are a helpful assistant that use tools to access and retrieve information from a weather API. Today is 2025-03-04. Help the user with their questions. Use the history to answer the question."
# )
 
# Define our tool node
def call_tool(state: AgentState):
    outputs = []
    # Iterate over the tool calls in the last message
    for tool_call in state["messages"][-1].tool_calls:
        # Get the tool by name
        tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
        outputs.append(
            ToolMessage(
                content=tool_result,
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
    return {"messages": outputs}
 
def call_model(
    state: AgentState,
    config: RunnableConfig,
):
    # Invoke the model with the system prompt and the messages
    response = model.invoke(state["messages"], config)
    # We return a list, because this will get added to the existing messages state using the add_messages reducer
    return {"messages": [response]}
 
 
# Define the conditional edge that determines whether to continue or not
def should_continue(state: AgentState):
    messages = state["messages"]
    # If the last message is not a tool call, then we finish
    if not messages[-1].tool_calls:
        return "end"
    # default to continue
    return "continue"