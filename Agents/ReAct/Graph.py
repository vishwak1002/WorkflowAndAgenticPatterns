from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI 
from dotenv import load_dotenv
import os
from tools import get_weather_forecast
from ReACTGemini import AgentState,call_tool,call_model,should_continue
# Load the .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY","xxx")

 # Create LLM class 
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
 
# Test the model with tools
model.invoke("What is the weather in Berlin on 12th of March 2025?")

# Define a new graph with our state
workflow = StateGraph(AgentState)
 
# 1. Add our nodes 
workflow.add_node("llm", call_model)
workflow.add_node("tools",  call_tool)
# 2. Set the entrypoint as `agent`, this is the first node called
workflow.set_entry_point("llm")
# 3. Add a conditional edge after the `llm` node is called.
workflow.add_conditional_edges(
    # Edge is used after the `llm` node is called.
    "llm",
    # The function that will determine which node is called next.
    should_continue,
    # Mapping for where to go next, keys are strings from the function return, and the values are other nodes.
    # END is a special node marking that the graph is finish.
    {
        # If `tools`, then we call the tool node.
        "continue": "tools",
        # Otherwise we finish.
        "end": END,
    },
)
# 4. Add a normal edge after `tools` is called, `llm` node is called next.
workflow.add_edge("tools", "llm")
 
# Now we can compile and visualize our graph
graph = workflow.compile()


 
from IPython.display import Image, display
 
display(Image(graph.get_graph().draw_mermaid_png()))



# Create our initial message dictionary
inputs = {"messages": [("user", "How is the weather in Berlin today (2025-07-02)")]}
 
# call our graph with streaming to see the steps
 
for state in graph.stream(inputs, stream_mode="values"):
    last_message = state["messages"][-1]
    last_message.pretty_print()

# This is graph one