import os
import json
from google import genai
from pydantic import BaseModel
import enum
 
# Configure the client (ensure GEMINI_API_KEY is set in your environment)
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
 
# Define Routing Schema
class Category(enum.Enum):
    WEATHER = "weather"
    SCIENCE = "science"
    UNKNOWN = "unknown"
 
class RoutingDecision(BaseModel):
    category: Category
    reasoning: str
 
# Step 1: Route the Query
user_query = "What's the weather like in Paris?"
# user_query = "Explain quantum physics simply."
# user_query = "What is the capital of France?"
 
prompt_router = f"""
Analyze the user query below and determine its category.
Categories:
- weather: For questions about weather conditions.
- science: For questions about science.
- unknown: If the category is unclear.
 
Query: {user_query}
"""
 
# Use client.models.generate_content with config for structured output
response_router = client.models.generate_content(
    model= 'gemini-2.0-flash-lite',
    contents=prompt_router,
    config={
        'response_mime_type': 'application/json',
        'response_schema': RoutingDecision,
    },
)
print(f"Routing Decision: Category={response_router.parsed.category}, Reasoning={response_router.parsed.reasoning}")
 
# Step 2: Handoff based on Routing
final_response = ""
if response_router.parsed.category == Category.WEATHER:
    weather_prompt = f"Provide a brief weather forecast for the location mentioned in: '{user_query}'"
    weather_response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=weather_prompt
    )
    final_response = weather_response.text
elif response_router.parsed.category == Category.SCIENCE:
    science_response = client.models.generate_content(
        model="gemini-2.5-flash-preview-04-17",
        contents=user_query
    )
    final_response = science_response.text
else:
    unknown_response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=f"The user query is: {prompt_router}, but could not be answered. Here is the reasoning: {response_router.parsed.reasoning}. Write a helpful response to the user for him to try again."
    )
    final_response = unknown_response.text
print(f"\nFinal Response: {final_response}")
 



#  Use Cases:

# RAG with query decomposition: Breaking a complex query into sub-queries, running retrievals for each in parallel, and synthesizing the results.
# Analyzing large documents: Dividing the document into sections, summarizing each section in parallel, and then combining the summaries.
# Generating multiple perspectives: Asking multiple LLMs the same question with different persona prompts and aggregating their responses.
# Map-reduce style operations on data.