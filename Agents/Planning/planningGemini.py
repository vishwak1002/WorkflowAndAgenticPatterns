 
import os
from google import genai
from pydantic import BaseModel, Field
from typing import List
 
# Configure the client (ensure GEMINI_API_KEY is set in your environment)
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
 
# Define the Plan Schema
class Task(BaseModel):
    task_id: int
    description: str
    assigned_to: str = Field(description="Which worker type should handle this? E.g., Researcher, Writer, Coder")
 
class Plan(BaseModel):
    goal: str
    steps: List[Task]
 
# Step 1: Generate the Plan (Planner LLM)
user_goal = "Write a short blog post about the benefits of AI agents."
 
prompt_planner = f"""
Create a step-by-step plan to achieve the following goal. 
Assign each step to a hypothetical worker type (Researcher, Writer).
 
Goal: {user_goal}
"""
 
print(f"Goal: {user_goal}")
print("Generating plan...")
 
# Use a model capable of planning and structured output
response_plan = client.models.generate_content(
    model='gemini-2.5-pro-preview-03-25',
    contents=prompt_planner,
    config={
        'response_mime_type': 'application/json',
        'response_schema': Plan,
    },
)
 
# Step 2: Execute the Plan (Orchestrator/Workers - Omitted for brevity) 
for step in response_plan.parsed.steps:
    print(f"Step {step.task_id}: {step.description} (Assignee: {step.assigned_to})")


# Use Cases:

# Complex software development tasks: Breaking down "build a feature" into planning, coding, testing, and documentation subtasks.
# Research and report generation: Planning steps like literature search, data extraction, analysis, and report writing.
# Multi-modal tasks: Planning steps involving image generation, text analysis, and data integration.
# Executing complex user requests like "Plan a 3-day trip to Paris, book flights and a hotel within my budget."