from google import genai
from pydantic import BaseModel, Field
 
# Configure the client (ensure GEMINI_API_KEY is set in your environment)
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
 
# Define Structured Output Schemas
class Response(BaseModel):
    handoff: str = Field(default="", description="The name/role of the agent to hand off to. Available agents: 'Restaurant Agent', 'Hotel Agent'")
    message: str = Field(description="The response message to the user or context for the next agent")
 
# Agent Function
def run_agent(agent_name: str, system_prompt: str, prompt: str) -> Response:
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=prompt,
        config = {'system_instruction': f'You are {agent_name}. {system_prompt}', 'response_mime_type': 'application/json', 'response_schema': Response}
    )
    return response.parsed
 
 
# Define System Prompts for the agents
hotel_system_prompt = "You are a Hotel Booking Agent. You ONLY handle hotel bookings. If the user asks about restaurants, flights, or anything else, respond with a short handoff message containing the original request and set the 'handoff' field to 'Restaurant Agent'. Otherwise, handle the hotel request and leave 'handoff' empty."
restaurant_system_prompt = "You are a Restaurant Booking Agent. You handle restaurant recommendations and bookings based on the user's request provided in the prompt."
 
# Prompt to be about a restaurant
initial_prompt = "Can you book me a table at an Italian restaurant for 2 people tonight?"
print(f"Initial User Request: {initial_prompt}")
 
# Run the first agent (Hotel Agent) to force handoff logic
output = run_agent("Hotel Agent", hotel_system_prompt, initial_prompt)
 
# simulate a user interaction to change the prompt and handoff
if output.handoff == "Restaurant Agent":
    print("Handoff Triggered: Hotel to Restaurant")
    output = run_agent("Restaurant Agent", restaurant_system_prompt, initial_prompt)
elif output.handoff == "Hotel Agent":
    print("Handoff Triggered: Restaurant to Hotel")
    output = run_agent("Hotel Agent", hotel_system_prompt, initial_prompt)
 
print(output.message)    


# Use Cases:

# Simulating debates or brainstorming sessions with different AI personas.
# Complex software creation involving agents for planning, coding, testing, and deployment.
# Running virtual experiments or simulations with agents representing different actors.
# Collaborative writing or content creation processes.