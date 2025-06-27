import os
from google import genai
from google.genai import types
 
# Configure the client (ensure GEMINI_API_KEY is set in your environment)
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
 
# Define the function declaration for the model
weather_function = {
    "name": "get_current_temperature",
    "description": "Gets the current temperature for a given location.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city name, e.g. San Francisco",
            },
        },
        "required": ["location"],
    },
}
 
# Placeholder function to simulate API call
def get_current_temperature(location: str) -> dict:
    return {"temperature": "15", "unit": "Celsius"}
 
# Create the config object as shown in the user's example
# Use client.models.generate_content with model, contents, and config
tools = types.Tool(function_declarations=[weather_function])
contents = ["What's the temperature in London right now?"]
response = client.models.generate_content(
    model='gemini-2.0-flash',
    contents=contents,
    config = types.GenerateContentConfig(tools=[tools])
)
 
# Process the Response (Check for Function Call)
response_part = response.candidates[0].content.parts[0]
if response_part.function_call:
    function_call = response_part.function_call
    print(f"Function to call: {function_call.name}")
    print(f"Arguments: {dict(function_call.args)}")
 
    # Execute the Function
    if function_call.name == "get_current_temperature":        
        # Call the actual function
        api_result = get_current_temperature(*function_call.args)
        # Append function call and result of the function execution to contents
        follow_up_contents = [
            types.Part(function_call=function_call),
            types.Part.from_function_response(
                name="get_current_temperature",
                response=api_result
            )
        ]
        # Generate final response
        response_final = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=contents + follow_up_contents,
            config=types.GenerateContentConfig(tools=[tools])
        )
        print(response_final.text)
    else:
        print(f"Error: Unknown function call requested: {function_call.name}")
else:
    print("No function call found in the response.")
    print(response.text)