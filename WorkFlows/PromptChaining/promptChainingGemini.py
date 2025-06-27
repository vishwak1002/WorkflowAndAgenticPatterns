import os
from google import genai

 
# Configure the client (ensure GEMINI_API_KEY is set in your environment)
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
 
# --- Step 1: Summarize Text ---
original_text = "Large language models are powerful AI systems trained on vast amounts of text data. They can generate human-like text, translate languages, write different kinds of creative content, and answer your questions in an informative way."
prompt1 = f"Summarize the following text in one sentence: {original_text}"
 
# Use client.models.generate_content
response1 = client.models.generate_content(
    model='gemini-2.0-flash',
    contents=prompt1
)
summary = response1.text.strip()
print(f"Summary: {summary}")
 
# --- Step 2: Translate the Summary ---
prompt2 = f"Translate the following summary into French, only return the translation, no other text: {summary}"
 
# Use client.models.generate_content
response2 = client.models.generate_content(
    model='gemini-2.0-flash',
    contents=prompt2
)
translation = response2.text.strip()
print(f"Translation: {translation}")
 