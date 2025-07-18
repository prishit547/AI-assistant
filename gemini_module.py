# modules/gemini_module.py

import os
from dotenv import load_dotenv
from google.generativeai import configure, GenerativeModel

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
configure(api_key=API_KEY)

# 🧠 Customize Gemini with Jarvis-style prompt
PROMPT_TEMPLATE = """
You are Jarvis, an intelligent, concise, and slightly witty AI assistant.
Always answer in a short, to-the-point, confident style like Jarvis from Iron Man.
Avoid unnecessary words. Respond as if you're helping your creator.

Query: {query}
"""

model = GenerativeModel("gemini-2.5-flash")

def ask_gemini(prompt: str) -> str:
    full_prompt = PROMPT_TEMPLATE.format(query=prompt)
    try:
        response = model.generate_content(full_prompt)
        return response.text.strip()
    except Exception as e:
        return f"[Gemini Error] {str(e)}"
