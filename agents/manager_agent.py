from dotenv import load_dotenv
import os

import json

load_dotenv()

from langchain_groq import ChatGroq

llm = ChatGroq(temperature=0, model="llama-3.3-70b-versatile")

def manager_agent(state):
    text = state["input_text"]

    prompt = f"""
    Analyze the following project and extract:
    - main themes
    - target audience
    - main goal

    Text:
    {text}

    Return as a JSON object with the keys "main_themes", "target_audience", and "main_goal".
    Do not include markdown formatting or backticks.
    """

    response = llm.invoke(prompt)
    
    raw_content = response.content.strip()
    if raw_content.startswith("```json"):
        raw_content = raw_content[7:]
    if raw_content.endswith("```"):
        raw_content = raw_content[:-3]
    raw_content = raw_content.strip()

    try:
        context = json.loads(raw_content)
    except json.JSONDecodeError:
        context = {"summary": response.content}

    return {"context": context}