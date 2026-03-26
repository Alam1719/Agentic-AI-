from langchain_groq import ChatGroq
from dotenv import load_dotenv
import json

load_dotenv()

llm = ChatGroq(
    temperature=0,
    model="llama-3.3-70b-versatile",
    model_kwargs={"response_format": {"type": "json_object"}}
)

def reviewer_agent(state):
    title = state.get("title", "")
    tags = state.get("tags", [])
    tldr = state.get("tldr", "")
    
    prompt = f"""
    Review the following AI-generated output for quality, relevance, and completeness.
    
    Title: {title}
    Tags: {tags}
    TL;DR: {tldr}
    
    Evaluate if this output meets the required logical constraints:
    1. Title must be professional and catchy.
    2. Tags must have 5-8 items.
    3. TL;DR must be concise (2-3 lines).
    
    Return a JSON object uniquely providing:
    - "status": exactly "pass" or "fail"
    - "feedback": a dictionary mapping the field name ("title", "tags", or "tldr") to specific improvement instructions if it failed. If it passes, feedback can be empty.
    
    Example:
    {{"status": "fail", "feedback": {{"tags": "Only found 3 tags, need 5-8."}}}}
    """
    
    response = llm.invoke(prompt)
    
    try:
        data = json.loads(response.content)
        status = data.get("status", "pass")
        feedback = data.get("feedback", {})
    except json.JSONDecodeError:
        status = "pass"
        feedback = {}
        
    return {"review": {"status": status}, "feedback": feedback}
