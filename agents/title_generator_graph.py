from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
import json
from dotenv import load_dotenv

load_dotenv()

# We set up three different LLM behaviors:
llm_strict = ChatGroq(
    temperature=0,
    model="llama-3.3-70b-versatile",
    model_kwargs={"response_format": {"type": "json_object"}}
)

llm_creative = ChatGroq(
    temperature=0.7,  # High temperature for brainstorming
    model="llama-3.3-70b-versatile",
    model_kwargs={"response_format": {"type": "json_object"}}
)

# 1. Sub-State
class TitleState(TypedDict):
    context: dict
    text: str
    feedback: str
    
    creative_title: str
    technical_title: str
    seo_title: str
    
    final_title: str

# 2. Parallel Generator Nodes
def generate_creative(state: TitleState):
    prompt = f"""
    You are a visionary copywriter. Brainstorm a highly creative, catchy, and engaging title for this AI project.
    
    Context: {state.get("context")}
    Text: {state.get("text")}
    
    Return a JSON object with exactly one key "title" mapped to your single best creative title string.
    Make it punchy and memorable. Do not use quotes inside the string.
    """
    
    response = llm_creative.invoke(prompt)
    try:
        title = json.loads(response.content).get("title", "")
    except json.JSONDecodeError:
        title = "Creative Title Fallback"
        
    print(f"   [Creative Node] Drafted: {title}")
    return {"creative_title": title}

def generate_technical(state: TitleState):
    prompt = f"""
    You are a rigorous academic researcher. Draft a highly formal, precise, and technical title for this AI project publication.
    
    Context: {state.get("context")}
    Text: {state.get("text")}
    
    Return a JSON object with exactly one key "title" mapped to your single best technical title string.
    Focus on methodology, scope, and objective accuracy. Do no use quotes inside the string.
    """
    
    response = llm_strict.invoke(prompt)
    try:
        title = json.loads(response.content).get("title", "")
    except json.JSONDecodeError:
        title = "Technical Title Fallback"

    print(f"   [Technical Node] Drafted: {title}")
    return {"technical_title": title}

def generate_seo(state: TitleState):
    prompt = f"""
    You are an SEO specialist. Draft a highly searchable, clear, and keyword-dense title for this AI project.
    
    Context: {state.get("context")}
    Text: {state.get("text")}
    
    Return a JSON object with exactly one key "title" mapped to your single best SEO-optimized title string.
    Focus on clarity and exactly what the target audience would Google. Do no use quotes inside the string.
    """
    
    response = llm_strict.invoke(prompt)
    try:
        title = json.loads(response.content).get("title", "")
    except json.JSONDecodeError:
        title = "SEO Keyword Title Fallback"

    print(f"   [SEO Node] Drafted: {title}")
    return {"seo_title": title}

# 3. Aggregation Node (Ranker)
def rank_titles(state: TitleState):
    c_title = state.get("creative_title", "")
    t_title = state.get("technical_title", "")
    s_title = state.get("seo_title", "")
    feedback = state.get("feedback", "")
    
    extra = f"WARNING! The outer Reviewer Agent rejected a previous title with this feedback: {feedback}. ENSURE your final selection fixes this!" if feedback else ""

    prompt = f"""
    You are the Executive Editor (Title Aggregator). You must evaluate three candidate titles for an AI project:
    
    Option 1 (Creative): "{c_title}"
    Option 2 (Technical): "{t_title}"
    Option 3 (SEO/Clear): "{s_title}"
    
    Project Context: {state.get("context")}
    
    Critique the options, heavily penalizing any title that violates the outer constraints.
    {extra}
    
    Select the absolute best overall title from the three, OR synthesize a new flawless hybrid title based on them.
    Return a JSON object with a single key "title" mapped to your final polished string.
    """
    
    response = llm_strict.invoke(prompt)
    try:
        final_title = json.loads(response.content).get("title", c_title)
    except json.JSONDecodeError:
        final_title = c_title

    print(f"   [Ranker Node] Final Selection: {final_title}")
    return {"final_title": final_title}

# 4. Compile Sub-Graph
builder = StateGraph(TitleState)
builder.add_node("creative", generate_creative)
builder.add_node("technical", generate_technical)
builder.add_node("seo", generate_seo)
builder.add_node("ranker", rank_titles)

builder.add_edge(START, "creative")
builder.add_edge(START, "technical")
builder.add_edge(START, "seo")
builder.add_edge(["creative", "technical", "seo"], "ranker")
builder.add_edge("ranker", END)

title_generator_subgraph = builder.compile()
