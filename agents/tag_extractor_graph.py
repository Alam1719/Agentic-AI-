from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from agents.gazetteer import TECH_KEYWORDS
import spacy
from langchain_groq import ChatGroq
import json
import os
from dotenv import load_dotenv

load_dotenv()

# We try to load the spaCy model gracefully.
try:
    nlp = spacy.load("en_core_web_sm")
except BaseException:
    # Failsafe if download is not complete yet
    nlp = None

llm = ChatGroq(
    temperature=0,
    model="llama-3.3-70b-versatile",
    model_kwargs={"response_format": {"type": "json_object"}}
)

# 1. Define Sub-State
class TagState(TypedDict):
    context: dict
    text: str
    feedback: str
    
    gazetteer_tags: List[str]
    spacy_tags: List[str]
    llm_tags: List[str]
    
    final_tags: List[str]

# 2. Nodes
def extract_gazetteer(state: TagState):
    text = state.get("text", "").lower()
    context_str = str(state.get("context", "")).lower()
    combined = text + " " + context_str
    
    found_tags = []
    for keyword in TECH_KEYWORDS:
        if keyword.lower() in combined:
            found_tags.append(keyword)
            
    print(f"   [Gazetteer] Extracted: {found_tags}")
    return {"gazetteer_tags": list(set(found_tags))}

def extract_spacy(state: TagState):
    text = state.get("text", "")
    context_str = str(state.get("context", ""))
    combined = text + " " + context_str
    
    tags = []
    if nlp is not None:
        doc = nlp(combined)
        for ent in doc.ents:
            if ent.label_ in ["ORG", "PRODUCT", "WORK_OF_ART", "GPE", "LOC", "EVENT"]:
                tags.append(ent.text)
                
    print(f"   [spaCy NER] Extracted: {list(set(tags))}")
    return {"spacy_tags": list(set(tags))}

def extract_llm(state: TagState):
    context = state.get("context", "")
    text = state.get("text", "")
    feedback = state.get("feedback", "")
    
    extra = f"Feedback to correct from Reviewer: {feedback}" if feedback else ""
        
    prompt = f"""
    Extract relevant tags for this AI project.
    Context: {context}
    Text: {text}
    
    Return a JSON object with a single key "tags" mapped to a list of loose tags.
    {extra}
    """
    
    response = llm.invoke(prompt)
    try:
        data = json.loads(response.content)
        tags = data.get("tags", [])
    except json.JSONDecodeError:
        tags = []
        
    print(f"   [LLM Generator] Extracted: {tags}")
    return {"llm_tags": tags}

def aggregate_tags(state: TagState):
    g_tags = state.get("gazetteer_tags", [])
    s_tags = state.get("spacy_tags", [])
    l_tags = state.get("llm_tags", [])
    
    all_tags = list(set(g_tags + s_tags + l_tags))
    
    prompt = f"""
    You are the final Aggregator Node. Review this pool of candidate tags:
    {all_tags}
    
    Based on the project context: {state.get('context')}
    Select the Top 5-8 most relevant and professional tags from the pool.
    Return a JSON object with a single key "tags" mapped to your final list of 5-8 tags.
    """
    
    response = llm.invoke(prompt)
    try:
        data = json.loads(response.content)
        final_tags = data.get("tags", [])
    except json.JSONDecodeError:
        final_tags = all_tags[:5]
        
    print(f"   [Aggregator] Synthesized Final 5-8 Tags: {final_tags}")
    return {"final_tags": final_tags}

# 3. Compile Sub-Graph
builder = StateGraph(TagState)
builder.add_node("gazetteer", extract_gazetteer)
builder.add_node("spacy", extract_spacy)
builder.add_node("llm", extract_llm)
builder.add_node("aggregate", aggregate_tags)

builder.add_edge(START, "gazetteer")
builder.add_edge(START, "spacy")
builder.add_edge(START, "llm")
builder.add_edge(["gazetteer", "spacy", "llm"], "aggregate")
builder.add_edge("aggregate", END)

tag_extractor_subgraph = builder.compile()
