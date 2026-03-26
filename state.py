from typing import TypedDict, List, Dict

class AgentState(TypedDict):
    input_text: str
    
    context: Dict

    title: str
    tags: List[str]
    tldr: str

    review: Dict
    feedback: Dict