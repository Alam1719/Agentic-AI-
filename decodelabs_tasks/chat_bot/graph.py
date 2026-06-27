from typing import TypedDict
from langgraph.graph import StateGraph, END



class ChatState(TypedDict):
    user_input: str
    clean_input: str
    response: str


def sanitize_node(state: ChatState):
    return {
        "user_input": state["user_input"],
        "clean_input": state["user_input"].lower().strip(),
        "response": ""
    }


def knowledge_node(state: ChatState):
    kb = {
        "hello": "Hi! How can I help you?",
        "hi": "Hello! Ask me anything.",
        "who are you": "I am a rule-based chatbot built using LangGraph.",
        "purpose": "I demonstrate deterministic flow.",
        "creator": "I was built by an AI Engineer intern."
    }

    return {
        **state,
        "response": kb.get(state["clean_input"], "No answer found.")
    }


def mismatch_node(state: ChatState):
    return {
        **state,
        "response": "I didn't understand that. Please try again."
    }


def bye_node(state: ChatState):
    return {
        **state,
        "response": "Goodbye! Take care :)"
    }



def router(state: ChatState):
    text = state["clean_input"]

    exit_words = {"bye", "exit", "quit", "allah hafiz"}
    kb_keys = {"hello", "hi", "who are you", "purpose", "creator"}

    if text in exit_words:
        return "bye"

    if text in kb_keys:
        return "knowledge"

    return "mismatch"



graph = StateGraph(ChatState)

graph.add_node("sanitize", sanitize_node)
graph.add_node("knowledge", knowledge_node)
graph.add_node("mismatch", mismatch_node)
graph.add_node("bye", bye_node)

graph.set_entry_point("sanitize")

graph.add_conditional_edges(
    "sanitize",
    router,
    {
        "bye": "bye",
        "knowledge": "knowledge",
        "mismatch": "mismatch"
    }
)

graph.add_edge("knowledge", END)
graph.add_edge("mismatch", END)
graph.add_edge("bye", END)

chatbot = graph.compile()