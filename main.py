from langgraph.graph import StateGraph, START, END
from state import AgentState

from agents.manager_agent import manager_agent
from agents.title_agent import title_agent
from agents.tag_agent import tag_agent
from agents.tldr_agent import tldr_agent
from agents.reviewer_agent import reviewer_agent

def route_after_review(state):
    review = state.get("review", {})
    if review.get("status") == "fail":
        # Route back to all workers if it fails.
        return ["title_agent", "tag_agent", "tldr_agent"]
    return END

# Build the graph
builder = StateGraph(AgentState)

builder.add_node("manager_agent", manager_agent)
builder.add_node("title_agent", title_agent)
builder.add_node("tag_agent", tag_agent)
builder.add_node("tldr_agent", tldr_agent)
builder.add_node("reviewer_agent", reviewer_agent)

# Outline sequence
builder.add_edge(START, "manager_agent")

# Parallel branch from Manager to Workers
builder.add_edge("manager_agent", "title_agent")
builder.add_edge("manager_agent", "tag_agent")
builder.add_edge("manager_agent", "tldr_agent")

# Fan-in to Reviewer
builder.add_edge(["title_agent", "tag_agent", "tldr_agent"], "reviewer_agent")

# Conditional loop
builder.add_conditional_edges("reviewer_agent", route_after_review)

graph = builder.compile()

if __name__ == "__main__":
    print("\n🚀 LangGraph Multi-Agent System Initialized.")
    
    user_input = input("\n📝 Enter the text or project description you want the agents to process:\n> ")
    
    if not user_input.strip():
        print("No input provided. Exiting.")
        exit()

    print("\n⚙️  Agents are now processing your input... Please wait.")
    
    initial_state = {
        "input_text": user_input,
        "feedback": {}
    }

    try:
        final_state = graph.invoke(initial_state)
        
        print("\n✅ FINAL EXECUTED STATE:\n")
        print("TITLE:", final_state.get("title"))
        print("TAGS:", final_state.get("tags"))
        print("TL;DR:", final_state.get("tldr"))
        print("\nFINAL REVIEW STATUS:", final_state.get("review"))
        
    except Exception as e:
        print("Execution failed:", e)