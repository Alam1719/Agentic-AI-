from agents.title_generator_graph import title_generator_subgraph

def title_agent(state):
    print("\n   [Title Agent] Firing Nested Title Generation Sub-Graph...")
    
    # Prepare the local state required by the tag extractor subgraph
    subgraph_input = {
        "text": state.get("input_text", ""),
        "context": state.get("context", {}),
        "feedback": state.get("feedback", {}).get("title", "")
    }
    
    # Invoke the highly parallel subgraph
    subgraph_result = title_generator_subgraph.invoke(subgraph_input)
    
    # Pass the synthesised final tags back to the main state
    final_title = subgraph_result.get("final_title", "")
    
    return {"title": final_title}