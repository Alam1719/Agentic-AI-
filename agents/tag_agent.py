from agents.tag_extractor_graph import tag_extractor_subgraph

def tag_agent(state):
    print("\n   [Tag Agent] Firing Nested Tag Extraction Sub-Graph...")
    
    # Prepare the local state required by the tag extractor subgraph
    subgraph_input = {
        "text": state.get("input_text", ""),
        "context": state.get("context", {}),
        "feedback": state.get("feedback", {}).get("tags", "")
    }
    
    # Invoke the highly parallel subgraph
    subgraph_result = tag_extractor_subgraph.invoke(subgraph_input)
    
    # Pass the synthesised final tags back to the main state
    final_tags = subgraph_result.get("final_tags", [])
    
    return {"tags": final_tags}