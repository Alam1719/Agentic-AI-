from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    temperature=0.3,
    model="llama-3.3-70b-versatile"
)

def tldr_agent(state):
    text = state["input_text"]
    context = state["context"]

    feedback = state.get("feedback", {})

    extra_instruction = ""
    if "tldr" in feedback:
        extra_instruction = f"Improve based on feedback: {feedback['tldr']}"

    prompt = f"""
    Write a concise TL;DR (2-3 lines) for this project.

    Context:
    {context}

    Text:
    {text}

    {extra_instruction}
    """

    response = llm.invoke(prompt)

    return {"tldr": response.content.strip()}