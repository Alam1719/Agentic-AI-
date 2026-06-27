import streamlit as st
from graph import chatbot



st.set_page_config(page_title="LangGraph Chatbot", page_icon="🤖")

st.title("🤖 Rule-Based Chatbot (LangGraph + Streamlit)")



if "messages" not in st.session_state:
    st.session_state.messages = []



for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["text"])



user_input = st.chat_input("Type your message...")


if user_input:

  
    st.chat_message("user").write(user_input)

    
    result = chatbot.invoke({
        "user_input": user_input,
        "clean_input": "",
        "response": ""
    })

    bot_reply = result["response"]

    
    st.chat_message("assistant").write(bot_reply)

    
    st.session_state.messages.append({"role": "user", "text": user_input})
    st.session_state.messages.append({"role": "assistant", "text": bot_reply})