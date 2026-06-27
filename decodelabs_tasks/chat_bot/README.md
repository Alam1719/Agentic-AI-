# 🤖 Rule-Based Chatbot

A deterministic chatbot built with LangGraph state machines and a Streamlit frontend.

## Overview

This chatbot uses a graph-based architecture to route user messages through sanitization, knowledge lookup, and response nodes. It demonstrates how LangGraph can be used to build structured, predictable conversational flows.

## Project Structure

```
chat_bot/
├── app.py               # Streamlit web interface
├── graph.py             # LangGraph state machine definition
├── requirements.txt     # Python dependencies
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

## How It Works

The chatbot follows a graph-based flow:

1. **Sanitize Node** — Normalizes user input (lowercase, strip whitespace)
2. **Router** — Routes to one of three paths:
   - `knowledge` — Matches known queries (hello, hi, who are you, etc.)
   - `bye` — Handles exit commands (bye, exit, quit)
   - `mismatch` — Handles unrecognized input
3. **Response Node** — Returns the appropriate response

## Supported Commands

| Input | Response |
|-------|----------|
| `hello` / `hi` | Greeting message |
| `who are you` | Bot identity |
| `purpose` | Bot's purpose |
| `creator` | Creator info |
| `bye` / `exit` / `quit` | Farewell message |
