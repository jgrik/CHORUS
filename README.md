# CHORUS
Multi Model AI Safety Verification

# Chorus

A tool for comparing how different AI models handle content safety decisions.

## What it does

Chorus sends the same prompt to Claude, GPT-5, and Llama simultaneously, then compares their safety classifications. The goal was to see where these models disagree on what's "safe" vs "unsafe" content.

## Why I built this

I wanted to understand how different AI companies approach content moderation. Turns out they have pretty different philosophies:

- **Claude** tends to allow educational/research content even if it could be misused
- **Llama** is more conservative, prioritizes preventing harm over information access  
- **GPT-5** sits somewhere in the middle, context-dependent

They agreed about 73% of the time. The disagreements were interesting. Usually, around dual-use information (e.g., "how to extract nicotine from tobacco"; chemistry question or substance abuse enabler?).

## How it works

**Backend:**
- FastAPI handles the web server
- Sends requests to all three APIs in parallel (async)
- SQLite stores the results and tracks disagreement patterns

**Frontend:**
- Gradio web interface for testing
- Real-time comparison of all three models
- Shows consensus verdict and which models flagged concerns

**APIs used:**
- Anthropic (Claude Sonnet 4)
- OpenAI (GPT-5)
- Together AI (Llama 3.1 70B)

## Setup

You'll need API keys for all three services. Set them in a `.env` file:
```
ANTHROPIC_API_KEY=your_key
OPENAI_API_KEY=your_key  
TOGETHERAI_API_KEY=your_key
```

Install dependencies:
```bash
pip install fastapi anthropic openai together-python python-dotenv gradio
```

Run the API:
```bash
uvicorn main:app --reload
```

Run the UI:
```bash
python ui.py
```

## Files

- `main.py` - FastAPI backend, API integrations, consensus logic
- `database.py` - SQLite operations and queries
- `ui.py` - Gradio web interface

## What I learned

- How to integrate multiple AI APIs and handle async requests
- Different approaches to AI safety across providers
- Building a full-stack application with database + UI + backend
- The complexity of content moderation decisions (there's rarely a "correct" answer)

Built in Python over about 6 weeks while learning the language.
