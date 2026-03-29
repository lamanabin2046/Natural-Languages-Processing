
# A7: MCP-Server, AI Agent, and External Tool Integration
**Course:** AT82.05 Natural Language Understanding  
**Student:** Nabin Gangtan Lama

---

## Overview

This assignment demonstrates the setup of an **MCP (Model Context Protocol) Server** using n8n, the deployment of an **AI Agent** that consumes MCP tools, and integration with external APIs including **Telegram** and **Google Calendar** — all connected into a fully automated end-to-end pipeline.

---

## Architecture

```
Telegram User
     │
     ▼
Telegram Trigger (n8n)
     │
     ▼
AI Agent Node
  ├── Groq Chat Model (LLM)
  ├── Simple Memory (context retention)
  └── MCP Client ──► MCP Server
                        ├── Calculator Tool
                        ├── Date & Time Tool
                        └── Code Tool (Text Formatter)
     │
     ▼
Google Calendar (Create/Read Events)
     │
     ▼
Telegram Send Message (response to user)
```

---

## Task 1: MCP Server & AI Agent Setup

### Task 1.1 — Server Deployment

- Deployed **n8n** locally using **Docker Desktop** on Windows
- Used **ngrok** to expose the local instance publicly over HTTPS

**Docker command used:**
```bash
docker run -it --rm --name n8n -p 5678:5678 \
  -e WEBHOOK_URL=https://daria-condensedoveroptimistically.ngrok-free.dev \
  -v n8n_data:/home/node/.n8n \
  docker.n8n.io/n8nio/n8n
```

**ngrok command (separate terminal):**
```bash
ngrok http 5678
```

The `WEBHOOK_URL` environment variable ensures all external webhooks (including Telegram Trigger) register using the public HTTPS URL instead of localhost.

---

### Task 1.2 — MCP Server Workflow

- Created an n8n workflow titled **"MCP Server"** (Published state)
- Workflow starts with an **MCP Server Trigger** node
- Connected to **3 internal tools:**
  - Calculator
  - Date & Time
  - Code Tool (Text Formatter)

---

### Task 1.3 — AI Agent Client Workflow

- Created an n8n workflow titled **"AI Agent A7"**
- **Trigger:** When Chat Message Received
- **AI Agent node** configured with:
  - **LLM:** Groq Chat Model (free API)
  - **Memory:** Simple Memory (conversation context)
  - **Tool:** MCP Client → connected to MCP Server Production URL

**Verification Tests:**

| User Input | Agent Response | Tool Used |
|---|---|---|
| "What is 25 x 48?" | "The result of 25 x 48 is 1200." | Calculator (via MCP) |
| "What is the current date and time?" | "2026-03-28T03:45:05.759-04:00" | Date & Time (via MCP) |

Both tests executed successfully in under 600ms.

---

## Task 2: External API Integration

### Task 2.1 — Telegram Bot Integration

- Created a Telegram bot: **@st125985_bot**
- n8n workflow listens via **Telegram Trigger** node
- AI Agent processes the message and responds via **Send a Text Message** node
- Full bidirectional communication verified

---

### Task 2.2 — Google Calendar Tool Integration

- Enabled **Google Calendar API** via Google Cloud Console
- Created **OAuth 2.0 Client ID** and added credentials to n8n
- Google Calendar node added as a sub-tool connected to the AI Agent

**Dynamic fields configured using `$fromAI()` expressions:**
```
Summary (Title): {{ $fromAI('Summary', ``, 'string') }}
Start Time:      {{ $fromAI('Start Time', ``, 'string') }}
End Time:        {{ $fromAI('End Time', ``, 'string') }}
```

This allows the AI Agent to populate calendar event fields directly from natural language user input.

---

### Task 2.3 — Automated Project Scheduling

Sent a single Telegram message to the bot requesting a 4-phase project schedule. The agent created all four Google Calendar events automatically:

| Phase | Event | Date | Time |
|---|---|---|---|
| 1st | Literature Review | March 30, 2026 | 09:00–10:00 |
| 2nd | Project Proposal | April 6, 2026 | 09:00–10:00 |
| 3rd | Update Progress | April 13, 2026 | 09:00–10:00 |
| 4th | Final Presentation | April 20, 2026 | 09:00–10:00 |

All events verified in Google Calendar.

---

### Task 2.4 — Interaction Verification

Queried the bot via Telegram:
> *"What project phases do I have on my Google Calendar? Please list them."*

The agent used the Google Calendar read tool and returned all four events correctly, confirming the full end-to-end pipeline works:

```
Telegram Input → AI Agent → Google Calendar (read/write) → Telegram Response
```

---

## Environment Variables

Sensitive credentials are stored in a `.env` file and never committed to version control.

```env
GROQ_API_KEY=your_groq_api_key_here
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
GOOGLE_CLIENT_ID=your_google_client_id_here
GOOGLE_CLIENT_SECRET=your_google_client_secret_here
```

> `.env` is listed in `.gitignore` — do not commit API keys.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| n8n | Workflow automation platform |
| Docker Desktop | Local n8n deployment |
| ngrok | Public HTTPS tunnel |
| Groq API | LLM (language model) |
| Telegram Bot API | User interface / chat trigger |
| Google Calendar API | Event creation & reading |
| MCP Protocol | Tool discovery & invocation |

---

## Setup Instructions

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
2. Install [ngrok](https://ngrok.com/) and authenticate
3. Run ngrok: `ngrok http 5678`
4. Copy the generated HTTPS URL
5. Start n8n with the Docker command above (replace `WEBHOOK_URL`)
6. Open n8n at `http://localhost:5678`
7. Import or recreate the two workflows: **MCP Server** and **AI Agent A7**
8. Configure credentials (Groq, Telegram, Google OAuth) in n8n settings
9. Activate both workflows

---

## .gitignore

```
.env
*.env
__pycache__/
*.pyc
```
