# Elyse AI Workshop Application

## Overview

Elyse is an advanced AI companion application built for a workshop focused on LlamaIndex and modern LLM practices. It features a modular architecture, a multi-step workflow for orchestrating LLM interactions, and a command-line interface for chat. The project emphasizes dynamic prompt engineering, concurrent calls to multiple LLM providers via LiteLLM, and human-in-the-loop response curation.

This application serves as a practical example for learning how to build sophisticated AI chat systems with LlamaIndex, from basic workflow construction to more advanced concepts like session management, API integration, and RAG implementation.

## Core Technologies

*   **Python**: The primary programming language.
*   **LlamaIndex**: The core framework for orchestrating LLM workflows, data indexing, and retrieval.
*   **LiteLLM**: Used for making calls to a wide variety of LLM providers (OpenAI, Gemini, Anthropic, OpenRouter, Groq, etc.) with a unified interface.
*   **Rich**: For creating beautiful and informative command-line interfaces.
*   **uv**: For Python packaging and virtual environment management (as a fast alternative to pip/venv).
*   **Pydantic**: For data validation and settings management.
*   **Phoenix (Aryze)**: For observability and tracing of LlamaIndex applications (optional).
*   **FastAPI** (Planned): For creating an API to interact with a frontend.
*   **Expo.js** (Planned): For building the frontend client application.

## Setup and Installation

### Prerequisites

*   Python 3.10 or higher.
*   `uv` installed (see [uv installation guide](https://astral.sh/docs/uv/install.sh) or `pip install uv`).

### 1. Clone the Repository

```bash
git clone <your_repository_url> # Replace with your actual repo URL
cd elyse_llama
```

### 2. Create and Activate Virtual Environment

It's highly recommended to use a virtual environment.

```bash
# Create a virtual environment using uv (e.g., named .venv)
python -m venv .venv  # Or use: uv venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows (Git Bash or similar):
# source .venv/Scripts/activate
# On Windows (Command Prompt/PowerShell):
# .venv\Scripts\activate
```

### 3. Install Dependencies

Dependencies will be managed via a `requirements.txt` file or `pyproject.toml` (if it's set up for project dependencies with `uv sync`). 

*If a `requirements.txt` is provided (we should generate this based on current imports):*
```bash
uv pip install -r requirements.txt
```

*If using `pyproject.toml` for dependencies (more robust for projects):*
Ensure your `pyproject.toml` lists dependencies under `[project.dependencies]`. Example:
```toml
# In pyproject.toml
[project]
name = "elyse_ai_workshop"
version = "0.1.0"
dependencies = [
    "llama-index-core",
    "llama-index-llms-litellm", 
    "litellm",
    "python-dotenv",
    "rich",
    "openinference-instrumentation-llama-index",
    "opentelemetry-exporter-otlp-proto-http", # For Phoenix
    "fastapi", # Planned
    "uvicorn[standard]" # Planned, for FastAPI
]
```
Then run:
```bash
uv sync # If using pyproject.toml dependencies
```
*(We will add a step to generate `requirements.txt` or ensure `pyproject.toml` is set up correctly for this.)*

### 4. Environment Variables

Create a `.env` file in the project root directory (`elyse_llama/.env`) to store your API keys and other sensitive configurations.

Example `.env` file:

```env
# OpenAI API Key (Required if using OpenAI models)
OPENAI_API_KEY="sk-your-openai-api-key"

# Gemini API Key (Required if using Google Gemini models directly via LiteLLM)
# GEMINI_API_KEY="your-gemini-api-key"

# Anthropic API Key (Required if using Anthropic Claude models directly via LiteLLM)
# ANTHROPIC_API_KEY="your-anthropic-api-key"

# Groq API Key (Required if using Groq models)
# GROQ_API_KEY="your-groq-api-key"

# OpenRouter API Key (Required if using models via OpenRouter)
# OPENROUTER_API_KEY="your-openrouter-api-key"

# LiteLLM specific configurations (optional)
# LITELLM_LOG_LEVEL="DEBUG" # For verbose LiteLLM logging

# Phoenix Tracing (Optional)
# PHOENIX_TRACE_ENABLED="true"
# PHOENIX_ENDPOINT_URL="http://127.0.0.1:6006/v1/traces"
```

**Important:** Ensure you have the necessary API keys for the LLM providers you intend to use, as configured in `backend/cli.py` (the `default_models_to_use` list) or as selected in the future UI.

## Running the Application

Once the setup is complete:

1.  Ensure your virtual environment is activated.
2.  Navigate to the project root directory (`elyse_llama`).
3.  Run the command-line interface:

    ```bash
    uv run -m backend.cli
    ```

## Project Plan

*(The project plan we discussed earlier will be inserted here. For brevity, I'm not repeating it in this tool call, but it would be part of the README content.)*

### Phase 1: CLI Enhancements & Core Polish
*   [x] **Task 1.1: CLI Output & Formatting**
    *   [x] Task 1.1.1: Differentiate logs
    *   [x] Task 1.1.2: Structure LLM candidate responses (Rich Panel)
    *   [x] Task 1.1.3: Improve spacing and readability
    *   [x] Task 1.1.4: Basic Markdown parsing for final AI response
    *   [x] Task 1.1.5: Review and trim redundant output
*   [x] **Task 1.2: Docstrings & Code Comments**
    *   [x] `backend/app_models.py`
    *   [x] `backend/services/llm_service.py`
    *   [x] `backend/prompts/prompt_manager.py`
    *   [x] `backend/workflow.py`
    *   [x] `backend/cli.py`
*   [ ] **Task 1.3: Initial README.md** (This task)

### Phase 2: Persistence & Session Management
*   [ ] **Task 2.1: Chat Session Persistence**
    *   [ ] Design JSON structure for sessions.
    *   [ ] Implement save session functionality.
    *   [ ] Implement list available sessions.
    *   [ ] Implement load session functionality.
    *   [ ] Address restarting chat from a specific point (new thread from selection).

### Phase 3: API Development (for Frontend Interaction)
*   [x] **Task 3.1: FastAPI Setup**
    *   [x] Add FastAPI dependency.
    *   [x] Create `backend/main.py` for FastAPI app.
    *   [x] Design API endpoints (e.g., `/chat/invoke`).
    *   [x] Adapt workflow for API calls.
    *   [x] Implement basic chat session loading/saving per API call.
*   [ ] **Task 3.2: Server-Sent Events (SSE) for Streaming**
    *   [x] Task 3.2.1: Define Streamable Event Models (`WorkflowStepUpdateEvent`, `LLMTokenStreamEvent`, `LLMCandidateReadyEvent`, `CurationRequiredEvent`, `WorkflowErrorEvent` in `app_models.py`).
    *   [x] Task 3.2.2: Modify workflow steps (`workflow.py`) and `llm_service.py` to emit these events using `ctx.write_event_to_stream()`.
    *   [x] Task 3.2.3: Implement FastAPI SSE endpoint (`/chat/stream` in `main.py`) to run the workflow with `astream_events()` and send events to the client.
    *   [ ] **Task 3.2.a: Implement Asynchronous Curation using `asyncio.Future`** (Handles pausing workflow for UI-based curation)
        *   [ ] Update Pydantic Models (`app_models.py`):
            *   Add `workflow_run_id` to `WorkflowStartEvent` (optional) and `GlobalContext` (optional).
            *   Make `workflow_run_id` required in `CurationRequiredEvent`.
        *   [ ] Modify Workflow (`workflow.py` - `AppWorkFlow`):
            *   Initialize `self.active_futures: Dict[str, asyncio.Future]` in `__init__`.
            *   `process_input` step: Propagate `workflow_run_id` from `WorkflowStartEvent` to `GlobalContext`.
            *   `curation_manager` step:
                *   Retrieve `workflow_run_id` from context.
                *   Create and store `asyncio.Future()` in `self.active_futures` keyed by `workflow_run_id`.
                *   Emit `CurationRequiredEvent` (including `workflow_run_id`).
                *   Remove console input/Rich Panel; `await future` instead.
                *   Get result from `future.result()`.
                *   Use `try...finally` to clean up future from `self.active_futures`.
        *   [ ] Modify API (`main.py`):
            *   Import `uuid`.
            *   `/chat/stream` endpoint:
                *   Generate unique `workflow_run_id` per request.
                *   Pass `workflow_run_id` in `WorkflowStartEvent`.
                *   Implement robust cleanup for `active_futures` on client disconnect/stream cancellation (cancel the future).
            *   Create `CuratedResponseRequest` model (fields: `workflow_run_id`, `curated_response`).
            *   Create `/chat/curate` (POST) endpoint:
                *   Accepts `CuratedResponseRequest`.
                *   Looks up future in `workflow.active_futures`.
                *   If future exists and not done, use `loop.call_soon_threadsafe(future.set_result, ...)` to set its result.
                *   Remove future from `active_futures` after setting result.
                *   Return appropriate success/error responses.

### Phase 4: Frontend Integration (Expo.js)
*   [ ] **Task 4.1: Basic Frontend Client**
    *   [ ] Setup Astro.js project.
    *   [ ] Implement basic chat interface (send/receive, display candidates, curation).
*   [ ] **Task 4.2: Model Selection from `model_lib.yml`**
    *   [ ] Parse `model_lib.yml`.
    *   [ ] API endpoint to fetch models.
    *   [ ] Frontend UI for model selection.
*   [ ] **Task 4.3: App Settings Adjustment in UI**
    *   [ ] API endpoint for settings.
    *   [ ] Frontend UI for settings.

### Future Phases (Details TBD)
*   [ ] **Task 5: Advanced Memory System**
*   [ ] **Task 6: Retrieval/RAG Agent**
*   [ ] **Task 7: Advanced Conversation Features**