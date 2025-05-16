# Elyse AI Workshop Application

This project is a LlamaIndex-powered chat application built for a workshop, demonstrating a human-in-the-loop workflow with multi-LLM interaction, response curation, and a FastAPI backend for potential frontend integration.

## Architecture Overview

The application is structured into a backend a LlamaIndex workflow, FastAPI server, and supporting services/models.

**Core Components:**

*   **`backend/workflow.py` (`AppWorkFlow`):**
    *   The heart of the application, defining a LlamaIndex `Workflow`.
    *   Orchestrates a sequence of steps for each chat turn:
        1.  `process_input`: Handles user messages, settings, and chat history.
        2.  `retrieve_context`: (Simulated) Placeholder for future RAG capabilities.
        3.  `build_dynamic_prompt`: Constructs prompts using `backend/prompts/prompt_manager.py`.
        4.  `llm_manager`: Leverages `backend/services/llm_service.py` to call multiple LLMs concurrently.
        5.  `curation_manager`: Implements the human-in-the-loop step. This is where the async SSE magic for external input happens.
        6.  `stop_event`: Finalizes the turn, updates chat history, and packages the output.
    *   Uses a shared `Context` (`GlobalContext` from `backend/app_models.py`) for state management between steps.
    *   Emits various events (e.g., `WorkflowStepUpdateEvent`, `LLMTokenStreamEvent`, `CurationRequiredEvent`) for real-time client updates.

*   **`backend/main.py` (FastAPI Application):**
    *   Exposes API endpoints for interacting with the `AppWorkFlow`.
    *   `/chat/stream` (POST): The primary endpoint for chat interactions. It initiates a workflow run and streams events back to the client using Server-Sent Events (SSE). This endpoint generates a unique `workflow_run_id`.
    *   `/chat/curate` (POST): Allows an external client (or user via an API tool) to submit the curated AI response. This resolves an `asyncio.Future` that the `curation_manager` step in the workflow is awaiting for a specific `workflow_run_id`.
    *   `/chat/invoke` (POST): A non-streaming endpoint for simpler request-response interactions (less emphasized in the current SSE-focused design).
    *   Manages chat session persistence using `SimpleChatStore` (from LlamaIndex), saving conversations to JSON files in the `chat_sessions/` directory.

*   **`backend/services/llm_service.py` (`WorkflowLlmService`):**
    *   Handles the actual calls to LLMs using `LiteLLM`.
    *   Streams token-by-token responses (`LLMTokenStreamEvent`) and full candidate responses (`LLMCandidateReadyEvent`) back to the workflow, which then get relayed over SSE.

*   **`backend/app_models.py`:**
    *   Contains all Pydantic models for request/response bodies, workflow events (both internal and for SSE), settings, and context. This ensures data consistency and validation.

*   **`backend/cli.py`:**
    *   A command-line interface for direct interaction with the workflow, useful for testing and development. It simulates a chat session in the terminal.

**Data Flow & Asynchronous Curation (SSE):**

The interaction involving SSE and human-in-the-loop curation is a key aspect:

1.  **Client Request:** A client (e.g., Postman, or a future frontend) sends a POST request to `/chat/stream` with the user's message and session details.
2.  **Workflow Initiation:** The FastAPI server generates a unique `workflow_run_id` and starts an `AppWorkFlow` instance via `workflow.run(start_event=...)`. The `start_event` includes the `workflow_run_id`.
3.  **Event Streaming (SSE):**
    *   The `event_generator` in `backend/main.py` iterates through events produced by `handler.stream_events()`.
    *   As `AppWorkFlow` executes its steps (`process_input`, `retrieve_context`, `build_dynamic_prompt`, `llm_manager`), it uses `ctx.write_event_to_stream(...)` to emit custom events like `WorkflowStepUpdateEvent`, `LLMTokenStreamEvent` (from `WorkflowLlmService`), and `LLMCandidateReadyEvent`.
    *   These events are serialized to JSON and sent as SSE data chunks to the client, allowing for real-time UI updates (e.g., showing tokens as they arrive, displaying which step is active).
4.  **Curation Pause:**
    *   When the workflow reaches the `curation_manager` step:
        *   It retrieves the `workflow_run_id` from the context.
        *   It creates an `asyncio.Future` and stores it in `AppWorkFlow.active_futures` dictionary, keyed by the `workflow_run_id`.
        *   It emits a `CurationRequiredEvent` via `ctx.write_event_to_stream(...)`. This event, sent over SSE, includes the `workflow_run_id` and the list of AI response candidates.
        *   The `curation_manager` step then `await`s this `asyncio.Future`. This pauses the execution of this specific workflow run at this step.
5.  **External Curation Input:**
    *   The client, upon receiving the `CurationRequiredEvent`, displays the candidates to the user.
    *   The user makes a selection (or provides a custom response).
    *   The client then sends a POST request to the `/chat/curate` endpoint, including the *original* `workflow_run_id` and the `curated_response` text.
6.  **Workflow Resumption:**
    *   The `/chat/curate` endpoint in `backend/main.py` retrieves the corresponding `asyncio.Future` from `AppWorkFlow.active_futures` using the `workflow_run_id`.
    *   It calls `future.set_result(curated_text)`, which resolves the future.
    *   This awakens the `curation_manager` step in the paused workflow run.
7.  **Workflow Completion & Final Output:**
    *   The `curation_manager` step completes, passing the curated response to the `stop_event` step.
    *   The `stop_event` step finalizes the `WorkflowRunOutput` (containing the final curated response and updated chat history).
    *   This `WorkflowRunOutput` is the result of the `StopEvent`, which is then typically sent as a final message or metadata chunk over the SSE stream (e.g., prefixed with "0:" for final text or "d:" for data payload in Vercel AI SDK format).
    *   The chat history is persisted.
8.  **Client Disconnect Handling:**
    *   If the client disconnects from the `/chat/stream` endpoint while the workflow is awaiting curation, the `event_generator` detects this via `httpRequest.is_disconnected()`.
    *   It then attempts to find the `asyncio.Future` associated with the `workflow_run_id` and cancel it (`future.cancel()`). This causes the `await future` in `curation_manager` to raise an `asyncio.CancelledError`, allowing the workflow to handle the cancellation gracefully (e.g., log it, clean up, and terminate).

This asynchronous, event-driven mechanism allows the workflow to pause for external human input without blocking the FastAPI server, while providing real-time feedback to the client.

## Getting Started

1.  **Clone the repository.**
2.  **Set up a Python virtual environment.** (e.g., using `python -m venv .venv` and `source .venv/bin/activate`)
3.  **Install dependencies:**
    ```bash
    uv pip install -r requirements.txt 
    # Or if adding packages:
    # uv add <package_name>
    # uv sync 
    ```
    *(Ensure `requirements.txt` is created and up-to-date with packages like `fastapi`, `uvicorn`, `llama-index`, `python-dotenv`, `rich`, `litellm` etc.)*
4.  **Environment Variables:**
    *   Create a `.env` file in the project root.
    *   Add necessary API keys, e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`.
5.  **Run the FastAPI server:**
    ```bash
    uvicorn backend.main:app --reload --port 8000
    ```
    The API will be available at `http://127.0.0.1:8000` and docs at `http://127.0.0.1:8000/docs`.
6.  **Run the CLI (optional for testing):**
    ```bash
    uv run -m backend.cli
    ```

*(More sections on API endpoints, frontend integration, and further development can be added here.)*