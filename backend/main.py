import asyncio
import json  # For serializing events to JSON for SSE
import os
import uuid  # Added for generating unique workflow run IDs
from typing import AsyncGenerator, List, Optional  # Added AsyncGenerator

# LlamaIndex Core Imports for Debugging
import llama_index.core
from dotenv import load_dotenv
from fastapi import (  # Added Request for client disconnect handling
    FastAPI,
    HTTPException,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware  # To allow frontend calls
from fastapi.responses import StreamingResponse  # Added for SSE
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler

# ----
from llama_index.core.llms import ChatMessage
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.workflow import StopEvent  # Added StopEvent import
from pydantic import BaseModel as PydanticBaseModel  # For request/response models
from rich.console import Console  # Added for styled printing

from backend.app_models import (
    CurationRequiredEvent,
    LLMCandidateReadyEvent,
    LLMTokenStreamEvent,
    ModelSettings,
    WorkflowErrorEvent,
    WorkflowRunOutput,
    WorkflowStartEvent,
    # Import event types that can be streamed if specific handling is needed, though generic Event is often enough
    WorkflowStepUpdateEvent,
)
from backend.workflow import AppWorkFlow

"""
Provides the FastAPI application server for the Elyse AI backend.

This module defines the API endpoints that client applications (e.g., a frontend UI)
use to interact with the AI workflow. Key functionalities include:
- Starting new chat sessions or continuing existing ones.
- Sending user messages and receiving AI-generated responses.
- Streaming workflow events (like LLM tokens, step updates, and curation requests)
  over Server-Sent Events (SSE) for real-time updates.
- Handling human-in-the-loop curation of AI responses.
- Managing chat history persistence.

The server utilizes LlamaIndex for the core workflow logic and FastAPI for the
web framework. It's configured for CORS to allow cross-origin requests from
frontends.
"""

load_dotenv()

# --- LlamaIndex Global Debugger Setup ---
# This should be one of the first things to run
llama_debug = LlamaDebugHandler()
callback_manager = CallbackManager([llama_debug])
llama_index.core.Settings.callback_manager = callback_manager
print("--- LlamaDebugHandler and global CallbackManager initialized. --- ")
# ---

# --- Rich Console ---
console = Console()  # Added for styled printing

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Elyse AI Workflow API",
    description="API for interacting with the Elyse AI chat workflow, powered by LlamaIndex and FastAPI. Provides endpoints for chat, streaming, and curation.",
    version="0.1.0",
)

# --- CORS Middleware ---
# Configure Cross-Origin Resource Sharing (CORS) to allow requests from your frontend.
# Replace "http://localhost:3000" with the actual origin of your frontend if different.
# For development, allowing all origins ("*") can be easier, but be more restrictive for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for now, adjust for production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# --- Chat Store Setup ---
CHAT_SESSIONS_DIR = "chat_sessions"
DEFAULT_SESSION_ID = "default_session"
# Ensure the directory exists at startup
os.makedirs(CHAT_SESSIONS_DIR, exist_ok=True)

# Global workflow instance (can be initialized per request if state becomes an issue)
# For now, a single instance is okay as workflow itself is mostly stateless per run,
# relying on context passed in via StartEvent and managed by SimpleChatStore.
workflow = AppWorkFlow(timeout=120, verbose=False)  # verbose=False for cleaner API logs
print(f"--- Workflow instance created. Type: {type(workflow)}")
# Print all attributes to see available methods like run, arun, etc.
print(f"--- Attributes of workflow instance: {str(dir(workflow))}")

# --- API Request and Response Models ---


class ChatInvokeRequest(PydanticBaseModel):
    """
    Request model for the `/chat/invoke` endpoint.

    Used for non-streaming, single-response chat interactions.
    """

    user_message: str
    session_id: Optional[str] = DEFAULT_SESSION_ID
    settings: Optional[ModelSettings] = None  # If None, workflow defaults will be used
    initial_models_to_use: Optional[List[str]] = (
        None  # If None, workflow defaults will be used
    )


class ChatStreamRequest(PydanticBaseModel):
    """
    Request model for the `/chat/stream` endpoint.

    Used for initiating a chat interaction where events are streamed back to the client.
    """

    user_message: str
    session_id: Optional[str] = DEFAULT_SESSION_ID
    settings: Optional[ModelSettings] = None
    initial_models_to_use: Optional[List[str]] = None


class CuratedResponseRequest(PydanticBaseModel):
    """
    Request model for the `/chat/curate` endpoint.

    Used to submit the human-selected or edited response for a workflow run
    that is awaiting curation.
    """

    workflow_run_id: str
    curated_response: str


class ChatInvokeResponse(PydanticBaseModel):
    """
    Response model for the `/chat/invoke` endpoint.

    Mirrors `WorkflowRunOutput` for consistency.
    """

    final_response: str
    chat_history: List[ChatMessage]
    session_id: str
    # Potentially add error messages or status codes here later


# --- API Endpoints ---


@app.post("/chat/invoke", response_model=ChatInvokeResponse)
async def chat_invoke(request: ChatInvokeRequest) -> ChatInvokeResponse:
    """
    Endpoint for a single, non-streaming chat interaction.

    Receives a user message, session ID (optional), and settings (optional).
    It runs the full AI workflow, waits for its completion (including any
    internal curation if configured differently or if it's a simplified path),
    persists the updated chat history, and returns the final AI response.

    This endpoint is suitable for clients that do not require real-time streaming
    of events or intermediate LLM outputs.

    Args:
        request: A `ChatInvokeRequest` Pydantic model containing the user's message,
                 session ID, and optional settings.

    Returns:
        A `ChatInvokeResponse` Pydantic model with the final AI response,
        the complete updated chat history, and the session ID.

    Raises:
        HTTPException: If there's an internal server error during workflow execution
                       or if the workflow returns an unexpected output type.
    """
    print(f"[CHAT INVOKE] Received request: {request.model_dump()}")
    session_id = request.session_id or DEFAULT_SESSION_ID
    chat_store_path = os.path.join(CHAT_SESSIONS_DIR, f"{session_id}_store.json")

    # Load chat store for the session
    try:
        chat_store = SimpleChatStore.from_persist_path(persist_path=chat_store_path)
        current_chat_history = chat_store.get_messages(session_id)
    except FileNotFoundError:
        console.print(
            f"[CHAT INVOKE] No chat history found for session {session_id} at {chat_store_path}, starting new."
        )
        chat_store = SimpleChatStore()  # Initialize a new store
        current_chat_history = []

    # Prepare settings and models to use
    active_model_settings = request.settings or ModelSettings()
    active_models_to_use = (
        request.initial_models_to_use
    )  # Can be None, workflow handles default

    start_event = WorkflowStartEvent(
        user_message=request.user_message,
        settings=active_model_settings,
        initial_models_to_use=active_models_to_use,
        chat_history=current_chat_history,
        # workflow_run_id is not strictly needed here unless invoke also supported async curation later
    )

    try:
        # Note: The AppWorkFlow.run method is synchronous. If it internally awaits
        # a future (like in curation_manager), this FastAPI path will block.
        # For /invoke, this might be acceptable if curation is fast or if this path
        # implies a workflow variant without long waits.
        # If curation is always async via API, /invoke might need a different workflow path
        # or a timeout mechanism for user input if it were to support it directly.
        # Current AppWorkFlow expects curation_manager to potentially wait on an external signal.
        # This means /invoke might hang if it hits curation_manager and no one calls /chat/curate.
        # For true non-streaming invoke, the workflow might need a different "curation_mode".
        # For now, we assume if /invoke is used, curation might be simpler or handled differently.
        # Or, more likely, /invoke is less used if full async curation is the primary mode.

        # Let's assume a synchronous run, which will indeed block if curation_manager waits on a future.
        # To make this truly synchronous without external curation, workflow would need modification.
        # Given current setup, this /invoke will block if it hits the future.wait in curation.
        # This implies /invoke is for workflows that resolve curation internally or don't require it.

        # Re-evaluating: workflow.run() returns a handler. We need to await the handler for the result.
        handler = workflow.run(start_event=start_event)
        workflow_run_output = await handler  # Await the final result from the handler

        if not isinstance(workflow_run_output, WorkflowRunOutput):
            raise HTTPException(
                status_code=500, detail="Workflow returned an unexpected output type."
            )

        # Update and persist chat store
        chat_store.set_messages(session_id, workflow_run_output.chat_history)
        chat_store.persist(persist_path=chat_store_path)
        console.print(
            f"[CHAT INVOKE] Chat history persisted for session {session_id} to {chat_store_path}"
        )

        return ChatInvokeResponse(
            final_response=workflow_run_output.final_response,
            chat_history=workflow_run_output.chat_history,
            session_id=session_id,
        )
    except Exception as e:
        console.print(
            f"Error during /chat/invoke for session {session_id}: {e}",
            style="bold red",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"An internal error occurred: {str(e)}"
        )


# --- SSE Chat Endpoint ---
@app.post("/chat/stream")
async def chat_stream(
    api_request: ChatStreamRequest, httpRequest: Request
):  # Added httpRequest for disconnect
    """
    Streaming endpoint for chat interactions using Server-Sent Events (SSE).

    This endpoint initiates a chat workflow run. It receives user input, loads
    relevant chat history, and then starts the `AppWorkFlow`. As the workflow
    progresses, it generates various events (e.g., `WorkflowStepUpdateEvent`,
    `LLMTokenStreamEvent`, `LLMCandidateReadyEvent`, `CurationRequiredEvent`).
    These events are streamed back to the client in real-time, formatted
    according to the Vercel AI SDK Data Stream specification.

    The `workflow_run_id` generated for this interaction is crucial for the
    `CurationRequiredEvent`. When this event is emitted, the client (or user)
    is expected to make a separate call to the `/chat/curate` endpoint, providing
    this `workflow_run_id` and the chosen response, to allow the workflow to continue.

    Handles client disconnects by attempting to cancel the ongoing workflow's
    curation future.

    Args:
        api_request: A `ChatStreamRequest` Pydantic model containing the user's message,
                     session ID, and optional settings.
        httpRequest: The FastAPI `Request` object, used to detect client disconnects.

    Returns:
        A `StreamingResponse` that sends events to the client.
    """
    print(
        f"[/chat/stream] Received API request: {api_request.model_dump_json(indent=2)}"
    )
    session_id = api_request.session_id or DEFAULT_SESSION_ID
    chat_store_path = os.path.join(CHAT_SESSIONS_DIR, f"{session_id}_store.json")
    workflow_run_id = str(
        uuid.uuid4()
    )  # Unique ID for this specific workflow interaction
    print(f"[/chat/stream] Generated workflow_run_id: {workflow_run_id}")

    try:
        chat_store = SimpleChatStore.from_persist_path(persist_path=chat_store_path)
        raw_chat_history = chat_store.get_messages(session_id)
    except FileNotFoundError:
        console.print(
            f"[/chat/stream] No chat history found for session {session_id} at {chat_store_path}, starting new."
        )
        chat_store = SimpleChatStore()  # Initialize a new store
        raw_chat_history = []

    print(
        f"[/chat/stream] Loaded {len(raw_chat_history)} raw messages from chat history for session {session_id}"
    )

    # Normalize chat history: Ensure all messages are simple ChatMessage objects
    # without complex nested structures like 'blocks' that might cause issues
    # with Pydantic validation or LlamaIndex components expecting simpler ChatMessage.
    normalized_chat_history: List[ChatMessage] = []
    for msg in raw_chat_history:
        content = ""
        if hasattr(msg, "blocks") and msg.blocks and isinstance(msg.blocks, list):
            # Extract text from the first text block if present
            for block in msg.blocks:
                if (
                    hasattr(block, "block_type")
                    and block.block_type == "text"
                    and hasattr(block, "text")
                ):
                    content = block.text
                    break  # Use first text block
        elif hasattr(msg, "content") and msg.content is not None:
            content = str(msg.content)
        else:
            content = "[Non-text content or unable to parse]"

        normalized_chat_history.append(
            ChatMessage(role=msg.role, content=content, additional_kwargs={})
        )

    print(
        f"[/chat/stream] Normalized chat history ({len(normalized_chat_history)} messages) for event construction."
    )
    # Verbose logging of normalized history can be added here if needed for debugging

    active_model_settings = api_request.settings or ModelSettings()
    active_models_to_use = api_request.initial_models_to_use

    start_event = WorkflowStartEvent(
        user_message=api_request.user_message,
        settings=active_model_settings,
        initial_models_to_use=active_models_to_use,
        chat_history=normalized_chat_history,
        workflow_run_id=workflow_run_id,  # Pass the unique ID to the workflow
    )

    print(
        f"[/chat/stream] Logging WorkflowStartEvent (via model_dump_json) to be passed to workflow: {start_event.model_dump_json(indent=2)}"
    )

    async def event_generator() -> AsyncGenerator[str, None]:
        """
        Async generator that runs the workflow and yields SSE-formatted events.
        """
        print(f"[SSE WORKFLOW_RUN_ID: {workflow_run_id}] Event generator started.")
        handler = None  # WorkflowHandler
        try:
            print(f"[SSE WORKFLOW_RUN_ID: {workflow_run_id}] Calling workflow.run()...")
            # workflow.run() is synchronous but returns a handler quickly.
            # The actual async work happens when we iterate handler.stream_events().
            handler = workflow.run(start_event=start_event)
            print(
                f"[SSE WORKFLOW_RUN_ID: {workflow_run_id}] workflow.run() returned handler: {type(handler)}. Iterating handler.stream_events()..."
            )

            async for event in handler.stream_events():
                print(
                    f"[SSE WORKFLOW_RUN_ID: {workflow_run_id}] Received event from handler.stream_events(): {type(event).__name__}"
                )
                if await httpRequest.is_disconnected():
                    print(
                        f"[SSE WORKFLOW_RUN_ID: {workflow_run_id}] Client disconnected detected."
                    )
                    future_to_cancel = workflow.active_futures.pop(
                        workflow_run_id, None
                    )
                    if future_to_cancel and not future_to_cancel.done():
                        loop = future_to_cancel.get_loop()
                        loop.call_soon_threadsafe(future_to_cancel.cancel)
                        print(
                            f"[SSE WORKFLOW_RUN_ID: {workflow_run_id}] Curation future cancelled due to client disconnect."
                        )
                    else:
                        print(
                            f"[SSE WORKFLOW_RUN_ID: {workflow_run_id}] Client disconnected, no active future or future already done for this run_id."
                        )
                    print(
                        f"[SSE WORKFLOW_RUN_ID: {workflow_run_id}] Breaking stream due to client disconnect."
                    )
                    break  # Exit the event streaming loop

                event_payload_str = None
                # Vercel AI SDK Data Stream Format:
                # Type 0: Text delta (e.g., LLM token stream)
                # Type 1: Error
                # Type 2: JSON data (custom events like step updates, candidate ready, curation required)
                # Type 'd': Metadata (sent with finishReason when stream stops)

                if isinstance(event, LLMTokenStreamEvent):
                    event_payload_str = f"0:{json.dumps(event.token)}\\n"
                elif isinstance(event, WorkflowStepUpdateEvent):
                    payload = {"type": "workflow_step_update", **event.model_dump()}
                    event_payload_str = f"2:{json.dumps(payload)}\\n"
                elif isinstance(event, LLMCandidateReadyEvent):
                    payload = {"type": "llm_candidate_ready", **event.model_dump()}
                    event_payload_str = f"2:{json.dumps(payload)}\\n"
                elif isinstance(event, CurationRequiredEvent):
                    payload = {"type": "curation_required", **event.model_dump()}
                    event_payload_str = f"2:{json.dumps(payload)}\\n"
                elif isinstance(event, WorkflowErrorEvent):
                    payload = {"type": "workflow_error", **event.model_dump()}
                    event_payload_str = f"1:{json.dumps(payload)}\\n"
                elif isinstance(event, StopEvent):
                    workflow_run_output = event.result
                    if isinstance(workflow_run_output, WorkflowRunOutput):
                        # Send final response text as Type 0
                        yield f"0:{json.dumps(workflow_run_output.final_response)}\\n"
                        # Persist chat history upon successful completion before sending metadata
                        try:
                            chat_store.set_messages(
                                session_id, workflow_run_output.chat_history
                            )
                            chat_store.persist(persist_path=chat_store_path)
                            console.print(
                                f"[SSE WORKFLOW_RUN_ID: {workflow_run_id}] Chat history persisted for session {session_id} to {chat_store_path} after StopEvent."
                            )
                        except Exception as e_persist:
                            console.print(
                                f"[SSE WORKFLOW_RUN_ID: {workflow_run_id}] Error persisting chat history for session {session_id}: {e_persist}",
                                style="bold red",
                                exc_info=True,
                            )
                            # Decide if this should be a streamed error or just server log

                        chat_history_serializable = [
                            msg.model_dump() for msg in workflow_run_output.chat_history
                        ]
                        metadata_payload = {
                            "finishReason": "stop",
                            "chatHistory": chat_history_serializable,
                        }
                        yield f"d:{json.dumps(metadata_payload)}\\n"
                    elif (
                        isinstance(workflow_run_output, dict)
                        and "error" in workflow_run_output  # from on_error
                    ):
                        error_payload = {
                            "type": "workflow_error",
                            "step_name": workflow_run_output.get(
                                "failed_step", "stop_event_error"
                            ),
                            "error_message": str(workflow_run_output["error"]),
                        }
                        yield f"1:{json.dumps(error_payload)}\\n"
                    else:  # Unexpected result from StopEvent
                        error_payload = {
                            "type": "workflow_error",
                            "step_name": "stop_event",
                            "error_message": "Workflow completed with an unexpected result type in StopEvent.",
                        }
                        yield f"1:{json.dumps(error_payload)}\\n"
                    print(
                        f"[SSE WORKFLOW_RUN_ID: {workflow_run_id}] StopEvent received and processed, breaking stream."
                    )
                    break  # Exit the event streaming loop

                if event_payload_str:
                    yield event_payload_str

        except asyncio.CancelledError:
            # This can happen if the workflow itself is cancelled, e.g., by timeout,
            # or if the future in curation_manager is cancelled and the error propagates.
            print(
                f"[SSE WORKFLOW_RUN_ID: {workflow_run_id}] Event generator's main task or workflow run was cancelled (asyncio.CancelledError)."
            )
            # Send an error message to the client
            error_payload_model = WorkflowErrorEvent(
                step_name="workflow_execution_cancelled",
                error_message="The workflow execution was cancelled, possibly due to timeout or client disconnect during curation.",
                workflow_run_id=workflow_run_id,
            )
            error_data = {"type": "workflow_error", **error_payload_model.model_dump()}
            yield f"1:{json.dumps(error_data)}\\n"
        except Exception as e:
            console.print(
                f"[SSE WORKFLOW_RUN_ID: {workflow_run_id}] Error in event_generator: {type(e).__name__} - {e}",
                style="bold red",
                exc_info=True,
            )
            error_payload_model = WorkflowErrorEvent(
                step_name="event_generator_error",
                error_message=str(e),
                workflow_run_id=workflow_run_id,
            )
            error_data = {"type": "workflow_error", **error_payload_model.model_dump()}
            yield f"1:{json.dumps(error_data)}\\n"
        finally:
            print(
                f"[SSE WORKFLOW_RUN_ID: {workflow_run_id}] Event generator finishing."
            )
            # Ensure the future is removed if it's still there and the workflow was handling it
            # (e.g. if an error occurred before curation_manager's finally block)
            if workflow_run_id in workflow.active_futures:
                lingering_future = workflow.active_futures.pop(workflow_run_id, None)
                if lingering_future and not lingering_future.done():
                    try:
                        loop = lingering_future.get_loop()
                        loop.call_soon_threadsafe(lingering_future.cancel)
                        print(
                            f"[SSE WORKFLOW_RUN_ID: {workflow_run_id}] Lingering curation future cancelled in event_generator finally block."
                        )
                    except Exception as e_cancel_linger:
                        console.print(
                            f"[SSE WORKFLOW_RUN_ID: {workflow_run_id}] Error cancelling lingering future in finally: {e_cancel_linger}",
                            style="yellow",
                        )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        # Headers for Vercel AI SDK compatibility
        headers={"X-Vercel-AI-Data-Stream": "v1"},
    )


@app.post("/chat/curate")
async def chat_curate(curation_request: CuratedResponseRequest):
    """
    Endpoint for submitting a human-curated AI response.

    When the `/chat/stream` endpoint emits a `CurationRequiredEvent`, it includes
    a `workflow_run_id`. The client should use this ID to call this `/chat/curate`
    endpoint, providing the selected or edited AI response. This action resolves
    an `asyncio.Future` that the `curation_manager` step in the `AppWorkFlow`
    is awaiting, allowing the workflow to proceed.

    Args:
        curation_request: A `CuratedResponseRequest` Pydantic model containing
                          the `workflow_run_id` and the `curated_response` string.

    Returns:
        A JSON object confirming success or an error.

    Raises:
        HTTPException:
            - 404 if the `workflow_run_id` is not found or already processed.
            - 409 if curation for the `workflow_run_id` has already been submitted or cancelled.
            - 500 if there's an internal error processing the curation.
    """
    run_id = curation_request.workflow_run_id
    curated_text = curation_request.curated_response

    console.print(
        f'[API /chat/curate] Received curation for run_id: {run_id}, response: "{curated_text[:70]}..."'
    )

    future = workflow.active_futures.get(run_id)

    if not future:
        console.print(
            f"[API /chat/curate] Error: No active future found for run_id: {run_id}",
            style="yellow",
        )
        raise HTTPException(
            status_code=404,
            detail=f"Workflow run ID {run_id} not found, already processed, or timed out.",
        )

    if future.done():
        console.print(
            f"[API /chat/curate] Warning: Future for run_id: {run_id} is already done.",
            style="yellow",
        )
        # Attempt to remove it if it wasn't by the workflow's finally block (e.g. client sent twice)
        workflow.active_futures.pop(run_id, None)
        raise HTTPException(
            status_code=409,  # Conflict
            detail=f"Curation for workflow run ID {run_id} has already been submitted or the run was cancelled/timed out.",
        )

    try:
        loop = future.get_loop()
        loop.call_soon_threadsafe(future.set_result, curated_text)
        console.print(
            f"[API /chat/curate] Successfully set future result for run_id: {run_id}"
        )
        # The future is removed by the curation_manager step's finally block upon successful completion.
        # No need to pop it here if future.set_result() is successful.
        return {
            "status": "success",
            "message": f"Curation received for workflow run {run_id}.",
        }
    except Exception as e:
        console.print(
            f"[API /chat/curate] Error setting future result for run_id {run_id}: {e}",
            style="bold red",
            exc_info=True,
        )
        # If setting result fails, try to set an exception on the future so the workflow doesn't hang.
        if not future.done():
            try:
                loop = future.get_loop()  # Should be the same loop
                loop.call_soon_threadsafe(
                    future.set_exception,
                    RuntimeError(f"Failed to process curation via API: {e}"),
                )
            except Exception as fut_e:
                console.print(
                    f"[API /chat/curate] Further error trying to set exception on future for {run_id}: {fut_e}",
                    style="bold red",
                )
        # Even if setting exception fails, remove from active_futures to prevent leaks if possible.
        workflow.active_futures.pop(run_id, None)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process curation for workflow run {run_id}: {str(e)}",
        )


@app.get("/")
async def read_root():
    """Simple root endpoint to confirm the API is running and accessible."""
    return {
        "message": "Welcome to the Elyse AI Workflow API! Visit /docs for API documentation."
    }


# --- To run this FastAPI app (from the project root directory 'elyse_llama'): ---
# uvicorn backend.main:app --reload
#
# Then access the API at http://127.0.0.1:8000
# And interactive documentation at http://127.0.0.1:8000/docs
