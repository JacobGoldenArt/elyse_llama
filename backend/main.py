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
This module provides the FastAPI application server for the Elyse AI backend.

It exposes API endpoints that allow clients (e.g., a frontend application) 
to interact with the AI workflow, send messages, manage sessions, and receive responses.
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
    description="API for interacting with the Elyse AI chat workflow, powered by LlamaIndex and FastAPI.",
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
    """Request model for the /chat/invoke endpoint."""

    user_message: str
    session_id: Optional[str] = DEFAULT_SESSION_ID
    settings: Optional[ModelSettings] = None  # If None, workflow defaults will be used
    initial_models_to_use: Optional[List[str]] = (
        None  # If None, workflow defaults will be used
    )


# Define a similar request model for the streaming endpoint
class ChatStreamRequest(PydanticBaseModel):
    """Request model for the /chat/stream endpoint."""

    user_message: str
    session_id: Optional[str] = DEFAULT_SESSION_ID
    settings: Optional[ModelSettings] = None
    initial_models_to_use: Optional[List[str]] = None


class CuratedResponseRequest(PydanticBaseModel):
    """Request model for submitting a curated response."""

    workflow_run_id: str
    curated_response: str


class ChatInvokeResponse(PydanticBaseModel):
    """Response model for the /chat/invoke endpoint, mirrors WorkflowRunOutput."""

    final_response: str
    chat_history: List[ChatMessage]
    session_id: str
    # Potentially add error messages or status codes here later


# --- API Endpoints ---


@app.post("/chat/invoke", response_model=ChatInvokeResponse)
async def chat_invoke(request: ChatInvokeRequest) -> ChatInvokeResponse:
    """
    Main endpoint to interact with the chat workflow.

    Receives a user message and other session parameters, runs the AI workflow,
    persists the updated chat history, and returns the AI's response along with
    the updated history.
    """
    print(f"[CHAT INVOKE] Received request: {request.model_dump()}")
    session_id = request.session_id or DEFAULT_SESSION_ID
    chat_store_path = os.path.join(CHAT_SESSIONS_DIR, f"{session_id}_store.json")

    # Load chat store for the session
    chat_store = SimpleChatStore.from_persist_path(persist_path=chat_store_path)
    current_chat_history = chat_store.get_messages(session_id)

    # Prepare settings and models to use
    # If request provides them, use those, otherwise use defaults (which workflow handles if None)
    active_model_settings = (
        request.settings if request.settings else ModelSettings()
    )  # Default if not provided
    active_models_to_use = (
        request.initial_models_to_use
    )  # Can be None, workflow handles default

    start_event = WorkflowStartEvent(
        user_message=request.user_message,
        settings=active_model_settings,
        initial_models_to_use=active_models_to_use,
        chat_history=current_chat_history,
    )

    try:
        workflow_run_output = await workflow.run(start_event=start_event)

        if not isinstance(workflow_run_output, WorkflowRunOutput):
            # This should ideally not happen if the workflow is correctly configured
            raise HTTPException(
                status_code=500, detail="Workflow returned an unexpected output type."
            )

        # Update and persist chat store
        chat_store.set_messages(session_id, workflow_run_output.chat_history)
        chat_store.persist(persist_path=chat_store_path)

        return ChatInvokeResponse(
            final_response=workflow_run_output.final_response,
            chat_history=workflow_run_output.chat_history,
            session_id=session_id,
        )
    except Exception as e:
        # Log the exception for server-side debugging
        print(f"Error during workflow execution for session {session_id}: {e}")
        # Consider more specific error handling and user-friendly messages
        raise HTTPException(
            status_code=500, detail=f"An internal error occurred: {str(e)}"
        )


# --- SSE Chat Endpoint ---
@app.post("/chat/stream")
async def chat_stream(
    api_request: ChatStreamRequest, httpRequest: Request
):  # Added httpRequest for disconnect
    """
    Streaming endpoint for chat, using Server-Sent Events (SSE).

    Receives user message and session parameters, then streams workflow events
    (step updates, LLM tokens, candidate readiness, curation requests) back to the client.
    Handles chat history loading and persistence similarly to the invoke endpoint.
    Curation is now handled asynchronously via a specific workflow_run_id and the /chat/curate endpoint.
    """
    print(
        f"[/chat/stream] Received API request: {api_request.model_dump_json(indent=2)}"
    )
    session_id = api_request.session_id or DEFAULT_SESSION_ID
    chat_store_path = os.path.join(CHAT_SESSIONS_DIR, f"{session_id}_store.json")
    workflow_run_id = str(uuid.uuid4())
    print(f"[/chat/stream] Generated workflow_run_id: {workflow_run_id}")

    chat_store = SimpleChatStore.from_persist_path(persist_path=chat_store_path)
    raw_chat_history = chat_store.get_messages(session_id)
    print(
        f"[/chat/stream] Loaded {len(raw_chat_history)} raw messages from chat history for session {session_id}"
    )

    # Normalize chat history
    normalized_chat_history: List[ChatMessage] = []
    for msg in raw_chat_history:
        content = ""
        # Try to extract content from known structures
        if hasattr(msg, "blocks") and msg.blocks and isinstance(msg.blocks, list):
            for block in msg.blocks:
                if (
                    hasattr(block, "block_type")
                    and block.block_type == "text"
                    and hasattr(block, "text")
                ):
                    content = block.text
                    break
        elif hasattr(msg, "content") and msg.content is not None:
            content = str(msg.content)
        else:
            # Fallback if content cannot be readily extracted (e.g. tool calls we aren't handling yet for history)
            content = "[Non-text content or unable to parse]"

        # Create a new, simple ChatMessage, explicitly setting additional_kwargs to empty
        normalized_chat_history.append(
            ChatMessage(role=msg.role, content=content, additional_kwargs={})
        )

    print(
        f"[/chat/stream] Normalized chat history ({len(normalized_chat_history)} messages) for event construction:"
    )
    for i, msg_obj in enumerate(normalized_chat_history):
        # For printing, let's see its dict representation to be sure
        print(
            f"  [{i}] Role: {msg_obj.role}, Content: '{str(msg_obj.content)[:70]}...', KWArgs: {msg_obj.additional_kwargs}"
        )

    active_model_settings = (
        api_request.settings if api_request.settings else ModelSettings()
    )
    active_models_to_use = api_request.initial_models_to_use

    # Create the start event WITH the normalized history
    start_event = WorkflowStartEvent(
        user_message=api_request.user_message,
        settings=active_model_settings,
        initial_models_to_use=active_models_to_use,
        chat_history=normalized_chat_history,  # Use the list of new, simple ChatMessage objects
        workflow_run_id=workflow_run_id,
    )

    # Now, let's log the chat_history part of the actual start_event *object* before model_dump_json
    print(
        f"[/chat/stream] Verifying chat_history in constructed start_event object (before serialization):"
    )
    if start_event.chat_history:
        for i, ch_msg in enumerate(start_event.chat_history):
            print(
                f"  EVENT_MSG [{i}] Role: {ch_msg.role}, Content: '{str(ch_msg.content)[:70]}...', KWArgs: {ch_msg.additional_kwargs}, Has Blocks: {hasattr(ch_msg, 'blocks') and bool(getattr(ch_msg, 'blocks', None))}"
            )

    print(
        f"[/chat/stream] Logging WorkflowStartEvent (via model_dump_json) to be passed to workflow: {start_event.model_dump_json(indent=2)}"
    )

    async def event_generator() -> AsyncGenerator[str, None]:
        print(f"[SSE WORKFLOW_RUN_ID: {workflow_run_id}] Event generator started.")
        stream_active = True
        handler = None
        try:
            print(f"[SSE WORKFLOW_RUN_ID: {workflow_run_id}] Calling workflow.run()...")
            # Run the asynchronous workflow.run()
            handler = workflow.run(start_event=start_event)
            print(
                f"[SSE WORKFLOW_RUN_ID: {workflow_run_id}] workflow.run() returned handler: {type(handler)}. Iterating handler.stream_events()..."
            )

            async for event in handler.stream_events():
                print(
                    f"[SSE WORKFLOW_RUN_ID: {workflow_run_id}] Received event from handler.stream_events(): {type(event).__name__}"
                )
                if await httpRequest.is_disconnected():
                    stream_active = False
                    # Attempt to retrieve the future associated with this workflow_run_id
                    # The workflow_run_id should be available in the scope
                    future_to_cancel = workflow.active_futures.pop(
                        workflow_run_id, None
                    )
                    if future_to_cancel and not future_to_cancel.done():
                        # Ensure the future is cancelled on the loop it was created on
                        loop = future_to_cancel.get_loop()
                        loop.call_soon_threadsafe(future_to_cancel.cancel)
                        print(
                            f"[SSE WORKFLOW_RUN_ID: {workflow_run_id}] Client disconnected, future cancelled."
                        )
                    else:
                        print(
                            f"[SSE WORKFLOW_RUN_ID: {workflow_run_id}] Client disconnected, no active future or future already done for this run_id."
                        )
                    print(
                        f"[SSE WORKFLOW_RUN_ID: {workflow_run_id}] Breaking stream due to client disconnect."
                    )
                    break

                event_payload_str = None
                if isinstance(event, LLMTokenStreamEvent):
                    event_payload_str = f"0:{json.dumps(event.token)}\n"
                elif isinstance(event, WorkflowStepUpdateEvent):
                    payload = {"type": "workflow_step_update", **event.model_dump()}
                    event_payload_str = f"2:{json.dumps(payload)}\n"
                elif isinstance(event, LLMCandidateReadyEvent):
                    payload = {"type": "llm_candidate_ready", **event.model_dump()}
                    event_payload_str = f"2:{json.dumps(payload)}\n"
                elif isinstance(event, CurationRequiredEvent):
                    payload = {"type": "curation_required", **event.model_dump()}
                    event_payload_str = f"2:{json.dumps(payload)}\n"
                elif isinstance(event, WorkflowErrorEvent):
                    payload = {"type": "workflow_error", **event.model_dump()}
                    event_payload_str = f"1:{json.dumps(payload)}\n"
                elif isinstance(event, StopEvent):
                    workflow_run_output = event.result
                    if isinstance(workflow_run_output, WorkflowRunOutput):
                        yield f"0:{json.dumps(workflow_run_output.final_response)}\n"
                        chat_history_serializable = [
                            msg.model_dump() for msg in workflow_run_output.chat_history
                        ]
                        metadata_payload = {
                            "finishReason": "stop",
                            "chatHistory": chat_history_serializable,
                        }
                        yield f"d:{json.dumps(metadata_payload)}\n"
                    elif (
                        isinstance(workflow_run_output, dict)
                        and "error" in workflow_run_output
                    ):
                        error_payload = {
                            "type": "workflow_error",
                            "step_name": workflow_run_output.get(
                                "failed_step", "stop_event_error"
                            ),
                            "error_message": str(workflow_run_output["error"]),
                        }
                        yield f"1:{json.dumps(error_payload)}\n"
                    else:
                        error_payload = {
                            "type": "workflow_error",
                            "step_name": "stop_event",
                            "error_message": "Workflow completed with an unexpected result type.",
                        }
                        yield f"1:{json.dumps(error_payload)}\n"
                    print(
                        f"[SSE WORKFLOW_RUN_ID: {workflow_run_id}] StopEvent received, breaking stream."
                    )
                    break

                if event_payload_str:
                    yield event_payload_str

        except asyncio.CancelledError:
            print(
                f"[SSE WORKFLOW_RUN_ID: {workflow_run_id}] Stream cancelled (asyncio.CancelledError)."
            )
        except Exception as e:
            # Corrected: removed exc_info=True from print(), which is for logging module
            print(
                f"[SSE WORKFLOW_RUN_ID: {workflow_run_id}] Error in event_generator: {type(e).__name__} - {e}"
            )
            # For more detailed trace, consider using actual logging:
            # import logging; logging.exception(f"[SSE WORKFLOW_RUN_ID: {workflow_run_id}] Error in event_generator")
            error_payload_model = WorkflowErrorEvent(
                step_name="event_generator_error", error_message=str(e)
            )
            error_data = {"type": "workflow_error", **error_payload_model.model_dump()}
            yield f"1:{json.dumps(error_data)}\n"
        finally:
            print(
                f"[SSE WORKFLOW_RUN_ID: {workflow_run_id}] Event generator finishing."
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"X-Vercel-AI-Data-Stream": "v1"},
    )


@app.post("/chat/curate")
async def chat_curate(curation_request: CuratedResponseRequest):
    """
    Endpoint for the client to submit the curated response for a specific workflow run.
    This will resolve the asyncio.Future that the corresponding workflow run is awaiting.
    """
    run_id = curation_request.workflow_run_id
    curated_text = curation_request.curated_response

    console.print(
        f'[API /chat/curate] Received curation for run_id: {run_id}, response: "{curated_text[:50]}..."'
    )

    future = workflow.active_futures.get(run_id)

    if not future:
        console.print(
            f"[API /chat/curate] Error: No active future found for run_id: {run_id}"
        )
        raise HTTPException(
            status_code=404,
            detail=f"Workflow run ID {run_id} not found or already processed.",
        )

    if future.done():
        console.print(
            f"[API /chat/curate] Error: Future for run_id: {run_id} is already done."
        )
        # Future is already resolved or cancelled, perhaps client sent twice or too late.
        # We might still want to remove it if it wasn't removed by the workflow's finally block for some reason (unlikely)
        workflow.active_futures.pop(run_id, None)
        raise HTTPException(
            status_code=409,
            detail=f"Curation for workflow run ID {run_id} has already been submitted or the run was cancelled.",
        )

    try:
        # Get the loop associated with the future (which should be the one the workflow task is running on)
        loop = future.get_loop()
        # Safely set the result of the future from this (FastAPI worker) thread/context
        loop.call_soon_threadsafe(future.set_result, curated_text)
        console.print(
            f"[API /chat/curate] Successfully set future result for run_id: {run_id}"
        )
        # The future should ideally be removed by the workflow step itself in its finally block once it awakens.
        # However, to be absolutely sure it doesn't linger if the workflow step somehow fails *after* awakening but *before* its finally:
        # workflow.active_futures.pop(run_id, None) # Reconsidering this: let workflow manage its own future removal on completion.
        return {
            "status": "success",
            "message": f"Curation received for workflow run {run_id}.",
        }
    except Exception as e:
        console.print(
            f"[API /chat/curate] Error setting future result for run_id {run_id}: {e}"
        )
        # If setting result fails, try to set an exception on the future so the workflow doesn't hang forever.
        if not future.done():
            try:
                loop = future.get_loop()
                loop.call_soon_threadsafe(future.set_exception, e)
            except Exception as fut_e:
                console.print(
                    f"[API /chat/curate] Further error trying to set exception on future for {run_id}: {fut_e}"
                )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process curation for workflow run {run_id}: {str(e)}",
        )


@app.get("/")
async def read_root():
    """Simple root endpoint to confirm the API is running."""
    return {
        "message": "Welcome to the Elyse AI Workflow API! Visit /docs for API documentation."
    }


# --- To run this FastAPI app (from the project root directory 'elyse_llama'): ---
# uvicorn backend.main:app --reload
#
# Then access the API at http://127.0.0.1:8000
# And interactive documentation at http://127.0.0.1:8000/docs
