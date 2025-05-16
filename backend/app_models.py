from typing import Any, Dict, List, Optional

from litellm import (
    BaseModel as LiteLLMBaseModel,  # Alias to avoid conflict if llama_index.core.BaseModel is used elsewhere
)
from llama_index.core.llms import ChatMessage
from llama_index.core.workflow import Context, Event, StartEvent

"""
Defines Pydantic models for the Elyse AI application.

These models structure data for:
-   Application-level settings (`AppSettings`).
-   LLM call-specific settings (`ModelSettings`).
-   Workflow initiation data (`WorkflowStartEvent`).
-   Shared state within a workflow run (`GlobalContext`).
-   Events passed between workflow steps (e.g., `ProcessInput`, `RetrieveContext`).
-   The final output of a workflow run (`WorkflowRunOutput`).
-   Events specifically designed for streaming to clients via Server-Sent Events (SSE),
    such as `WorkflowStepUpdateEvent`, `LLMTokenStreamEvent`, `LLMCandidateReadyEvent`,
    `CurationRequiredEvent`, and `WorkflowErrorEvent`.

Using Pydantic ensures data validation, clear schemas, and improved developer experience.
"""

# Settings and Context Models


class AppSettings(LiteLLMBaseModel):
    """
    Global application settings, not specific to a single LLM call.
    These settings might control broader application behaviors.
    """

    tts_enabled: Optional[bool] = False
    """If True, enables Text-to-Speech functionality for AI responses."""
    stt_enabled: Optional[bool] = False
    """If True, enables Speech-to-Text functionality for user input."""
    stt_model: Optional[str] = "google"
    """Specifies the Speech-to-Text model/provider to be used (e.g., 'google', 'whisper')."""
    embedding_model: Optional[str] = "openai"
    """Specifies the embedding model for tasks like RAG or semantic search."""
    sfw_mode: Optional[bool] = False
    """If True, activates Safe-for-Work mode, potentially filtering or modifying content."""


class ModelSettings(LiteLLMBaseModel):
    """
    Settings applied to individual Language Model (LLM) API calls.
    These parameters control the generation behavior of the LLM.
    """

    model: Optional[str] = "gpt-4o-mini"  # Example default, can be overridden
    """The identifier of the LLM to be used (e.g., 'openai/gpt-4o-mini', 'claude-3-opus-20240229')."""
    temperature: Optional[float] = 0.7  # Common default for balanced creativity
    """Controls randomness: lower values (e.g., 0.2) make output more deterministic, higher values (e.g., 1.0) make it more random."""
    max_tokens: Optional[int] = (
        1500  # Default, adjust based on expected response length
    )
    """The maximum number of tokens the LLM should generate in its response."""
    top_p: Optional[float] = 1.0
    """Nucleus sampling: the model considers only tokens with cumulative probability mass up to `top_p`."""
    frequency_penalty: Optional[float] = 0.0
    """Positive values penalize new tokens based on their existing frequency, discouraging repetition."""
    presence_penalty: Optional[float] = 0.0
    """Positive values penalize new tokens based on whether they've appeared, encouraging topic diversity."""


class WorkflowStartEvent(StartEvent):
    """
    Event that initiates a run of the `AppWorkFlow`.
    Extends LlamaIndex's `StartEvent` to include application-specific initial data.
    """

    user_message: str
    """The message input by the user for the current conversational turn."""
    settings: ModelSettings
    """The LLM settings (temperature, max_tokens, etc.) to be applied for this workflow run."""
    initial_models_to_use: Optional[List[str]] = None
    """A list of LLM identifiers (e.g., ['gpt-4o-mini', 'claude-3-sonnet-20240229']) to be used for generating candidate responses."""
    chat_history: Optional[List[ChatMessage]] = None
    """The conversation history up to the previous turn, as a list of `llama_index.core.llms.ChatMessage` objects."""
    workflow_run_id: Optional[str] = None
    """A unique identifier for this specific workflow run. Crucial for asynchronous operations like API-based curation and client-side event correlation."""


class GlobalContext(Context):
    """
    The shared context object for an `AppWorkFlow` run, accessible by all workflow steps.
    It holds the evolving state of the workflow, such as messages, settings, and intermediate results.
    Inherits from `llama_index.core.workflow.Context`.
    """

    user_message: Optional[str] = None
    """The current user's message being processed in this turn."""
    chat_history: Optional[List[ChatMessage]] = None
    """The full conversation history, including the current user message and any previous AI responses. Updated by the workflow."""
    context_needed: Optional[bool] = False  # Placeholder, for future RAG implementation
    """Flag indicating if context retrieval (RAG) is deemed necessary for the current query."""
    context_retrieved: Optional[str] = (
        None  # Placeholder, for future RAG implementation
    )
    """The text content retrieved from a knowledge base if RAG is performed."""
    app_settings: Optional[AppSettings] = (
        None  # Defaulted by workflow if not provided in StartEvent
    )
    """Current application-level settings (TTS, STT, etc.)."""
    model_settings: Optional[ModelSettings] = (
        None  # Defaulted by workflow if not provided in StartEvent
    )
    """LLM generation settings (temperature, etc.) active for this workflow run."""
    models_to_use: Optional[List[str]] = None  # Defaulted by workflow if not provided
    """List of LLM identifiers selected for generating responses in this run."""
    response_candidates: Optional[List[Dict[str, str]]] = (
        None  # Changed to List[Dict] for model name + response
    )
    """A list of dictionaries, each containing 'model_name' and 'response' (or error) from an LLM."""
    current_system_prompt: Optional[str] = None
    """The system prompt constructed and used for the LLMs in the current turn."""
    curated_ai_response: Optional[str] = None
    """The AI response selected (or provided) by the user during the curation step. Can be a default if curation skipped/failed."""
    workflow_run_id: Optional[str] = None
    """The unique identifier for the current workflow run, propagated from `WorkflowStartEvent`."""


# Workflow Step Events (Internal Communication)
# These models represent data passed between sequential workflow steps.
# Each event typically signifies the completion of a step and carries its primary output.


class ProcessInput(Event):
    """Event produced after the `process_input` step has initialized the context."""

    first_output: str
    """A summary or confirmation message from the input processing step."""


class RetrieveContext(Event):
    """Event produced after the `retrieve_context` step (currently simulated)."""

    context_retrieved: str
    """The text content retrieved (or simulated) for potential RAG usage."""


class DynamicPromptBuilt(Event):
    """Event produced after the `build_dynamic_prompt` step has constructed the system prompt."""

    prompt_updated: str
    """A confirmation or snippet indicating the system prompt has been prepared."""


class LlmResponsesCollected(Event):
    """Event produced by `llm_manager` after all selected LLMs have generated responses (or errored)."""

    responses_collected_count: int
    """The number of response candidates (successful or error messages) collected from the LLMs."""


class CurationManager(Event):
    """Event produced by `curation_manager` after awaiting and receiving the user-curated response."""

    curated_response: str
    """The final AI response string, either chosen by the user or a default if curation was skipped/failed."""


# Workflow Final Output Model


class WorkflowRunOutput(LiteLLMBaseModel):
    """
    Defines the final output structure of a successful `AppWorkFlow` run.
    This model is the `result` payload of the `StopEvent` that terminates the workflow.
    """

    final_response: str
    """The curated AI response to be presented to the user."""
    chat_history: List[ChatMessage]
    """The complete, updated chat history, including the latest user message and the curated AI response."""
    workflow_run_id: Optional[str] = (
        None  # Ensure this is part of the final output for client tracking
    )
    """The unique identifier of the workflow run that produced this output."""


# Streaming-Specific Events (for SSE to Client)
# These events are designed to be written to `ctx.write_event_to_stream()` during workflow execution
# and are intended to be relayed via Server-Sent Events (SSE) to the client for real-time UI updates.


class WorkflowStepUpdateEvent(Event):
    """
    Signals real-time progress or status change from a specific workflow step.
    Useful for UIs to show which part of the process is currently active.
    """

    step_name: str
    """The programmatic name of the workflow step generating this update (e.g., 'retrieve_context')."""
    status: str  # E.g., "started", "completed", "skipped", "error", "fetching_data"
    """A descriptive status of the step (e.g., 'started', 'completed', 'fetching_data', 'error')."""
    data: Optional[Dict[str, Any]] = None  # Changed to Dict[str, Any] for flexibility
    """Optional dictionary for any supplementary data related to this update (e.g., a snippet of retrieved context)."""
    workflow_run_id: Optional[str] = None
    """The unique identifier of the workflow run this event belongs to."""


class LLMTokenStreamEvent(Event):
    """
    Carries a single token (or chunk) from an LLM's streaming response.
    Allows for a typewriter effect in the UI for LLM responses.
    """

    model_name: str
    """The identifier of the LLM that generated this token."""
    token: str
    """The token string itself."""
    is_final_chunk: bool = False
    """If True, indicates this is the last token/chunk for this specific model's response stream."""
    workflow_run_id: Optional[str] = None
    """The unique identifier of the workflow run this event belongs to."""


class LLMCandidateReadyEvent(Event):
    """
    Indicates that a full candidate response from one of the LLMs is ready.
    This can be used by the UI to display individual LLM responses as they arrive,
    before the final curation step.
    """

    model_name: str
    """The identifier of the model that generated this candidate response."""
    candidate_response: (
        str  # Could be a successful response or an error message for this model
    )
    """The full text of the candidate response from the LLM, or an error message if the call failed for this model."""
    workflow_run_id: Optional[str] = None
    """The unique identifier of the workflow run this event belongs to."""


class CurationRequiredEvent(Event):
    """
    Signals that all LLM candidates are ready and user curation is now required.
    The workflow will pause at the `curation_manager` step until a response is submitted
    via the `/chat/curate` API endpoint using the provided `workflow_run_id`.
    """

    response_candidates: List[
        Dict[str, str]
    ]  # Changed to List[Dict] to include model_name
    """A list of dictionaries, where each dictionary contains {'model_name': str, 'response': str} for each candidate."""
    message: str = "Curation required: Please select or provide the best response."
    """A user-facing message indicating that curation is needed."""
    workflow_run_id: str  # This is crucial.
    """The unique ID of the workflow run that requires curation. The client MUST send this back with the curated choice."""


class WorkflowErrorEvent(Event):
    """
    Signals that an error occurred during workflow execution, either within a step
    or during a service call (like an LLM API call).
    """

    step_name: Optional[str] = (
        None  # e.g., "llm_manager", "retrieve_context", "llm_service_call"
    )
    """The name of the workflow step or service operation where the error occurred, if identifiable."""
    error_message: str
    """A description of the error encountered."""
    model_name: Optional[str] = None  # If the error is specific to an LLM call
    """If the error is specific to an LLM call, the identifier of the model that failed."""
    workflow_run_id: Optional[str] = None
    """The unique identifier of the workflow run this error event belongs to."""
