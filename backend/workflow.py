import asyncio
import logging  # Added for file logging
from typing import Any, Dict, List, Optional

from llama_index.core.llms import ChatMessage
from llama_index.core.workflow import (
    Context,
    Event,  # Added Event for on_error typing
    StopEvent,
    Workflow,
    step,
)
from rich.console import Console
from rich.panel import Panel

from backend.app_models import (
    CurationManager,
    CurationRequiredEvent,
    DynamicPromptBuilt,
    LLMCandidateReadyEvent,  # Will be used more by llm_service later
    LlmResponsesCollected,
    LLMTokenStreamEvent,  # Will be used more by llm_service later
    ModelSettings,  # For llm_manager default
    ProcessInput,
    RetrieveContext,
    WorkflowErrorEvent,
    WorkflowRunOutput,
    WorkflowStartEvent,
    # Streaming events
    WorkflowStepUpdateEvent,
)
from backend.prompts.prompt_manager import generate_prompt
from backend.services.llm_service import WorkflowLlmService

"""
Defines the `AppWorkFlow`, the central LlamaIndex workflow for the Elyse AI application.

`AppWorkFlow` orchestrates a multi-step process for each conversational turn:
1.  **Input Processing**: Receives user messages and settings.
2.  **Context Retrieval**: (Currently simulated) Fetches relevant information.
3.  **Prompt Engineering**: Dynamically builds prompts for LLMs.
4.  **LLM Interaction**: Calls multiple LLMs concurrently via `WorkflowLlmService`.
5.  **Curation**: Pauses for human-in-the-loop selection of the best AI response.
    This step uses `asyncio.Future` to wait for an external signal (from the
    `/chat/curate` API endpoint) before proceeding.
6.  **Output Generation**: Finalizes the turn and packages the results.

Each step is defined as an asynchronous method decorated with `@step`. Steps communicate
by passing Pydantic models (events) and by reading/writing to a shared `Context` object.
The workflow is designed to be streamable, with steps emitting events like
`WorkflowStepUpdateEvent`, `LLMTokenStreamEvent`, and `CurationRequiredEvent` to
provide real-time feedback to the client via Server-Sent Events (SSE).

Error handling is managed by the `on_error` method, which attempts to log and stream
error details.
"""

# --- Logger Setup ---
LOG_FILE = "workflow.log"
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(LOG_FILE, mode="w")
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
# --- End Logger Setup ---

console = Console()


class AppWorkFlow(Workflow):
    """
    The main LlamaIndex workflow for the Elyse AI chat application.

    This workflow orchestrates a conversational turn, involving input processing,
    context retrieval (simulated), dynamic prompt building, LLM calls (via
    `WorkflowLlmService`), asynchronous human-in-the-loop curation, and
    final response generation. It emits various events for client-side streaming.

    Attributes:
        active_futures (Dict[str, asyncio.Future]): A dictionary to store active
            `asyncio.Future` objects, keyed by `workflow_run_id`. These are used
            by the `curation_manager` step to pause execution and await external
            curation input via the `/chat/curate` API endpoint.
    """

    def __init__(
        self, verbose: bool = False, timeout: Optional[float] = None, **kwargs: Any
    ):
        """
        Initializes the AppWorkFlow.

        Args:
            verbose: If True, enables verbose logging from LlamaIndex internals.
                     Defaults to False for cleaner API/application logs.
            timeout: An optional timeout in seconds for the entire workflow run.
            **kwargs: Additional keyword arguments passed to the parent `Workflow` class.
        """
        super().__init__(
            verbose=verbose,
            timeout=timeout,
            **kwargs,
        )
        self.active_futures: Dict[str, asyncio.Future] = {}
        logger.info("[AppWorkFlow] Initialized.")

    @step
    async def process_input(self, ctx: Context, ev: WorkflowStartEvent) -> ProcessInput:
        """
        Processes the initial `WorkflowStartEvent`.

        This first step initializes the workflow's `Context` for the current run.
        It stores the user's message, LLM settings, models to use, the unique
        `workflow_run_id` (if provided for streaming scenarios), and updates the
        chat history by appending the new user message.

        Emits `WorkflowStepUpdateEvent` for `started` and `completed` states.

        Args:
            ctx: The shared `Context` object for this workflow run.
            ev: The `WorkflowStartEvent` containing initial data like user message,
                settings, chat history, and `workflow_run_id`.

        Returns:
            A `ProcessInput` event, signaling completion and readiness for the next step.
        """
        step_name = "process_input"
        logger.info(f"[{step_name}] ({ev.workflow_run_id or 'N/A'}) Running step...")
        ctx.write_event_to_stream(
            WorkflowStepUpdateEvent(
                step_name=step_name,
                status="started",
                workflow_run_id=ev.workflow_run_id,
            )
        )

        await ctx.set("user_message", ev.user_message)
        await ctx.set("model_settings", ev.settings)
        await ctx.set("models_to_use", ev.initial_models_to_use or ["gpt-4o-mini"])
        if ev.workflow_run_id:
            await ctx.set("workflow_run_id", ev.workflow_run_id)
            logger.info(
                f"[{step_name}] ({ev.workflow_run_id}) Workflow run ID set in context."
            )

        initial_history = ev.chat_history or []
        updated_history_for_turn = initial_history + [
            ChatMessage(role="user", content=ev.user_message)
        ]
        await ctx.set("chat_history", updated_history_for_turn)
        logger.info(
            f"[{step_name}] ({ev.workflow_run_id or 'N/A'}) User message: '{ev.user_message[:50]}...', History updated to {len(updated_history_for_turn)} messages."
        )

        ctx.write_event_to_stream(
            WorkflowStepUpdateEvent(
                step_name=step_name,
                status="completed",
                workflow_run_id=ev.workflow_run_id,
            )
        )
        logger.info(f"[{step_name}] ({ev.workflow_run_id or 'N/A'}) Completed step.")
        return ProcessInput(
            first_output=f"Processed user message: {ev.user_message[:50]}..."
        )

    @step
    async def retrieve_context(self, ctx: Context, ev: ProcessInput) -> RetrieveContext:
        """
        (Simulated) Retrieves context relevant to the user's message.

        In a real RAG application, this step would query a knowledge base.
        Currently, it provides a placeholder implementation. The retrieved context
        is stored in the `Context`.

        Emits `WorkflowStepUpdateEvent` for `started` and `completed` states.

        Args:
            ctx: The shared `Context` object.
            ev: The `ProcessInput` event from the previous step.

        Returns:
            A `RetrieveContext` event containing the (simulated) retrieved text.
        """
        step_name = "retrieve_context"
        workflow_run_id = await ctx.get("workflow_run_id", "N/A")
        logger.info(f"[{step_name}] ({workflow_run_id}) Running step...")
        ctx.write_event_to_stream(
            WorkflowStepUpdateEvent(
                step_name=step_name, status="started", workflow_run_id=workflow_run_id
            )
        )

        user_message: str = await ctx.get("user_message", "")
        retrieved_text = "No specific context found for your query (simulated)."
        if "weather" in user_message.lower():
            retrieved_text = "The weather is sunny today (simulated)."

        await ctx.set("context_retrieved", retrieved_text)
        logger.info(
            f"[{step_name}] ({workflow_run_id}) Retrieved context: '{retrieved_text[:50]}...' for user message: '{user_message[:50]}...'"
        )
        ctx.write_event_to_stream(
            WorkflowStepUpdateEvent(
                step_name=step_name,
                status="completed",
                data={"retrieved_context_snippet": retrieved_text[:50]},
                workflow_run_id=workflow_run_id,
            )
        )
        logger.info(f"[{step_name}] ({workflow_run_id}) Completed step.")
        return RetrieveContext(context_retrieved=retrieved_text)

    @step
    async def build_dynamic_prompt(
        self, ctx: Context, ev_rc: RetrieveContext
    ) -> DynamicPromptBuilt:
        """
        Constructs the system prompt for the LLMs.

        Uses `generate_prompt` (from `prompt_manager.py`) to create a system
        prompt, potentially incorporating the `context_retrieved` from the
        previous step. The generated prompt is stored in the `Context`.

        Emits `WorkflowStepUpdateEvent` for `started` and `completed` states.

        Args:
            ctx: The shared `Context` object.
            ev_rc: The `RetrieveContext` event, carrying the retrieved context.

        Returns:
            A `DynamicPromptBuilt` event, indicating the prompt has been prepared.
        """
        step_name = "build_dynamic_prompt"
        workflow_run_id = await ctx.get("workflow_run_id", "N/A")
        logger.info(f"[{step_name}] ({workflow_run_id}) Running step...")
        ctx.write_event_to_stream(
            WorkflowStepUpdateEvent(
                step_name=step_name, status="started", workflow_run_id=workflow_run_id
            )
        )

        context_retrieved: str = await ctx.get("context_retrieved", "")
        current_system_prompt_str = generate_prompt(context_retrieved)
        await ctx.set("current_system_prompt", current_system_prompt_str)
        logger.info(
            f"[{step_name}] ({workflow_run_id}) System prompt generated: '{current_system_prompt_str[:70]}...'"
        )

        ctx.write_event_to_stream(
            WorkflowStepUpdateEvent(
                step_name=step_name,
                status="completed",
                data={"system_prompt_snippet": current_system_prompt_str[:50]},
                workflow_run_id=workflow_run_id,
            )
        )
        logger.info(f"[{step_name}] ({workflow_run_id}) Completed step.")
        return DynamicPromptBuilt(
            prompt_updated=f"System prompt updated: {current_system_prompt_str[:50]}..."
        )

    @step
    async def llm_manager(
        self, ctx: Context, ev: DynamicPromptBuilt
    ) -> LlmResponsesCollected:
        """
        Manages concurrent calls to multiple Language Models (LLMs).

        Retrieves the system prompt, chat history, models to use, and settings from
        the `Context`. It then uses `WorkflowLlmService` to make concurrent API calls.
        Collected responses (or errors) are stored in `Context`. Events like
        `LLMTokenStreamEvent` and `LLMCandidateReadyEvent` are streamed by the service.

        Emits `WorkflowStepUpdateEvent` for `started` and `completed` states.

        Args:
            ctx: The shared `Context` object.
            ev: The `DynamicPromptBuilt` event from the previous step.

        Returns:
            An `LlmResponsesCollected` event with the count of collected responses.
        """
        step_name = "llm_manager"
        workflow_run_id = await ctx.get("workflow_run_id", "N/A")
        logger.info(f"[{step_name}] ({workflow_run_id}) Running step...")
        ctx.write_event_to_stream(
            WorkflowStepUpdateEvent(
                step_name=step_name, status="started", workflow_run_id=workflow_run_id
            )
        )

        chat_history_from_ctx: List[ChatMessage] = await ctx.get("chat_history", [])
        system_prompt: str = await ctx.get("current_system_prompt", "")
        models_to_use: List[str] = await ctx.get("models_to_use", [])
        global_model_settings: ModelSettings = (
            await ctx.get("model_settings") or ModelSettings()
        )

        logger.info(
            f"[{step_name}] ({workflow_run_id}) Models: {models_to_use}, System prompt: '{system_prompt[:70]}...', History: {len(chat_history_from_ctx)} msgs."
        )

        llm_service = WorkflowLlmService()
        collected_responses_content = await llm_service.get_responses_for_workflow(
            ctx=ctx,  # Pass context for service to stream events
            system_prompt=system_prompt,
            chat_history=chat_history_from_ctx,
            models_to_use=models_to_use,
            global_model_settings=global_model_settings,
        )

        if not models_to_use:
            logger.info(
                f"[{step_name}] ({workflow_run_id}) No models specified. Skipping LLM calls."
            )
            ctx.write_event_to_stream(
                WorkflowStepUpdateEvent(
                    step_name=step_name,
                    status="skipped",
                    data={"reason": "No models specified"},
                    workflow_run_id=workflow_run_id,
                )
            )

        await ctx.set("response_candidates", collected_responses_content)
        logger.info(
            f"[{step_name}] ({workflow_run_id}) Collected {len(collected_responses_content)} response candidates."
        )
        ctx.write_event_to_stream(
            WorkflowStepUpdateEvent(
                step_name=step_name,
                status="completed",
                data={"candidates_count": len(collected_responses_content)},
                workflow_run_id=workflow_run_id,
            )
        )
        logger.info(f"[{step_name}] ({workflow_run_id}) Completed step.")
        return LlmResponsesCollected(
            responses_collected_count=len(collected_responses_content)
        )

    @step
    async def curation_manager(
        self, ctx: Context, ev: LlmResponsesCollected
    ) -> CurationManager:
        """
        Manages human-in-the-loop (HITL) curation of LLM responses.

        If `response_candidates` are available and a `workflow_run_id` is set in
        the context (indicating a streaming/async session), this step creates an
        `asyncio.Future`. It stores this future in `self.active_futures` keyed by
        the `workflow_run_id` and emits a `CurationRequiredEvent` to the client.
        The workflow then `await`s this future.

        The future is expected to be resolved by an external call to the `/chat/curate`
        API endpoint, which provides the `workflow_run_id` and the curated text.
        Once resolved, the curated response is stored in the `Context`.
        If no candidates or no `workflow_run_id`, curation is skipped.

        Emits `WorkflowStepUpdateEvent` and `CurationRequiredEvent`.

        Args:
            ctx: The shared `Context` object.
            ev: The `LlmResponsesCollected` event.

        Returns:
            A `CurationManager` event containing the curated response string.
            If curation is skipped or cancelled, this will be a default message.
        """
        step_name = "curation_manager"
        workflow_run_id = await ctx.get("workflow_run_id")
        logger.info(f"[{step_name}] ({workflow_run_id or 'N/A'}) Running step...")
        ctx.write_event_to_stream(
            WorkflowStepUpdateEvent(
                step_name=step_name, status="started", workflow_run_id=workflow_run_id
            )
        )

        response_candidates: List[Dict[str, str]] = await ctx.get(
            "response_candidates", []
        )
        chosen_response_string = "No response was curated or curation was skipped."
        future: Optional[asyncio.Future] = None

        if not workflow_run_id:
            logger.error(
                f"[{step_name}] Error: `workflow_run_id` not found in context. Curation requires a `workflow_run_id`. Skipping."
            )
            ctx.write_event_to_stream(
                WorkflowStepUpdateEvent(
                    step_name=step_name,
                    status="error",
                    data={"reason": "Missing workflow_run_id for curation"},
                    workflow_run_id=None,
                )
            )
            ctx.write_event_to_stream(
                WorkflowErrorEvent(
                    step_name=step_name,
                    error_message="workflow_run_id missing; cannot enable API-based curation.",
                    workflow_run_id=None,
                )
            )
            # Fall through to set default response and complete step

        elif not response_candidates:
            logger.info(
                f"[{step_name}] ({workflow_run_id}) No response candidates to curate. Skipping actual wait."
            )
            ctx.write_event_to_stream(
                WorkflowStepUpdateEvent(
                    step_name=step_name,
                    status="skipped",
                    data={"reason": "No response candidates"},
                    workflow_run_id=workflow_run_id,
                )
            )
            # Fall through to set default response and complete step
        else:
            logger.info(
                f"[{step_name}] ({workflow_run_id}) Creating future and waiting for user curation via API..."
            )
            future = asyncio.Future()
            self.active_futures[workflow_run_id] = future

            ctx.write_event_to_stream(
                CurationRequiredEvent(
                    response_candidates=response_candidates,
                    message="Curation required: Please select or provide the best response via the /chat/curate endpoint.",
                    workflow_run_id=workflow_run_id,
                )
            )

            try:
                logger.info(
                    f"[{step_name}] ({workflow_run_id}) Awaiting future.set_result() from /chat/curate API..."
                )
                chosen_response_string = await future
                logger.info(
                    f"[{step_name}] ({workflow_run_id}) Future resolved. Curated response received: '{chosen_response_string[:70]}...'"
                )
            except asyncio.CancelledError:
                logger.warning(
                    f"[{step_name}] ({workflow_run_id}) Curation future was cancelled (e.g., client disconnected or workflow timeout)."
                )
                chosen_response_string = "Curation was cancelled."
            except Exception as e:
                logger.error(
                    f"[{step_name}] ({workflow_run_id}) Error awaiting curation future: {e}.",
                    exc_info=True,
                )
                chosen_response_string = (
                    f"Curation failed due to an error: {str(e)[:100]}"
                )
                ctx.write_event_to_stream(
                    WorkflowErrorEvent(
                        step_name=step_name,
                        error_message=f"Error during curation wait: {str(e)[:100]}",
                        workflow_run_id=workflow_run_id,
                    )
                )
            finally:
                if workflow_run_id in self.active_futures:
                    del self.active_futures[workflow_run_id]
                    logger.info(
                        f"[{step_name}] ({workflow_run_id}) Future removed from active_futures."
                    )

        await ctx.set("curated_ai_response", chosen_response_string)
        logger.info(
            f"[{step_name}] ({workflow_run_id or 'N/A'}) Curated AI response set to: '{chosen_response_string[:50]}...'"
        )
        ctx.write_event_to_stream(
            WorkflowStepUpdateEvent(
                step_name=step_name,
                status="completed",
                data={"curated_response_snippet": chosen_response_string[:50]},
                workflow_run_id=workflow_run_id,
            )
        )
        logger.info(f"[{step_name}] ({workflow_run_id or 'N/A'}) Completed step.")
        return CurationManager(curated_response=chosen_response_string)

    @step
    async def stop_event(self, ctx: Context, ev: CurationManager) -> StopEvent:
        """
        Final step: Concludes the workflow turn and packages the output.

        Retrieves the `curated_ai_response` (which might be a default if curation
        was skipped or failed) and the current `chat_history` from the `Context`.
        Appends the curated AI response to the chat history as an "assistant" message.
        Constructs a `WorkflowRunOutput` object containing the final response and
        the complete, updated chat history.
        This `WorkflowRunOutput` is then wrapped in LlamaIndex's `StopEvent`,
        signaling the end of the workflow execution for this turn.

        Emits `WorkflowStepUpdateEvent` for `started` and `completed` states.

        Args:
            ctx: The shared `Context` object.
            ev: The `CurationManager` event containing the (potentially default)
                curated AI response.

        Returns:
            A `StopEvent` with `WorkflowRunOutput` as its `result`.
        """
        step_name = "stop_event"
        workflow_run_id = await ctx.get("workflow_run_id", "N/A")
        logger.info(f"[{step_name}] ({workflow_run_id}) Running step...")
        ctx.write_event_to_stream(
            WorkflowStepUpdateEvent(
                step_name=step_name, status="started", workflow_run_id=workflow_run_id
            )
        )

        chat_history_for_this_run: List[ChatMessage] = await ctx.get("chat_history", [])
        curated_response_content: str = ev.curated_response
        final_chat_history = chat_history_for_this_run

        # Add assistant's curated response to history, unless it's the placeholder for no curation.
        if curated_response_content and curated_response_content not in [
            "No response was curated or curation was skipped.",
            "Curation was cancelled.",
        ]:  # Add more specific default/error messages if they shouldn't be in history
            final_chat_history = final_chat_history + [
                ChatMessage(role="assistant", content=curated_response_content)
            ]
            logger.info(
                f"[{step_name}] ({workflow_run_id}) Appended assistant response to history: '{curated_response_content[:50]}...'"
            )
        else:
            logger.info(
                f"[{step_name}] ({workflow_run_id}) No valid curated response to add to chat history (used: '{curated_response_content[:50]}...')."
            )
        logger.info(
            f"[{step_name}] ({workflow_run_id}) Final history length: {len(final_chat_history)} messages."
        )

        output_data = WorkflowRunOutput(
            final_response=curated_response_content,
            chat_history=final_chat_history,
            workflow_run_id=workflow_run_id,  # Include run_id in final output
        )

        ctx.write_event_to_stream(
            WorkflowStepUpdateEvent(
                step_name=step_name,
                status="completed",
                data={"final_response_snippet": curated_response_content[:50]},
                workflow_run_id=workflow_run_id,
            )
        )
        logger.info(
            f"[{step_name}] ({workflow_run_id}) Completed step. Returning StopEvent."
        )
        return StopEvent(result=output_data)

    async def on_error(self, error: Exception, event: Optional[Event]) -> Event:
        """
        Global error handler for the workflow.

        This method is invoked by LlamaIndex if an unhandled exception occurs in any
        workflow step. It logs the error with traceback and attempts to identify the
        failing step. It then constructs a `WorkflowErrorEvent` (though direct streaming
        from here is challenging as `ctx` isn't directly available) and returns a
        `StopEvent` containing error details. This ensures the workflow terminates
        and the client (if streaming) can be notified of the failure.

        Args:
            error: The exception that was raised.
            event: The LlamaIndex event object that was being processed by the failing step.
                   Can be None if error occurs outside step processing.

        Returns:
            A `StopEvent` with a dictionary result: `{"error": str(error), "failed_step": str}`.
        """
        step_name_from_event = "unknown_step"
        event_type_name = type(event).__name__ if event else "N/A"

        logger.error(
            f"[AppWorkFlow] on_error triggered. Error: {error}. Event type: {event_type_name}",
            exc_info=True,  # Include traceback in log
        )
        console.print(
            Panel(
                f"[bold red]Workflow Error:[/bold red] {error}\n"
                f"[dim]Event leading to error: {event_type_name}[/dim]",
                title="[bold red]Workflow Execution Failed[/bold red]",
                border_style="red",
            )
        )

        # Try to infer step name from the event that caused the error
        if event:
            if hasattr(event, "fn_name"):  # LlamaIndex internal step event attribute
                step_name_from_event = getattr(event, "fn_name", "unknown_fn_name")
            # Add more specific checks if standard event types are passed to on_error
            elif isinstance(event, WorkflowStartEvent):
                step_name_from_event = "process_input (inferred)"
            elif isinstance(event, ProcessInput):
                step_name_from_event = "retrieve_context (inferred)"
            elif isinstance(event, RetrieveContext):
                step_name_from_event = "build_dynamic_prompt (inferred)"
            elif isinstance(event, DynamicPromptBuilt):
                step_name_from_event = "llm_manager (inferred)"
            elif isinstance(event, LlmResponsesCollected):
                step_name_from_event = "curation_manager (inferred)"
            elif isinstance(event, CurationManager):
                step_name_from_event = "stop_event (inferred)"
            # Check for our custom WorkflowStepUpdateEvent if it bubbles up here
            elif hasattr(event, "step_name") and isinstance(
                getattr(event, "step_name"), str
            ):
                step_name_from_event = f"step_emitting_{getattr(event, 'step_name')}"

        # The primary way to notify client of error is via the StopEvent's result.
        # Direct streaming of WorkflowErrorEvent from on_error is not straightforward
        # as the context (ctx) of the specific failed run isn't directly available here.
        # Errors should ideally be caught within steps to use ctx.write_event_to_stream.
        logger.info(
            f"[AppWorkFlow] on_error is returning a StopEvent with error details. Failed step (best guess): {step_name_from_event}"
        )
        return StopEvent(
            result={"error": str(error), "failed_step": step_name_from_event}
        )
