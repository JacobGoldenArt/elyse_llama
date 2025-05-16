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
This module defines the main application workflow, `AppWorkFlow`.

`AppWorkFlow` orchestrates the sequence of operations for a single turn in the 
chat application, from processing user input to generating and curating AI responses.
It leverages LlamaIndex's `Workflow` and `@step` decorators to define a graph of 
interconnected operations.

Each step in the workflow typically takes an input event, interacts with the shared 
`Context` (to get/set data), and produces an output event that triggers the next step.
"""

# --- Logger Setup ---
# Get the root directory of the project if possible, or log relative to workflow.py
# For simplicity, we'll log to 'workflow.log' in the directory where the script is run from (project root ideally)
LOG_FILE = "workflow.log"

# Create a logger
logger = logging.getLogger(__name__)  # Use the module's name for the logger
logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels of logs

# Create a file handler
file_handler = logging.FileHandler(
    LOG_FILE, mode="w"
)  # 'w' to overwrite log each run for clarity
file_handler.setLevel(logging.DEBUG)

# Create a logging format
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)

# Add the handlers to the logger
if not logger.handlers:  # Avoid adding multiple handlers if module is reloaded
    logger.addHandler(file_handler)
# --- End Logger Setup ---

console = (
    Console()
)  # Keep rich console for any direct output if needed later, but logs go to file


class AppWorkFlow(Workflow):
    """
    The main LlamaIndex workflow for the Elyse AI chat application.

    This workflow defines the logic for a single conversational turn:
    1.  `process_input`: Accepts user message and initial settings, updates chat history.
    2.  `retrieve_context`: (Simulated) Fetches relevant context based on user message.
    3.  `build_dynamic_prompt`: Constructs the system prompt using retrieved context.
    4.  `llm_manager`: Invokes multiple LLMs to generate candidate responses.
    5.  `curation_manager`: Allows human-in-the-loop selection of the best response.
    6.  `stop_event`: Finalizes the turn, packaging the result and updated chat history.

    The workflow uses a `GlobalContext` (defined in `app_models.py`) to share state
    between steps, such as chat history, model settings, and LLM responses.
    It now also manages `asyncio.Future` objects for asynchronous user input during curation.
    """

    def __init__(
        self, verbose: bool = False, timeout: Optional[float] = None, **kwargs
    ):
        # Ensure CUSTOM_EVENT_BUS is set for astream_events and custom event handling
        super().__init__(
            verbose=verbose,
            timeout=timeout,
            **kwargs,
        )
        self.active_futures: Dict[str, asyncio.Future] = {}
        logger.info("[AppWorkFlow] Initialized.")
        # Ensure that the GlobalContext is used if not specified otherwise by LlamaIndex internals
        # This might not be strictly necessary if LlamaIndex handles context creation as expected.
        # self.context_type = GlobalContext

    @step
    async def process_input(self, ctx: Context, ev: WorkflowStartEvent) -> ProcessInput:
        """
        First step: Processes the initial user input and workflow start event.

        This step takes the `WorkflowStartEvent` (containing the user's message,
        LLM settings, models to use, and previous chat history) and initializes
        the workflow's `Context` for the current run. It appends the new user
        message to the chat history.

        Args:
            ctx: The shared `GlobalContext` object for the workflow run.
            ev: The `WorkflowStartEvent` that triggered this workflow run.

        Returns:
            A `ProcessInput` event, typically carrying a confirmation or summary,
            to signal completion and trigger the next step.
        """
        step_name = "process_input"
        logger.info(f"[AppWorkFlow] Running step: {step_name}...")
        ctx.write_event_to_stream(
            WorkflowStepUpdateEvent(step_name=step_name, status="started")
        )

        await ctx.set("user_message", ev.user_message)
        await ctx.set("model_settings", ev.settings)
        await ctx.set("models_to_use", ev.initial_models_to_use or ["gpt-4o-mini"])
        if ev.workflow_run_id:  # Propagate workflow_run_id if provided
            await ctx.set("workflow_run_id", ev.workflow_run_id)
            logger.info(
                f"[{step_name}] Workflow run ID set in context: {ev.workflow_run_id}"
            )

        initial_history = ev.chat_history or []
        # Append the current user's message to the history for this turn.
        updated_history_for_turn = initial_history + [
            ChatMessage(role="user", content=ev.user_message)
        ]
        await ctx.set("chat_history", updated_history_for_turn)
        logger.info(
            f"[{step_name}] User message: '{ev.user_message[:50]}...', History length: {len(updated_history_for_turn)}"
        )

        ctx.write_event_to_stream(
            WorkflowStepUpdateEvent(step_name=step_name, status="completed")
        )
        logger.info(f"[AppWorkFlow] Completed step: {step_name}.")
        return ProcessInput(
            first_output=f"First step: Received user message: {ev.user_message[:50]}..."
        )

    @step
    async def retrieve_context(self, ctx: Context, ev: ProcessInput) -> RetrieveContext:
        """
        Second step: Retrieves context relevant to the user's message (currently simulated).

        In a full RAG application, this step would query a knowledge base (e.g., vector store)
        using the user's message or derived queries. For now, it provides a placeholder
        implementation that returns a fixed string if certain keywords are present.

        Args:
            ctx: The shared `GlobalContext` object.
            ev: The `ProcessInput` event from the previous step.

        Returns:
            A `RetrieveContext` event containing the retrieved text.
        """
        step_name = "retrieve_context"
        logger.info(f"[AppWorkFlow] Running step: {step_name}...")
        ctx.write_event_to_stream(
            WorkflowStepUpdateEvent(step_name=step_name, status="started")
        )

        user_message = await ctx.get("user_message", "")
        retrieved_text = "No specific context found for your query."
        # Simple keyword-based context simulation.
        if "weather" in user_message.lower():
            retrieved_text = "The weather is sunny today."

        await ctx.set("context_retrieved", retrieved_text)
        logger.info(
            f"[{step_name}] User message: '{user_message[:50]}...', Retrieved context: '{retrieved_text[:50]}...'"
        )
        ctx.write_event_to_stream(
            WorkflowStepUpdateEvent(
                step_name=step_name,
                status="completed",
                data={"retrieved_context_snippet": retrieved_text[:50]},
            )
        )
        logger.info(f"[AppWorkFlow] Completed step: {step_name}.")
        return RetrieveContext(context_retrieved=retrieved_text)

    @step
    async def build_dynamic_prompt(
        self, ctx: Context, ev_rc: RetrieveContext
    ) -> DynamicPromptBuilt:
        """
        Third step: Constructs the system prompt for the LLMs.

        This step uses the `generate_prompt` utility (from `prompt_manager.py`)
        to create a system prompt, potentially incorporating the `context_retrieved`
        from the previous step. The generated system prompt is stored in the context.

        Args:
            ctx: The shared `GlobalContext` object.
            ev_rc: The `RetrieveContext` event, carrying the retrieved context string.

        Returns:
            A `DynamicPromptBuilt` event, indicating the prompt has been prepared.
        """
        step_name = "build_dynamic_prompt"
        logger.info(f"[AppWorkFlow] Running step: {step_name}...")
        ctx.write_event_to_stream(
            WorkflowStepUpdateEvent(step_name=step_name, status="started")
        )

        context_retrieved = await ctx.get(
            "context_retrieved", ""
        )  # Use context from ctx set by previous step
        current_system_prompt_str = generate_prompt(context_retrieved)
        await ctx.set("current_system_prompt", current_system_prompt_str)
        logger.info(
            f"[{step_name}] Context retrieved: '{context_retrieved[:50]}...', System prompt: '{current_system_prompt_str[:70]}...'"
        )

        ctx.write_event_to_stream(
            WorkflowStepUpdateEvent(
                step_name=step_name,
                status="completed",
                data={"system_prompt_snippet": current_system_prompt_str[:50]},
            )
        )
        logger.info(f"[AppWorkFlow] Completed step: {step_name}.")
        return DynamicPromptBuilt(
            prompt_updated=f"Prompt updated with: {current_system_prompt_str[:50]}..."
        )

    @step
    async def llm_manager(
        self, ctx: Context, ev: DynamicPromptBuilt
    ) -> LlmResponsesCollected:
        """
        Fourth step: Manages calls to multiple Language Models (LLMs).

        This step retrieves the prepared system prompt, current chat history,
        list of models to use, and LLM settings from the context. It then uses
        the `WorkflowLlmService` to make concurrent API calls to the specified LLMs.
        The collected responses (or error messages) are stored in the context.

        Args:
            ctx: The shared `GlobalContext` object.
            ev: The `DynamicPromptBuilt` event from the previous step.

        Returns:
            An `LlmResponsesCollected` event containing the count of collected responses.
        """
        step_name = "llm_manager"
        logger.info(f"[AppWorkFlow] Running step: {step_name}...")
        ctx.write_event_to_stream(
            WorkflowStepUpdateEvent(step_name=step_name, status="started")
        )

        chat_history_from_ctx = await ctx.get("chat_history", [])
        system_prompt = await ctx.get("current_system_prompt", "")
        models_to_use = await ctx.get("models_to_use", [])
        global_model_settings = await ctx.get("model_settings") or ModelSettings()

        logger.info(
            f"[{step_name}] Models to use: {models_to_use}, System prompt: '{system_prompt[:70]}...', History length: {len(chat_history_from_ctx)}"
        )

        llm_service = WorkflowLlmService()

        collected_responses_content = await llm_service.get_responses_for_workflow(
            ctx=ctx,
            system_prompt=system_prompt,
            chat_history=chat_history_from_ctx,
            models_to_use=models_to_use,
            global_model_settings=global_model_settings,
        )

        if not models_to_use:
            logger.info(f"[{step_name}] No models specified. Skipping LLM calls.")
            # Potentially emit an event here too if that's useful for UI
            ctx.write_event_to_stream(
                WorkflowStepUpdateEvent(
                    step_name=step_name,
                    status="skipped",
                    data={"reason": "No models specified"},
                )
            )

        await ctx.set("response_candidates", collected_responses_content)
        logger.info(
            f"[{step_name}] Collected {len(collected_responses_content)} response candidates."
        )
        ctx.write_event_to_stream(
            WorkflowStepUpdateEvent(
                step_name=step_name,
                status="completed",
                data={"candidates_count": len(collected_responses_content)},
            )
        )
        logger.info(f"[AppWorkFlow] Completed step: {step_name}.")
        return LlmResponsesCollected(
            responses_collected_count=len(collected_responses_content)
        )

    @step
    async def curation_manager(
        self, ctx: Context, ev: LlmResponsesCollected
    ) -> CurationManager:
        """
        Fifth step: Handles human-in-the-loop curation of LLM responses via async future.

        This step retrieves `response_candidates` and the `workflow_run_id` from context.
        If candidates exist, it creates an `asyncio.Future`, stores it keyed by the run ID,
        and emits a `CurationRequiredEvent` (with candidates and the run ID) to the stream.
        It then `await`s the future. The future is expected to be resolved by an external
        API call (e.g., /chat/curate) which provides the curated response.
        The chosen response is then stored in the context as `curated_ai_response`.
        If no candidates, it skips curation.

        Args:
            ctx: The shared `GlobalContext` object.
            ev: The `LlmResponsesCollected` event, indicating LLM responses are ready.

        Returns:
            A `CurationManager` event containing the user-curated response string.
        """
        step_name = "curation_manager"
        logger.info(f"[AppWorkFlow] Running step: {step_name}...")
        ctx.write_event_to_stream(
            WorkflowStepUpdateEvent(step_name=step_name, status="started")
        )

        response_candidates = await ctx.get("response_candidates", [])
        workflow_run_id = await ctx.get("workflow_run_id")  # Get the run ID
        logger.info(
            f"[{step_name}] Workflow Run ID: {workflow_run_id}, Candidates count: {len(response_candidates)}"
        )
        chosen_response_string = "No response was curated."
        future: Optional[asyncio.Future] = (
            None  # Define future here for the finally block
        )

        if not workflow_run_id:
            logger.error(
                f"[{step_name}] Error: workflow_run_id not found in context. Skipping curation."
            )
            ctx.write_event_to_stream(
                WorkflowStepUpdateEvent(
                    step_name=step_name,
                    status="error",
                    data={"reason": "Missing workflow_run_id"},
                )
            )
            ctx.write_event_to_stream(
                WorkflowErrorEvent(
                    step_name=step_name,
                    error_message="workflow_run_id missing in context during curation",
                )
            )
            await ctx.set("curated_ai_response", chosen_response_string)
            logger.info(f"[AppWorkFlow] Completed step (error path): {step_name}.")
            return CurationManager(curated_response=chosen_response_string)

        if not response_candidates:
            logger.info(
                f"[{step_name}] No response candidates to curate. Skipping actual wait."
            )
            ctx.write_event_to_stream(
                WorkflowStepUpdateEvent(
                    step_name=step_name,
                    status="skipped",
                    data={"reason": "No response candidates"},
                )
            )
        else:
            logger.info(
                f"[{step_name}] For run {workflow_run_id}: Waiting for user input via API..."
            )
            future = asyncio.Future()
            self.active_futures[workflow_run_id] = future

            ctx.write_event_to_stream(
                CurationRequiredEvent(
                    response_candidates=response_candidates,
                    message="Curation required: Please select or provide the best response via the /chat/curate endpoint.",
                    workflow_run_id=workflow_run_id,  # Pass the run ID
                )
            )

            try:
                logger.info(
                    f"[{step_name}] ({workflow_run_id}): Awaiting future.set_result()..."
                )
                chosen_response_string = await future
                logger.info(
                    f"[{step_name}] ({workflow_run_id}): Future resolved. Received: {chosen_response_string[:70]}..."
                )
            except asyncio.CancelledError:
                logger.warning(
                    f"[{step_name}] ({workflow_run_id}): Future was cancelled. Skipping curation."
                )
                chosen_response_string = "Curation was cancelled."
                # workflow step will continue, and this response will be passed to stop_event
            except Exception as e:
                logger.error(
                    f"[{step_name}] ({workflow_run_id}): Error awaiting future: {e}. Skipping curation."
                )
                chosen_response_string = f"Curation failed due to an error: {e}"
                # Optionally re-raise or emit a more specific error event if needed
            finally:
                # Clean up the future from the active_futures dict
                if workflow_run_id in self.active_futures:
                    del self.active_futures[workflow_run_id]
                    logger.info(
                        f"[{step_name}] ({workflow_run_id}): Future removed from active_futures."
                    )

        await ctx.set("curated_ai_response", chosen_response_string)
        logger.info(
            f"[{step_name}] Curated response: '{chosen_response_string[:50]}...'"
        )
        ctx.write_event_to_stream(
            WorkflowStepUpdateEvent(
                step_name=step_name,
                status="completed",
                data={"curated_response_snippet": chosen_response_string[:50]},
            )
        )
        logger.info(f"[AppWorkFlow] Completed step: {step_name}.")
        return CurationManager(curated_response=chosen_response_string)

    @step
    async def stop_event(self, ctx: Context, ev: CurationManager) -> StopEvent:
        """
        Final step: Concludes the workflow run for the current turn.

        This step takes the `curated_ai_response` from the `CurationManager` event,
        appends it to the `chat_history` in the context with the role "assistant",
        and then packages the `final_response` (the curated one) and the fully
        updated `chat_history` into a `WorkflowRunOutput` model.
        This output is then wrapped in LlamaIndex's `StopEvent` to signal the end
        of the workflow execution for this turn.

        Args:
            ctx: The shared `GlobalContext` object.
            ev: The `CurationManager` event containing the user-selected AI response.

        Returns:
            A `StopEvent` containing the `WorkflowRunOutput` as its result.
        """
        step_name = "stop_event"  # Or perhaps "finalize_turn"
        logger.info(f"[AppWorkFlow] Running step: {step_name}...")
        ctx.write_event_to_stream(
            WorkflowStepUpdateEvent(step_name=step_name, status="started")
        )

        chat_history_for_this_run = await ctx.get("chat_history", [])
        curated_response_content = ev.curated_response
        final_chat_history = chat_history_for_this_run

        if (
            curated_response_content
            and curated_response_content != "No response was curated."
        ):
            final_chat_history = final_chat_history + [
                ChatMessage(role="assistant", content=curated_response_content)
            ]
            logger.info(
                f"[{step_name}] Appended assistant response to history: {curated_response_content[:50]}..."
            )
        else:
            logger.info(
                f"[{step_name}] No valid curated response to add to chat history for this run."
            )
        logger.info(f"[{step_name}] Final history length: {len(final_chat_history)}")

        output_data = WorkflowRunOutput(
            final_response=curated_response_content, chat_history=final_chat_history
        )
        ctx.write_event_to_stream(
            WorkflowStepUpdateEvent(
                step_name=step_name,
                status="completed",
                data={"final_response_snippet": curated_response_content[:50]},
            )
        )
        logger.info(f"[AppWorkFlow] Completed step: {step_name}. Returning StopEvent.")
        return StopEvent(result=output_data)

    async def on_error(self, error: Exception, event: Event) -> Event:
        """
        Handles errors that occur during workflow execution.

        This method is called by LlamaIndex when an unhandled exception
        occurs in one of the workflow steps. It logs the error and
        emits a `WorkflowErrorEvent` to the stream.

        Args:
            error: The exception that occurred.
            event: The event that was being processed when the error occurred.
                   This could be a `WorkflowStepEvent` or another event type.

        Returns:
            The original event that caused the error, which will stop the workflow.
            Alternatively, a new `StopEvent` could be returned to gracefully halt.
        """
        logger.error(
            f"[AppWorkFlow] on_error triggered for event: {type(event).__name__} by error: {error}",
            exc_info=True,
        )
        # Keep rich panel for console visibility if uvicorn shows it, but main log is to file
        console.print(
            Panel(
                f"[bold red]Workflow Error:[/bold red] {error}\n"
                f"[dim]Event leading to error: {type(event).__name__}[/dim]",
                title="[bold red]Workflow Execution Failed[/bold red]",
                border_style="red",
            )
        )

        step_name_from_event = "unknown"
        # Try to infer step name if the event is a WorkflowStepEvent
        # This is a bit of a guess; LlamaIndex internal event types might vary.
        if hasattr(event, "fn_name"):  # Check if it's likely a step event
            step_name_from_event = getattr(event, "fn_name", "unknown_step_function")
        elif hasattr(event, "data") and isinstance(event.data, dict):
            # If we wrapped an original event like ProcessInput
            if "step_name" in event.data:
                step_name_from_event = event.data["step_name"]
            elif type(event) == ProcessInput:
                step_name_from_event = "process_input"
            elif type(event) == RetrieveContext:
                step_name_from_event = "retrieve_context"
            elif type(event) == DynamicPromptBuilt:
                step_name_from_event = "build_dynamic_prompt"
            elif type(event) == LlmResponsesCollected:
                step_name_from_event = "llm_manager"
            elif type(event) == CurationManager:
                step_name_from_event = "curation_manager"

        # Emit a WorkflowErrorEvent to the stream
        # Note: ctx might not be directly available here in LlamaIndex's on_error signature.
        # If ctx.write_event_to_stream is not available, this event won't be streamed.
        # This is a limitation; for critical error streaming, alternative error handling
        # within each step (try-except) might be needed to access ctx.
        # For now, this is an attempt.
        # We'll assume that if on_error is called, the workflow context might be inaccessible for writing.
        # The primary notification will be the log.
        # Let's create the event, but its streaming is best-effort from here.

        error_event_to_emit = WorkflowErrorEvent(
            step_name=step_name_from_event, error_message=str(error)
        )
        # Attempt to write if context is somehow available or if LlamaIndex provides a way
        # For now, we can't call self.context.write_event_to_stream as 'self' here is the workflow,
        # and 'context' is not an attribute of workflow directly in this method.
        # We'd need to get the current run's context.
        # This means the WorkflowErrorEvent might not make it to the SSE stream via this method alone.

        # The LlamaIndex docs show 'on_error' can return a StopEvent.
        # We can wrap our error details in the result of StopEvent if needed,
        # but the goal is to stream WorkflowErrorEvent.

        # A better way for streaming errors is to catch them in each step:
        # try: ...
        # except Exception as e:
        #   ctx.write_event_to_stream(WorkflowErrorEvent(...))
        #   raise e # to still trigger on_error for logging and stopping

        logger.info(
            f"[AppWorkFlow] Emitting WorkflowErrorEvent (best effort from on_error): {error_event_to_emit}"
        )

        # Propagate a StopEvent to ensure the workflow terminates
        # and the client receives a clear signal, even if it's an error state.
        # The actual result might indicate an error.
        logger.info(f"[AppWorkFlow] on_error returning StopEvent.")
        return StopEvent(
            result={"error": str(error), "failed_step": step_name_from_event}
        )
