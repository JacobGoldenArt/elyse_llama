from typing import Any, Dict, List

from llama_index.core.llms import ChatMessage
from llama_index.core.workflow import Context
from llama_index.llms.litellm import LiteLLM
from rich.console import Console

from backend.app_models import (
    LLMCandidateReadyEvent,
    LLMTokenStreamEvent,
    ModelSettings,
    WorkflowErrorEvent,
)

"""
Provides `WorkflowLlmService` for handling interactions with Language Models (LLMs)
using LiteLLM within the Elyse AI application workflow.

This service encapsulates:
-   Concurrent invocation of multiple LLMs specified in `models_to_use`.
-   Streaming of LLM response tokens via `LLMTokenStreamEvent`.
-   Signaling of complete LLM responses (candidates) via `LLMCandidateReadyEvent`.
-   Error handling for individual LLM API calls, reporting errors as part of the
    response candidates and optionally via `WorkflowErrorEvent`.
-   Construction of `LiteLLM` instances with appropriate model settings.
"""

console = Console()


class WorkflowLlmService:
    """
    Manages LLM API calls for the application's workflow using LiteLLM.

    This service is responsible for:
    - Iterating through a list of specified models (`models_to_use`).
    - For each model:
        - Instantiating `LiteLLM` with parameters from `global_model_settings`.
        - Invoking the LLM with the provided system prompt and chat history.
        - Streaming token deltas back to the workflow context via `LLMTokenStreamEvent`.
        - Emitting an `LLMCandidateReadyEvent` when a full response from a model is received
          or if an error occurs for that model.
    - Collecting all responses (successful or error messages) into a list.

    It uses the `Context` object passed from the workflow to stream events directly,
    allowing real-time updates for connected clients.
    """

    async def get_responses_for_workflow(
        self,
        ctx: Context,
        system_prompt: str,
        chat_history: List[ChatMessage],
        models_to_use: List[str],
        global_model_settings: ModelSettings,
    ) -> List[Dict[str, str]]:
        """
        Calls multiple LLMs concurrently, streams tokens, and signals candidate readiness.

        For each model in `models_to_use`:
        1.  Prepares messages by prepending the system prompt to the chat history.
        2.  Initializes a `LiteLLM` client with the specified model and settings.
        3.  Uses `llm_instance.astream_chat()` to get an asynchronous stream of response chunks.
        4.  For each chunk (token delta):
            -   Emits an `LLMTokenStreamEvent` via `ctx.write_event_to_stream()`.
            -   Appends the token to `full_response_for_model`.
        5.  Once the stream for a model is complete:
            -   If a response was received, an `LLMTokenStreamEvent` with `is_final_chunk=True`
                is emitted, followed by an `LLMCandidateReadyEvent` containing the full response.
            -   The full response is added to `collected_responses` with its model name.
        6.  If an error occurs during a model call:
            -   An error message is logged and captured.
            -   An `LLMCandidateReadyEvent` is emitted with the error message as the response.
            -   A `WorkflowErrorEvent` detailing the LLM call failure is streamed.
            -   The error message is added to `collected_responses` with its model name.

        Args:
            ctx: The workflow context, used for streaming events like `LLMTokenStreamEvent`,
                 `LLMCandidateReadyEvent`, and `WorkflowErrorEvent`.
            system_prompt: The system prompt string.
            chat_history: The conversation history (including the latest user message).
            models_to_use: List of model identifiers for LiteLLM (e.g., "gpt-4o-mini").
            global_model_settings: `ModelSettings` for temperature, max_tokens, etc.

        Returns:
            A list of dictionaries, where each dictionary contains:
            {'model_name': str, 'response': str}. The 'response' can be the LLM's
            generated text or an error message if the call to that model failed.
        """
        messages_for_llm = [
            ChatMessage(role="system", content=system_prompt)
        ] + chat_history

        workflow_run_id = await ctx.get("workflow_run_id", "N/A")

        if not models_to_use:
            console.print(
                f"[LLM Service] ({workflow_run_id}) No models specified. Skipping LLM calls.",
                style="dim",
            )
            return []

        collected_responses: List[Dict[str, str]] = []

        for model_name in models_to_use:
            full_response_for_model = ""
            try:
                console.print(
                    f"[LLM Service] ({workflow_run_id}) Attempting model [bold]{model_name}[/bold] with system prompt: '{system_prompt[:35]}...' (streaming)",
                    style="dim",
                )
                llm_instance = LiteLLM(
                    model=model_name,
                    temperature=global_model_settings.temperature,
                    max_tokens=global_model_settings.max_tokens,
                    # TODO: Add other relevant settings from global_model_settings if supported by LiteLLM wrapper
                    # e.g., top_p=global_model_settings.top_p,
                    # frequency_penalty=global_model_settings.frequency_penalty,
                    # presence_penalty=global_model_settings.presence_penalty
                )

                stream = await llm_instance.astream_chat(messages=messages_for_llm)
                token_received = False
                async for chunk in stream:
                    token = chunk.delta
                    if token:
                        token_received = True
                        full_response_for_model += token
                        ctx.write_event_to_stream(
                            LLMTokenStreamEvent(
                                model_name=model_name,
                                token=token,
                                is_final_chunk=False,
                                workflow_run_id=workflow_run_id,
                            )
                        )

                if token_received:
                    ctx.write_event_to_stream(
                        LLMTokenStreamEvent(
                            model_name=model_name,
                            token="",
                            is_final_chunk=True,
                            workflow_run_id=workflow_run_id,
                        )
                    )
                    ctx.write_event_to_stream(
                        LLMCandidateReadyEvent(
                            model_name=model_name,
                            candidate_response=full_response_for_model,
                            workflow_run_id=workflow_run_id,
                        )
                    )
                    collected_responses.append(
                        {"model_name": model_name, "response": full_response_for_model}
                    )
                    console.print(
                        f"[LLM Service] ({workflow_run_id}) Successfully received response from {model_name}. Length: {len(full_response_for_model)}",
                        style="green",
                    )
                else:
                    error_msg = f"LLM Service Info: No content streamed from {model_name}. The stream completed without errors but was empty."
                    console.print(
                        f"[LLM Service] ({workflow_run_id}) {error_msg}", style="yellow"
                    )
                    collected_responses.append(
                        {"model_name": model_name, "response": error_msg}
                    )
                    ctx.write_event_to_stream(
                        LLMCandidateReadyEvent(
                            model_name=model_name,
                            candidate_response=error_msg,
                            workflow_run_id=workflow_run_id,
                        )
                    )
                    ctx.write_event_to_stream(
                        WorkflowErrorEvent(
                            step_name="llm_service",
                            error_message=error_msg,
                            model_name=model_name,
                            workflow_run_id=workflow_run_id,
                        )
                    )

            except Exception as e:
                error_msg = f"LLM Service Error calling {model_name} (streaming): {type(e).__name__} - {str(e)}"
                console.print(
                    f"[LLM Service] ({workflow_run_id}) {error_msg}",
                    style="bold red",
                    exc_info=False,
                )
                collected_responses.append(
                    {"model_name": model_name, "response": error_msg}
                )

                ctx.write_event_to_stream(
                    LLMCandidateReadyEvent(
                        model_name=model_name,
                        candidate_response=error_msg,
                        workflow_run_id=workflow_run_id,
                    )
                )
                ctx.write_event_to_stream(
                    WorkflowErrorEvent(
                        step_name="llm_service_call",
                        error_message=error_msg,
                        model_name=model_name,
                        workflow_run_id=workflow_run_id,
                    )
                )
                ctx.write_event_to_stream(
                    LLMTokenStreamEvent(
                        model_name=model_name,
                        token="",
                        is_final_chunk=True,
                        workflow_run_id=workflow_run_id,
                    )
                )

        return collected_responses
