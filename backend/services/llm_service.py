from typing import List

from llama_index.core.llms import ChatMessage
from llama_index.core.workflow import Context
from llama_index.llms.litellm import LiteLLM
from rich.console import Console

from backend.app_models import (
    LLMCandidateReadyEvent,
    LLMTokenStreamEvent,
    ModelSettings,
)

"""
This module provides a service class for handling interactions with Language Models (LLMs)
using LiteLLM within the Elyse AI workflow.

It encapsulates the logic for making concurrent calls to multiple LLMs, 
handling their responses, and managing errors during the API calls.
"""

console = Console()


class WorkflowLlmService:
    """
    A service dedicated to managing LLM API calls for the application's workflow.

    This service uses LlamaIndex's LiteLLM integration to interact with various
    LLM providers. It supports calling multiple models and collecting their responses.
    """

    async def get_responses_for_workflow(
        self,
        ctx: Context,
        system_prompt: str,
        chat_history: List[ChatMessage],
        models_to_use: List[str],
        global_model_settings: ModelSettings,
    ) -> List[str]:
        """
        Calls multiple specified LLMs concurrently with a given system prompt and chat history,
        streaming tokens and signalling when full candidates are ready.

        For each model in `models_to_use`, this method constructs a LiteLLM client,
        sends the conversation history (prepended with the system prompt), and collects
        the textual response. It streams tokens using `LLMTokenStreamEvent` and signals
        a complete response using `LLMCandidateReadyEvent`. It handles potential errors
        during API calls for each model individually, returning either the successful
        response or an error message string.

        Args:
            ctx: The workflow context, used for streaming events.
            system_prompt: The system prompt string to guide the LLMs' behavior.
            chat_history: A list of ChatMessage objects representing the conversation so far.
                          This should include the latest user message for the current turn.
            models_to_use: A list of model identifier strings (e.g., "openai/gpt-4o-mini")
                           that LiteLLM will use.
            global_model_settings: A ModelSettings object containing parameters like
                                   temperature and max_tokens to be applied to each LLM call.

        Returns:
            A list of strings, where each string is either a successful LLM response
            or a formatted error message if the call to that specific model failed.
        """
        messages_for_llm = [
            ChatMessage(role="system", content=system_prompt)
        ] + chat_history

        if not models_to_use:
            console.print(
                "[dim]LLM Service: No models specified. Skipping LLM calls.[/dim]"
            )
            return []

        collected_responses_content: List[str] = []

        for model_name in models_to_use:
            full_response_for_model = ""
            try:
                console.print(
                    f"[dim]LLM Service: Attempting model [bold]{model_name}[/bold] with system prompt: '{system_prompt[:35]}...' (streaming)[/dim]"
                )
                llm_instance = LiteLLM(
                    model=model_name,
                    temperature=global_model_settings.temperature,
                    max_tokens=global_model_settings.max_tokens,
                    # Add other relevant settings from global_model_settings if needed
                    # e.g., top_p=global_model_settings.top_p,
                    # frequency_penalty=global_model_settings.frequency_penalty,
                    # presence_penalty=global_model_settings.presence_penalty
                )

                # Use astream_chat for streaming responses
                stream = await llm_instance.astream_chat(messages=messages_for_llm)

                async for chunk in stream:
                    token = chunk.delta
                    full_response_for_model += token
                    ctx.write_event_to_stream(
                        LLMTokenStreamEvent(
                            model_name=model_name,
                            token=token,
                            is_final_chunk=False,  # This will be true for the last empty chunk from some providers
                        )
                    )

                # After the loop, the full response is assembled.
                # Some streaming providers send a final empty chunk or have a specific way to signal end.
                # We'll assume the loop finishes when the stream ends.
                # Emit final token event to signal completion if necessary, or just rely on LLMCandidateReadyEvent
                # For simplicity, we'll emit one last LLMTokenStreamEvent with is_final_chunk=True if the last token was not empty
                # However, a better approach might be to check a specific attribute on the last chunk if available.
                # For now, we consider the stream finished and proceed to LLMCandidateReadyEvent.
                # If full_response_for_model is not empty, it means we received content.

                if full_response_for_model:  # Check if any token was received.
                    ctx.write_event_to_stream(
                        LLMTokenStreamEvent(
                            model_name=model_name,
                            token="",  # No specific final token, just signaling end
                            is_final_chunk=True,
                        )
                    )
                    ctx.write_event_to_stream(
                        LLMCandidateReadyEvent(
                            model_name=model_name,
                            candidate_response=full_response_for_model,
                        )
                    )
                    collected_responses_content.append(full_response_for_model)
                else:
                    # This case handles if the stream completed but produced no tokens (e.g. model error not caught by exception)
                    error_msg = f"LLM Service Error: No content streamed from {model_name}. The stream completed without errors but was empty."
                    console.print(f"[yellow]{error_msg}[/yellow]")
                    collected_responses_content.append(error_msg)
                    # Emit a candidate ready event with the error if that's desired for UI consistency
                    ctx.write_event_to_stream(
                        LLMCandidateReadyEvent(
                            model_name=model_name,
                            candidate_response=error_msg,  # Send error as candidate
                        )
                    )

            except Exception as e:
                error_msg = f"LLM Service Error calling {model_name} (streaming): {type(e).__name__} - {e}"
                console.print(f"[red]{error_msg}[/red]")
                collected_responses_content.append(error_msg)
                # Emit an event indicating this candidate failed
                ctx.write_event_to_stream(
                    LLMCandidateReadyEvent(
                        model_name=model_name,
                        candidate_response=error_msg,  # Send error as candidate
                    )
                )
                # Also emit a final "empty" token stream event for this failed model if strict clients expect it
                ctx.write_event_to_stream(
                    LLMTokenStreamEvent(
                        model_name=model_name,
                        token="",  # Error occurred
                        is_final_chunk=True,
                    )
                )

        return collected_responses_content
