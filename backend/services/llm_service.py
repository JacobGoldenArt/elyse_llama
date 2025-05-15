from typing import List

from llama_index.core.llms import ChatMessage
from llama_index.llms.litellm import LiteLLM
from rich.console import Console

from backend.app_models import ModelSettings

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
        system_prompt: str,
        chat_history: List[ChatMessage],
        models_to_use: List[str],
        global_model_settings: ModelSettings,
    ) -> List[str]:
        """
        Calls multiple specified LLMs concurrently with a given system prompt and chat history.

        For each model in `models_to_use`, this method constructs a LiteLLM client,
        sends the conversation history (prepended with the system prompt), and collects
        the textual response. It handles potential errors during API calls for each model
        individually, returning either the successful response or an error message string.

        Args:
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
            try:
                console.print(
                    f"[dim]LLM Service: Attempting model [bold]{model_name}[/bold] with system prompt: '{system_prompt[:35]}...'[/dim]"
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

                response_obj = await llm_instance.achat(messages=messages_for_llm)

                if (
                    response_obj
                    and response_obj.message
                    and response_obj.message.content
                ):
                    collected_responses_content.append(response_obj.message.content)
                    # console.print(f"[dim]LLM Service: Response from [bold]{model_name}[/bold]: {response_obj.message.content[:70]}...[/dim]") # Kept dim for now, candidates shown later
                else:
                    error_msg = (
                        f"LLM Service Error: No content received from {model_name}."
                    )
                    if (
                        response_obj
                        and hasattr(response_obj, "raw")
                        and response_obj.raw
                    ):
                        error_msg += f" Raw: {str(response_obj.raw)[:100]}"
                    console.print(f"[yellow]{error_msg}[/yellow]")
                    collected_responses_content.append(error_msg)
            except Exception as e:
                error_msg = (
                    f"LLM Service Error calling {model_name}: {type(e).__name__} - {e}"
                )
                console.print(f"[red]{error_msg}[/red]")
                collected_responses_content.append(error_msg)

        return collected_responses_content
