import asyncio

# from contextvars import Context # Unused import
from typing import Any, Dict, List, Optional

import litellm

# import pandas as pd # Unused import
from dotenv import load_dotenv
from litellm import BaseModel
from llama_index.core.llms import ChatMessage
from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from llama_index.llms.litellm import LiteLLM
from llama_index.utils.workflow import draw_all_possible_flows
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from phoenix.otel import register
from rich.console import Console
from rich.panel import Panel

from backend.prompts.prompt_manager import generate_prompt

load_dotenv()

# Settings and Context Models


class AppSettings(BaseModel):
    tts_enabled: Optional[bool] = False
    stt_enabled: Optional[bool] = False
    stt_model: Optional[str] = "google"
    embedding_model: Optional[str] = "openai"
    sfw_mode: Optional[bool] = False


class ModelSettings(BaseModel):
    model: Optional[str] = "gpt-4o-mini"
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = 1000
    top_p: Optional[float] = 1.0
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0


class WorkflowStartEvent(StartEvent):
    user_message: str
    settings: ModelSettings
    initial_models_to_use: Optional[List[str]] = None
    chat_history: Optional[List[ChatMessage]] = None


class GlobalContext(Context):
    user_message: Optional[str] = ""
    chat_history: Optional[List[ChatMessage]] = []
    context_needed: Optional[bool] = False
    context_retrieved: Optional[str] = ""
    app_settings: Optional[AppSettings] = AppSettings()
    model_settings: Optional[ModelSettings] = ModelSettings()
    models_to_use: Optional[List[str]] = []
    response_candidates: Optional[List[str]] = []
    current_system_prompt: Optional[str] = ""
    currated_ai_response: Optional[str] = ""


# Events


class ProcessInput(Event):
    first_output: str


class RetrieveContext(Event):
    # this is a placeholder for the context retrieval step
    context_retrieved: str


class DynamicPromptBuilt(Event):
    prompt_updated: str


class LlmResponsesCollected(Event):
    responses_collected_count: int


class CurationManager(Event):
    curated_response: str


# Add this Pydantic model for the workflow's final output
class WorkflowRunOutput(BaseModel):
    final_response: str
    chat_history: List[ChatMessage]


# Workflow Definition (Restoring this class)
class AppWorkFlow(Workflow):
    """
    This is the main workflow class.
    The flow is as follows:
    1. ProcessInput: Receive the user message
    2. RetrieveContext: Retrieve the context using the user message and or chat_history if there is any. (This is a placeholder for the context retrieval step)
    3. BuildDynamicPrompt: Build the prompt using static system prompt and inject the latest context and the user message.
    4. LllmManager: Using the global context key: models_to_use, we will concurently call the llms and store the responses in the global context key: response_candidates.
    5. CurationManager: We use Human in the loop to curate the response candidates and store the currated response in the global context key: currated_ai_response.
    6. This loops us back around to the user composing their next message. The chat_history is updated for the next turn.
    """

    @step
    async def process_input(self, ctx: Context, ev: WorkflowStartEvent) -> ProcessInput:
        await ctx.set("user_message", ev.user_message)
        await ctx.set("model_settings", ev.settings)
        await ctx.set("models_to_use", ev.initial_models_to_use or ["gpt-4o-mini"])

        # Initialize chat history in context from event, or as empty list
        initial_history = ev.chat_history or []
        # Append current user message to the history for this turn
        # Ensure role is 'user' for user messages.
        updated_history_for_turn = initial_history + [
            ChatMessage(role="user", content=ev.user_message)
        ]
        await ctx.set("chat_history", updated_history_for_turn)

        return ProcessInput(
            first_output=f"First step: Received user message: {ev.user_message}"
        )

    @step
    async def retrieve_context(self, ctx: Context, ev: ProcessInput) -> RetrieveContext:
        user_message = await ctx.get("user_message", "")
        retrieved_text = "No specific context found for your query."
        # example of context retrieval
        if "weather" in user_message.lower():
            retrieved_text = "The weather is sunny today."

        await ctx.set("context_retrieved", retrieved_text)
        return RetrieveContext(context_retrieved=retrieved_text)

    @step
    async def build_dynamic_prompt(
        self, ctx: Context, ev_rc: RetrieveContext
    ) -> DynamicPromptBuilt:
        context_retrieved = await ctx.get("context_retrieved", "")
        make_prompt = await ctx.set(
            "current_system_prompt", generate_prompt(context_retrieved)
        )
        final_prompt = await ctx.get("current_system_prompt", "")

        return DynamicPromptBuilt(prompt_updated=f"Prompt updated: {final_prompt}")

    @step
    async def llm_manager(
        self, ctx: Context, ev: DynamicPromptBuilt
    ) -> LlmResponsesCollected:
        user_message = await ctx.get(
            "user_message", ""
        )  # For logging or if needed elsewhere
        chat_history_from_ctx = await ctx.get(
            "chat_history", []
        )  # This now includes the latest user message
        system_prompt = await ctx.get("current_system_prompt", "")
        models_to_use = await ctx.get("models_to_use", [])
        global_model_settings = await ctx.get("model_settings") or ModelSettings()

        # Construct messages for LLM
        # The chat_history_from_ctx already contains the most recent user message thanks to process_input
        messages_for_llm = [
            ChatMessage(role="system", content=system_prompt)
        ] + chat_history_from_ctx

        if not models_to_use:
            print("Warning: No models specified in models_to_use. Skipping LLM calls.")
            await ctx.set("response_candidates", [])
            return LlmResponsesCollected(responses_collected_count=0)

        collected_responses_content: List[str] = []

        for model_name in models_to_use:
            try:
                print(
                    f"Attempting to call model: {model_name} with system prompt: '{system_prompt[:50]}...'"
                )
                # For debugging history, print the messages being sent
                # print(f"Full messages to {model_name}: {messages_for_llm}")

                llm_instance = LiteLLM(
                    model=model_name,
                    temperature=global_model_settings.temperature,
                    max_tokens=global_model_settings.max_tokens,
                )

                response_obj = await llm_instance.achat(messages=messages_for_llm)

                if (
                    response_obj
                    and response_obj.message
                    and response_obj.message.content
                ):
                    collected_responses_content.append(response_obj.message.content)
                    print(
                        f"Response from {model_name}: {response_obj.message.content[:100]}..."
                    )
                else:
                    error_msg = f"Error: No content received from {model_name}."
                    if (
                        response_obj
                        and hasattr(response_obj, "raw")
                        and response_obj.raw
                    ):
                        error_msg += f" Raw: {str(response_obj.raw)[:100]}"
                    print(error_msg)
                    collected_responses_content.append(error_msg)
            except Exception as e:
                error_msg = f"Error calling {model_name}: {type(e).__name__} - {e}"
                print(error_msg)
                collected_responses_content.append(error_msg)

        await ctx.set("response_candidates", collected_responses_content)
        return LlmResponsesCollected(
            responses_collected_count=len(collected_responses_content)
        )

    @step
    async def curation_manager(
        self, ctx: Context, ev: LlmResponsesCollected
    ) -> CurationManager:
        response_candidates = await ctx.get("response_candidates", [])
        chosen_response_string = "No response was curated."

        for response in response_candidates:
            print(response)

        # ask the user to select a response

        while True:
            choice = input("Paste the best response here: ")

            if choice:
                chosen_response_string = choice
                break
            else:
                print("Please enter a valid response.")

        await ctx.set("currated_ai_response", chosen_response_string)
        return CurationManager(curated_response=chosen_response_string)

    @step
    async def stop_event(self, ctx: Context, ev: CurationManager) -> StopEvent:
        # Get chat history (which includes the user's message for this turn from process_input)
        chat_history_for_this_run = await ctx.get("chat_history", [])
        curated_response_content = ev.curated_response

        final_chat_history = (
            chat_history_for_this_run  # Start with history up to user's message
        )

        if (
            curated_response_content
            and curated_response_content != "No response was curated."
        ):
            # Append the assistant's curated response to the history. Ensure role is 'assistant'.
            final_chat_history = final_chat_history + [
                ChatMessage(role="assistant", content=curated_response_content)
            ]
            print(
                f"Appended assistant response to history: {curated_response_content[:50]}..."
            )
        else:
            print("No valid curated response to add to chat history for this run.")

        # The StopEvent will carry the fully updated chat history for this turn
        return StopEvent(
            result=WorkflowRunOutput(
                final_response=curated_response_content,
                chat_history=final_chat_history,
            )
        )


async def main():
    console = Console()
    tracer_provider = register(endpoint="http://127.0.0.1:6006/v1/traces")
    LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
    draw_all_possible_flows(AppWorkFlow, filename="elyse_workflow.html")
    console.print(
        Panel(
            "[bold green]Elyse Workflow Chat Terminal[/bold green]\nType 'quit' or 'exit' to end the chat.",
            title="Welcome",
            expand=False,
        )
    )

    # Instantiate the workflow
    # Increased timeout for potentially multiple LLM calls and human input
    workflow = AppWorkFlow(timeout=120, verbose=False)

    # Initialize main's local state variable for chat history
    current_chat_history: List[ChatMessage] = []
    default_model_settings = ModelSettings()
    default_models_to_use = [
        "openai/gpt-4o-mini",
        "gemini/gemini-2.0-flash-lite",
        "anthropic/claude-3-5-sonnet-latest",
        "openrouter/qwen/qwen3-14b",
    ]

    # Draw the workflow visualization (optional, uncomment if needed)
    # from llama_index.utils.workflow import draw_all_possible_flows
    # draw_all_possible_flows(workflow, filename="elyse_workflow.html")

    while True:
        try:
            user_input = console.input("[bold cyan]You[/bold cyan]: ")
            if user_input.lower() in ["quit", "exit"]:
                console.print("[bold yellow]Exiting chat...[/bold yellow]")
                break

            if not user_input.strip():
                continue

            start_event = WorkflowStartEvent(
                user_message=user_input,
                settings=default_model_settings,
                initial_models_to_use=default_models_to_use,
                chat_history=current_chat_history,  # Pass current chat history
            )

            workflow_result = await workflow.run(start_event=start_event)

            if isinstance(workflow_result, WorkflowRunOutput):
                final_response = workflow_result.final_response
                current_chat_history = (
                    workflow_result.chat_history
                )  # Update main's chat_history

                console.print(
                    Panel(
                        (
                            final_response
                            if final_response
                            else "Elyse chose not to respond."
                        ),
                        title="[bold green]Elyse[/bold green]",
                        expand=False,
                        border_style="green",
                    )
                )
            else:
                # Handle unexpected result type if necessary
                console.print(
                    Panel(
                        str(workflow_result),
                        title="[bold red]Unexpected Workflow Result[/bold red]",
                    )
                )
                # Potentially reset chat history or handle error

        except Exception as e:
            console.print(f"[bold red]An error occurred:[/bold red] {e}")
            # Optionally, decide if the loop should break or continue on error
            # For robustness in a real app, more specific error handling would be needed.


if __name__ == "__main__":
    asyncio.run(main())
