import asyncio
from contextvars import Context
from typing import Any, Dict, List, Optional

import litellm
import pandas as pd
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


# set up a quick llm to test the workflow
# llm = OpenAI(model="gpt-4o-mini") # Removed this line

# Events are user-defined pydantic objects. You control the attributes and any other auxiliary methods.
# In this case, our workflow relies on a single user-defined event.


# marking keys are optional because some wont have values right away and I don't want pydantic errors.

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
    # chat_history: Optional[List[Dict[str, str]]] = None


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
        user_message = await ctx.get("user_message", "")
        chat_history = await ctx.get("chat_history", [])
        system_prompt = await ctx.get("current_system_prompt", "")
        models_to_use = await ctx.get("models_to_use", [])
        global_model_settings = await ctx.get("model_settings") or ModelSettings()

        # not sure how to work with chat history here.

        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_message),
        ]

        if not models_to_use:
            print("Warning: No models specified in models_to_use. Skipping LLM calls.")
            await ctx.set("response_candidates", [])
            return LlmResponsesCollected(responses_collected_count=0)

        candidate_responses = []

        llm = LiteLLM()

        for model_name in models_to_use:
            call_llm = await llm.achat(messages=messages)
            candidate_responses.append(call_llm)

        await ctx.set("response_candidates", candidate_responses)
        return LlmResponsesCollected(responses_collected_count=len(candidate_responses))

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
        current_chat_history = await ctx.get("chat_history", [])
        curated_response_from_event = ev.curated_response

        return StopEvent(
            result=WorkflowRunOutput(
                final_response=curated_response_from_event,
                chat_history=current_chat_history,
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
    workflow = AppWorkFlow(timeout=120, verbose=True)

    # Model settings can be loaded once or made configurable if needed
    default_model_settings = ModelSettings()
    # Models to use can be hardcoded or loaded from a config
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

            # Get the current model settings from our persistent context - This needs to change
            # current_model_settings = await chat_session_context.get("model_settings", None)
            # We will use default_model_settings or allow modification if GUI was present

            start_event = WorkflowStartEvent(
                user_message=user_input,
                settings=default_model_settings,  # Pass the settings
                initial_models_to_use=default_models_to_use,  # Pass models to use
                # chat_history=current_chat_history,  # Pass current chat history
            )

            # Run the workflow with the user's message
            # The context argument is removed as per the warning
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
