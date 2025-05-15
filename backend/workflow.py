from typing import List

from llama_index.core.llms import ChatMessage
from llama_index.core.workflow import (
    Context,
    StopEvent,
    Workflow,
    step,
)
from rich.console import Console
from rich.panel import Panel

from backend.app_models import (
    CurationManager,
    DynamicPromptBuilt,
    LlmResponsesCollected,
    ModelSettings,  # For llm_manager default
    ProcessInput,
    RetrieveContext,
    WorkflowRunOutput,
    WorkflowStartEvent,
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

console = Console()


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
    """

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
        await ctx.set("user_message", ev.user_message)
        await ctx.set("model_settings", ev.settings)
        await ctx.set("models_to_use", ev.initial_models_to_use or ["gpt-4o-mini"])

        initial_history = ev.chat_history or []
        # Append the current user's message to the history for this turn.
        updated_history_for_turn = initial_history + [
            ChatMessage(role="user", content=ev.user_message)
        ]
        await ctx.set("chat_history", updated_history_for_turn)

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
        user_message = await ctx.get("user_message", "")
        retrieved_text = "No specific context found for your query."
        # Simple keyword-based context simulation.
        if "weather" in user_message.lower():
            retrieved_text = "The weather is sunny today."

        await ctx.set("context_retrieved", retrieved_text)
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
        context_retrieved = await ctx.get(
            "context_retrieved", ""
        )  # Use context from ctx set by previous step
        current_system_prompt_str = generate_prompt(context_retrieved)
        await ctx.set("current_system_prompt", current_system_prompt_str)

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
        chat_history_from_ctx = await ctx.get("chat_history", [])
        system_prompt = await ctx.get("current_system_prompt", "")
        models_to_use = await ctx.get("models_to_use", [])
        global_model_settings = await ctx.get("model_settings") or ModelSettings()

        llm_service = WorkflowLlmService()

        collected_responses_content = await llm_service.get_responses_for_workflow(
            system_prompt=system_prompt,
            chat_history=chat_history_from_ctx,
            models_to_use=models_to_use,
            global_model_settings=global_model_settings,
        )

        if not models_to_use:
            console.print(
                "[dim]Workflow: No models specified. Skipping LLM calls.[/dim]"
            )

        await ctx.set("response_candidates", collected_responses_content)
        return LlmResponsesCollected(
            responses_collected_count=len(collected_responses_content)
        )

    @step
    async def curation_manager(
        self, ctx: Context, ev: LlmResponsesCollected
    ) -> CurationManager:
        """
        Fifth step: Handles human-in-the-loop curation of LLM responses.

        This step retrieves the list of `response_candidates` from the context,
        displays them to the user (via the CLI in the current implementation),
        and prompts the user to select or paste their preferred response.
        The chosen response is then stored in the context as `curated_ai_response`.

        Args:
            ctx: The shared `GlobalContext` object.
            ev: The `LlmResponsesCollected` event, indicating LLM responses are ready.

        Returns:
            A `CurationManager` event containing the user-curated response string.
        """
        response_candidates = await ctx.get("response_candidates", [])
        chosen_response_string = "No response was curated."

        if not response_candidates:
            console.print(
                "[yellow]Curation Manager: No response candidates to curate.[/yellow]"
            )
        else:
            candidate_display = ""
            for i, candidate_text in enumerate(response_candidates):
                candidate_display += f"[b]{i + 1}.[/b] {str(candidate_text)}\n\n"

            console.print(
                Panel(
                    candidate_display.strip(),
                    title="[bold cyan]--- Response Candidates for Curation ---\[/bold cyan]",
                    border_style="blue",
                    padding=(1, 2),
                )
            )

            while True:
                try:
                    choice_input = console.input(
                        f"[bold]Please select your preferred response (1-{len(response_candidates)}), or 0/Enter to skip/paste:[/bold] "
                    )
                    if not choice_input.strip():
                        manual_paste = console.input(
                            "[bold]Paste the full response here or type 'skip':[/bold] "
                        )
                        if manual_paste.lower() == "skip":
                            console.print("[italic]Curation skipped by user.[/italic]")
                            break
                        elif manual_paste:
                            chosen_response_string = manual_paste
                            console.print(
                                f"[italic]Selected (pasted) response: {chosen_response_string[:70]}...[/italic]"
                            )
                            break
                        else:
                            console.print(
                                "[yellow]No response pasted. Please try again or select by number.[/yellow]"
                            )
                            continue

                    selected_index = int(choice_input) - 1
                    if (
                        choice_input == "0"
                    ):  # Allows explicit skip/paste initiation via '0'
                        manual_paste = console.input(
                            "[bold]Paste the full response here or type 'skip':[/bold] "
                        )
                        if manual_paste.lower() == "skip":
                            console.print("[italic]Curation skipped by user.[/italic]")
                            break
                        elif manual_paste:
                            chosen_response_string = manual_paste
                            console.print(
                                f"[italic]Selected (pasted) response: {chosen_response_string[:70]}...[/italic]"
                            )
                            break
                        else:
                            console.print(
                                "[yellow]No response pasted. Please try again or select by number.[/yellow]"
                            )
                            continue

                    if 0 <= selected_index < len(response_candidates):
                        raw_candidate = response_candidates[selected_index]
                        chosen_response_string = str(
                            raw_candidate
                        )  # Candidates are now strings
                        console.print(
                            f"[italic]Selected response ({selected_index+1}): {chosen_response_string[:70]}...[/italic]"
                        )
                        break
                    else:
                        console.print(
                            f"[yellow]Invalid choice. Please enter a number between 1 and {len(response_candidates)}, or 0/Enter to paste.[/yellow]"
                        )
                except ValueError:
                    console.print(
                        "[red]Invalid input. Please enter a number, or press Enter to paste.[/red]"
                    )
                except Exception as e:
                    console.print(
                        f"[red]An unexpected error occurred during curation: {e}[/red]"
                    )
                    break

        await ctx.set("curated_ai_response", chosen_response_string)
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
            console.print(
                f"[dim]Appended assistant response to history: {curated_response_content[:50]}...[/dim]"
            )
        else:
            console.print(
                "[yellow]No valid curated response to add to chat history for this run.[/yellow]"
            )

        return StopEvent(
            result=WorkflowRunOutput(
                final_response=curated_response_content,
                chat_history=final_chat_history,
            )
        )
