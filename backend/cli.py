import asyncio
import json  # For loading data from SimpleChatStore, if needed directly
import os  # For path operations
from typing import List

from dotenv import load_dotenv
from llama_index.core.storage.chat_store import (
    SimpleChatStore,  # Import SimpleChatStore
)

# from llama_index.utils.workflow import draw_all_possible_flows # Keep if useful, comment out if not used
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from phoenix.otel import register
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from backend.app_models import (
    ChatMessage,  # Though WorkflowStartEvent takes List[ChatMessage], explicit import for clarity
    ModelSettings,
    WorkflowRunOutput,
    WorkflowStartEvent,
)
from backend.workflow import AppWorkFlow

"""
Command-Line Interface (CLI) for the Elyse AI application.

This module provides an interactive terminal-based chat loop for users to
engage with the AI workflow (`AppWorkFlow`). It handles:
- User input and command processing (e.g., 'quit').
- Initialization and execution of the `AppWorkFlow` for each turn.
- Management of chat history using `SimpleChatStore` for persistence across sessions.
  Chat sessions are stored in the `chat_sessions` directory.
- Display of AI responses, formatted using the Rich library for an enhanced
  terminal experience (e.g., Markdown rendering, styled panels).
- Optional integration with OpenInference and Phoenix for tracing workflow execution,
  aiding in debugging and observability.
- Optional generation of a workflow diagram (`elyse_workflow.html`) using LlamaIndex utilities.

To run the CLI:
  From the project root directory (`elyse_llama`), execute:
  `uv run -m backend.cli`
"""

load_dotenv()  # Load environment variables from .env file, if present

# --- Constants --- #
CHAT_SESSIONS_DIR = "chat_sessions"
DEFAULT_SESSION_ID = "cli_session"  # More specific session ID for CLI
DEFAULT_CHAT_STORE_PATH = os.path.join(
    CHAT_SESSIONS_DIR, f"{DEFAULT_SESSION_ID}_store.json"
)


async def run_chat_loop():
    """
    Asynchronously runs the main interactive chat loop in the terminal.

    Initializes the Rich console, sets up optional OpenInference/Phoenix tracing,
    instantiates the `AppWorkFlow`, and manages chat history persistence using
    `SimpleChatStore`. The loop prompts for user input, runs the workflow,
    displays the AI's response, and saves the updated chat history.

    The loop continues until the user types 'quit' or 'exit'.
    """
    console = Console()

    # Ensure chat_sessions directory exists for storing chat history
    os.makedirs(CHAT_SESSIONS_DIR, exist_ok=True)

    # Initialize or load the chat store for the default CLI session
    try:
        chat_store = SimpleChatStore.from_persist_path(
            persist_path=DEFAULT_CHAT_STORE_PATH
        )
        current_chat_history: List[ChatMessage] = chat_store.get_messages(
            DEFAULT_SESSION_ID
        )
        if current_chat_history:
            console.print(
                f"[dim]Chat history loaded from: [cyan]{DEFAULT_CHAT_STORE_PATH}[/cyan] ({len(current_chat_history)} messages)[/dim]"
            )
        else:
            console.print(
                f"[dim]No existing chat history at [cyan]{DEFAULT_CHAT_STORE_PATH}[/cyan]. Starting new session.[/dim]"
            )
    except FileNotFoundError:
        console.print(
            f"[dim]Chat store file [cyan]{DEFAULT_CHAT_STORE_PATH}[/cyan] not found. Initializing new chat store for session '{DEFAULT_SESSION_ID}'.[/dim]"
        )
        chat_store = SimpleChatStore()
        current_chat_history = []
    except Exception as e:
        console.print(
            f"[bold red]Error loading chat store:[/bold red] {e}. Starting with an empty history.",
            style="red",
        )
        chat_store = SimpleChatStore()
        current_chat_history = []

    # --- Optional: OpenInference and Phoenix Tracing Setup ---
    # This allows observing workflow execution in the Phoenix UI (http://127.0.0.1:6006).
    try:
        tracer_provider = register(endpoint="http://127.0.0.1:6006/v1/traces")
        LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
        console.print(
            "[dim]Phoenix tracer registered. View traces at [link=http://127.0.0.1:6006]http://127.0.0.1:6006[/link][/dim]"
        )
    except Exception as e:
        console.print(
            f"[yellow]Warning: Could not initialize Phoenix tracing: {e}. Tracing will be disabled.[/yellow]"
        )
        console.print(
            "[yellow]  (Ensure Phoenix server is running: `phoenix server serve`)[/yellow]"
        )

    # --- Optional: Draw Workflow Diagram ---
    # Generates an HTML representation of the AppWorkFlow structure for visualization.
    # try:
    #     draw_all_possible_flows(AppWorkFlow, filename="elyse_workflow.html")
    #     console.print(f"[dim]Workflow diagram saved to [cyan]elyse_workflow.html[/cyan][/dim]")
    # except Exception as e:
    #     console.print(f"[yellow]Warning: Could not draw workflow diagram: {e}[/yellow]")

    console.print(
        Panel(
            "[bold green]Elyse AI Chat Terminal[/bold green]\nType [magenta]'quit'[/magenta] or [magenta]'exit'[/magenta] to end the chat.",
            title="Welcome",
            expand=False,
            border_style="blue",
        )
    )

    # Instantiate the workflow.
    # `verbose=False` prevents LlamaIndex's default step logging, using our custom logs instead.
    workflow = AppWorkFlow(timeout=180, verbose=False)  # Increased timeout slightly

    # Default settings for the CLI session. Can be modified or made configurable.
    default_model_settings = ModelSettings()
    # Example: override specific settings if needed for CLI
    # default_model_settings.temperature = 0.5

    # LLMs to use for generating candidate responses in the CLI.
    # This list can be customized for different testing scenarios.
    default_models_to_use = [
        "gpt-4o-mini",  # Primary model
        "claude-3-haiku-20240307",  # A strong alternative
        "openrouter/google/gemini-flash-1.5",  # Another option for variety
    ]
    console.print(
        f"[dim]Using models for responses: [bold cyan]{', '.join(default_models_to_use)}[/bold cyan][/dim]"
    )

    # --- Main Chat Loop --- #
    while True:
        try:
            user_input = console.input("[bold steel_blue]You[/bold steel_blue]: ")
            if user_input.lower() in ["quit", "exit"]:
                console.print(
                    "[bold yellow]Exiting Elyse AI Chat... Goodbye![/bold yellow]"
                )
                break

            if not user_input.strip():
                continue  # Skip empty input and re-prompt

            # Prepare the WorkflowStartEvent for the current turn.
            start_event = WorkflowStartEvent(
                user_message=user_input,
                settings=default_model_settings,
                initial_models_to_use=default_models_to_use,
                chat_history=current_chat_history,  # Pass the history from the previous turn
                # workflow_run_id is not explicitly set here, as the CLI workflow path
                # might not always require external async callbacks for curation like the API does.
                # However, if CLI were to simulate API-style curation, it would need a run_id.
                # For now, AppWorkFlow can handle a missing workflow_run_id for non-API curation.
            )

            console.print("[italic dim]Elyse is thinking...[/italic dim]")
            # Execute the workflow. `workflow.run()` is synchronous but returns a handler.
            # We then `await` the handler to get the final result from the StopEvent.
            handler = workflow.run(start_event=start_event)
            workflow_result: WorkflowRunOutput = await handler

            if isinstance(workflow_result, WorkflowRunOutput):
                final_response = workflow_result.final_response
                current_chat_history = (
                    workflow_result.chat_history
                )  # Update history for the next turn

                # Persist the updated chat history to the store
                chat_store.set_messages(DEFAULT_SESSION_ID, current_chat_history)
                chat_store.persist(persist_path=DEFAULT_CHAT_STORE_PATH)
                # console.print(f"[dim]Chat history updated and saved for session '{DEFAULT_SESSION_ID}'.[/dim]")

                console.print(
                    Panel(
                        (
                            Markdown(final_response)
                            if final_response
                            and final_response.strip()
                            and final_response
                            not in [
                                "No response was curated or curation was skipped.",
                                "Curation was cancelled.",
                            ]
                            else "[italic dim]Elyse ponders in silence.[/italic dim]"
                        ),
                        title="[bold dodger_blue1]Elyse[/bold dodger_blue1]",
                        expand=False,
                        border_style="dodger_blue1",
                        padding=(1, 2),
                    )
                )
            else:
                # This case should ideally not be reached if the workflow's StopEvent
                # always results in a WorkflowRunOutput (or an error handled by on_error).
                console.print(
                    Panel(
                        f"Unexpected workflow result type: {type(workflow_result)}\nContent: {str(workflow_result)}",
                        title="[bold red]Internal Workflow Error[/bold red]",
                    )
                )
        except KeyboardInterrupt:
            console.print("\n[bold yellow]Chat interrupted. Exiting...[/bold yellow]")
            break
        except Exception as e:
            console.print_exception(show_locals=True)  # Rich traceback for debugging
            console.print(
                f"[bold red]An unexpected error occurred in the chat loop: {type(e).__name__}. Please check logs.[/bold red]"
            )
            # Decide whether to break or offer to continue. For robustness, let's try to continue.
            # break # Uncomment to exit on any error


if __name__ == "__main__":
    """Entry point for running the CLI application."""
    try:
        asyncio.run(run_chat_loop())
    except KeyboardInterrupt:
        Console().print(
            "\n[bold yellow]Application terminated by user (Ctrl+C).[/bold yellow]"
        )
    except Exception as e:
        Console().print(
            f"[bold red]Fatal error during application startup or shutdown: {e}[/bold red]"
        )
        Console().print_exception(show_locals=False)
