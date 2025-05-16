import asyncio
import json  # For loading data from SimpleChatStore, if needed directly
import os  # For path operations
from typing import List

from dotenv import load_dotenv
from llama_index.core.storage.chat_store import (
    SimpleChatStore,  # Import SimpleChatStore
)
from llama_index.utils.workflow import draw_all_possible_flows
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from phoenix.otel import register
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from backend.app_models import (
    ChatMessage,
    ModelSettings,
    WorkflowRunOutput,
    WorkflowStartEvent,
)
from backend.workflow import AppWorkFlow

"""
This module serves as the command-line interface (CLI) for the Elyse AI application.

It provides a simple, interactive terminal-based chat loop for users to interact 
with the AI workflow. The CLI handles user input, initiates workflow runs, 
and displays the AI's responses.

Key functionalities:
- Initializes and runs the main `AppWorkFlow`.
- Manages chat history across turns within a session using `SimpleChatStore` for persistence.
- Uses Rich library for formatted and styled terminal output.
- Integrates with OpenInference and Phoenix for tracing (optional).

To run the CLI: `uv run -m backend.cli` from the project root.
"""

load_dotenv()

# Define constants for chat store
CHAT_SESSIONS_DIR = "chat_sessions"
DEFAULT_SESSION_ID = "default_session"
DEFAULT_CHAT_STORE_PATH = os.path.join(
    CHAT_SESSIONS_DIR, f"{DEFAULT_SESSION_ID}_store.json"
)


async def run_chat_loop():
    """
    Asynchronously runs the main interactive chat loop in the terminal.

    This function initializes the Rich console, sets up optional tracing with Phoenix,
    instantiates the `AppWorkFlow`, loads/initializes a `SimpleChatStore` for chat history,
    and then enters a loop to:
    1. Get user input.
    2. Create a `WorkflowStartEvent` with the input and current chat state.
    3. Run the workflow.
    4. Display the curated AI response.
    5. Update the chat history in the `SimpleChatStore` and persist it.
    The loop continues until the user types 'quit' or 'exit'.
    """
    console = Console()

    # Ensure chat_sessions directory exists
    os.makedirs(CHAT_SESSIONS_DIR, exist_ok=True)

    # Initialize or load the chat store
    chat_store = SimpleChatStore.from_persist_path(persist_path=DEFAULT_CHAT_STORE_PATH)
    console.print(
        f"[dim]Chat history loaded from/will be saved to: {DEFAULT_CHAT_STORE_PATH}[/dim]"
    )

    # Get current chat history from the store for the default session key
    # This replaces the in-memory current_chat_history list initialization.
    current_chat_history = chat_store.get_messages(DEFAULT_SESSION_ID)
    if current_chat_history:
        console.print(
            f"[dim]Loaded {len(current_chat_history)} messages from existing session.[/dim]"
        )
    else:
        console.print("[dim]No existing session found, starting fresh.[/dim]")

    # Initialize OpenInference and Phoenix tracing (optional)
    # This allows for observing the workflow execution in Phoenix UI.
    try:
        tracer_provider = register(endpoint="http://127.0.0.1:6006/v1/traces")
        LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
        console.print(
            "[dim]Phoenix tracer registered and LlamaIndex instrumented.[/dim]"
        )
    except Exception as e:
        console.print(
            f"[yellow]Warning: Could not initialize Phoenix tracing: {e}[/yellow]"
        )
        console.print(
            "[yellow]Ensure Phoenix server is running at http://127.0.0.1:6006 if tracing is desired.[/yellow]"
        )

    # Draw workflow diagram (optional, useful for understanding the flow)
    # This saves an HTML representation of the AppWorkFlow structure.
    try:
        draw_all_possible_flows(AppWorkFlow, filename="elyse_workflow.html")
        console.print(f"[dim]Workflow diagram saved to elyse_workflow.html[/dim]")
    except Exception as e:
        console.print(f"[yellow]Warning: Could not draw workflow diagram: {e}[/yellow]")

    console.print(
        Panel(
            "[bold green]Elyse Workflow Chat Terminal[/bold green]\\nType 'quit' or 'exit' to end the chat.",
            title="Welcome",
            expand=False,
        )
    )

    # Instantiate the workflow defined in backend/workflow.py
    # `verbose=False` is set to avoid LlamaIndex's default step logging,
    # as we have custom Rich logging in place.
    workflow = AppWorkFlow(timeout=120, verbose=False)

    # Initialize state variables for the chat session managed by the CLI.
    default_model_settings = (
        ModelSettings()
    )  # Uses defaults defined in the Pydantic model

    # Define the list of LLMs to be used for generating candidate responses.
    # In a more advanced setup, this could be loaded from a configuration file or UI.

    # !important: leave these models as is for testing please!
    default_models_to_use = [
        "openrouter/qwen/qwen3-14b",
        "openrouter/gryphe/mythomax-l2-13b",
        "gemini/gemini-2.0-flash-lite",
    ]
    console.print(f"[dim]Using models: {', '.join(default_models_to_use)}[/dim]")

    # Main chat loop
    while True:
        try:
            user_input = console.input("[bold cyan]You[/bold cyan]: ")
            if user_input.lower() in ["quit", "exit"]:
                console.print("[bold yellow]Exiting chat...[/bold yellow]")
                break

            if not user_input.strip():  # Skip empty input
                continue

            # Prepare the start event for the workflow run.
            # This event carries all necessary data for the workflow to process the turn.
            start_event = WorkflowStartEvent(
                user_message=user_input,
                settings=default_model_settings,
                initial_models_to_use=default_models_to_use,
                chat_history=current_chat_history,  # Pass the history from the previous turn
            )

            # Execute the workflow with the start event.
            # The workflow.run() method returns the result from the StopEvent.
            workflow_result = await workflow.run(start_event=start_event)

            # Process the workflow result.
            if isinstance(workflow_result, WorkflowRunOutput):
                final_response = workflow_result.final_response
                # Update the chat history for the next turn with the latest exchange.
                current_chat_history = workflow_result.chat_history

                # Update the chat store with the new history and persist it
                chat_store.set_messages(DEFAULT_SESSION_ID, current_chat_history)
                chat_store.persist(persist_path=DEFAULT_CHAT_STORE_PATH)
                console.print(
                    f"[dim]Chat history updated and saved for session '{DEFAULT_SESSION_ID}'.[/dim]"
                )

                console.print(
                    Panel(
                        (
                            Markdown(
                                final_response
                            )  # Render AI response with Markdown formatting
                            if final_response
                            and final_response != "No response was curated."
                            else "[italic]Elyse chose not to respond this time.[/italic]"
                        ),
                        title="[bold green]Elyse[/bold green]",
                        expand=False,
                        border_style="green",
                    )
                )
            else:
                # Handle unexpected workflow result types, though this path should ideally not be hit
                # if the workflow's StopEvent always returns a WorkflowRunOutput.
                console.print(
                    Panel(
                        f"Unexpected workflow result type: {type(workflow_result)}\\nContent: {str(workflow_result)}",
                        title="[bold red]Unexpected Workflow Result[/bold red]",
                    )
                )
        except Exception as e:
            # General exception handling for the chat loop to prevent crashes.
            console.print(
                f"[bold red]An error occurred in chat loop:[/bold red] {type(e).__name__} - {e}"
            )
            # Consider whether to break or continue depending on error severity in a real application.


if __name__ == "__main__":
    # Entry point for running the CLI application.
    # `asyncio.run()` is used to execute the main asynchronous chat loop function.
    asyncio.run(run_chat_loop())
