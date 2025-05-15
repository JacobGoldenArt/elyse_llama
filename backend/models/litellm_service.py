import litellm
from dotenv import load_dotenv
from llama_index.core.llms import ChatMessage, ChatResponse
from llama_index.llms.litellm import LiteLLM
from llama_index.llms.openai import OpenAI

load_dotenv()


class LitellmService:
    # __init__ removed as self.llm was not used by methods creating new LiteLLM instances.

