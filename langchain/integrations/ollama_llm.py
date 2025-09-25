import requests
from langchain.llms.base import LLM

class OllamaLLM(LLM):
    """A custom LLM wrapper for interacting with Ollama models."""

    def __init__(self, endpoint: str):
        super().__init__()
        self.endpoint = endpoint

    @property
    def _llm_type(self) -> str:
        return "ollama"

    def _call(self, prompt: str, stop: list = None) -> str:
        response = requests.post(
            self.endpoint,
            json={"prompt": prompt, "stop": stop},
        )
        response.raise_for_status()
        return response.json().get("response", "")