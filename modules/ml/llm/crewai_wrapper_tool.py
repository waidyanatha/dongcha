from langchain.agents import Tool
from langchain.chat_models import ChatOllama

class CrewAIWrapperTool():
    def __init__(self, ollama_model: ChatOllama, model_name: str):
        """
        Initialize the tool with the Ollama model.

        Parameters:
        - ollama_model: The model used to run inference.
        - model_name: The name of the model to be used.
        """
        self.ollama_model = ollama_model
        self.model_name = model_name

    def _run(self, query: str) -> str:
        """Process the query using the Ollama model."""
        response = self.ollama_model.invoke(query)  # Assuming `invoke` is the method for generating responses
        return response

    def _arun(self, query: str) -> str:
        """For asynchronous support."""
        raise NotImplementedError("Async run not implemented yet")

    @property
    def name(self) -> str:
        """Required for LangChain Tool class, providing a unique name for the tool."""
        return f"Ollama-{self.model_name}"

    @property
    def is_single_input(self) -> bool:
        """
        Define whether this tool accepts a single input query or multiple.
        Returning True since this tool accepts a single input query.
        """
        return True

    @property
    def description(self) -> str:
        """Description of the tool's functionality."""
        return f"This tool uses the {self.model_name} model from Ollama to generate responses based on the input query."