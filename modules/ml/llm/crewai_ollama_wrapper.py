from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.messages import HumanMessage, AIMessage
from pydantic.v1 import BaseModel, Field
from typing import Any, List, Optional

class CrewAIOllamaWrapper(BaseChatModel):
    ollama_model: Any = Field(...)
    model_name: str = Field(...)

    @property
    def _llm_type(self) -> str:
        return "ollama_wrapper"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        input_text = "\n".join(
            [f"{m.type.upper()}: {m.content}" for m in messages if isinstance(m, (HumanMessage, AIMessage))]
        )
        response = self.ollama_model.invoke(input_text)
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=response.content))]
        )

    def bind(self, **kwargs):
        return self
