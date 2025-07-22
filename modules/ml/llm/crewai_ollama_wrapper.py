#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import Field, model_validator
from typing import Any, Dict, List, Optional

class CrewAIOllamaWrapper(BaseChatModel):
    ollama_model: Any = Field(..., exclude=True)  # Prevent serialization
    model_name: str = Field(...)
    temperature: float = Field(default=0.1)
    base_url: str = Field(default="http://localhost:11434")
    
    @model_validator(mode='before')
    @classmethod
    def validate_model(cls, values):
        """Extract critical params from the underlying model"""
        if isinstance(values, dict) and 'ollama_model' in values:
            model = values['ollama_model']
            # Extract temperature if not explicitly provided
            if 'temperature' not in values:
                values['temperature'] = getattr(model, 'temperature', 0.1)
            # Extract base_url if not explicitly provided
            if 'base_url' not in values:
                values['base_url'] = getattr(model, 'base_url', "http://localhost:11434")
        return values

    @property
    def _llm_type(self) -> str:
        return "ollama"  # Must match CrewAI's expected types

    def _get_model_params(self) -> Dict:
        """Expose params for CrewAI serialization"""
        return {
            'model': self.model_name,
            'temperature': self.temperature,
            'base_url': self.base_url
        }

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        """Generate response from ollama model"""
        try:
            # For single string input (common case)
            if isinstance(messages, str):
                response = self.ollama_model.invoke(messages)
            # For message list input
            elif isinstance(messages, list):
                # Convert messages to simple string format that Ollama expects
                formatted_text = "\n".join([
                    f"Human: {m.content}" if isinstance(m, HumanMessage) 
                    else f"Assistant: {m.content}" if isinstance(m, AIMessage)
                    else str(m.content)
                    for m in messages
                ])
                response = self.ollama_model.invoke(formatted_text)
            else:
                # Fallback for other formats
                response = self.ollama_model.invoke(str(messages))
            
            # Handle different response formats
            if hasattr(response, 'content'):
                content = response.content
            elif isinstance(response, str):
                content = response
            else:
                content = str(response)
                
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content=content))]
            )
            
        except Exception as e:
            # Return error message as response to avoid breaking the flow
            error_content = f"Error generating response: {str(e)}"
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content=error_content))]
            )

    def bind(self, **kwargs):
        """Proper binding for CrewAI integration"""
        new_params = {**self._get_model_params(), **kwargs}
        return type(self)(
            ollama_model=self.ollama_model,
            model_name=self.model_name,
            temperature=new_params.get('temperature', self.temperature),
            base_url=new_params.get('base_url', self.base_url)
        )

    def __repr__(self):
        return f"CrewAIOllamaWrapper(model='{self.model_name}')"
    
    def __str__(self):
        return f"CrewAIOllamaWrapper(model='{self.model_name}', temperature={self.temperature})"
    
    # Additional methods that CrewAI might expect
    def predict(self, text: str) -> str:
        """Simple prediction method for compatibility"""
        result = self._generate([HumanMessage(content=text)])
        return result.generations[0].message.content
    
    def invoke(self, input_data):
        """Invoke method for direct compatibility"""
        if isinstance(input_data, str):
            return self.predict(input_data)
        elif isinstance(input_data, list):
            result = self._generate(input_data)
            return result.generations[0].message
        else:
            return self.ollama_model.invoke(input_data)

# from langchain_core.language_models import BaseChatModel
# from langchain_core.outputs import ChatResult, ChatGeneration
# from langchain_core.messages import HumanMessage, AIMessage
# from pydantic import BaseModel, Field
# # from pydantic.v1 import BaseModel, Field
# from typing import Any, List, Optional

# class CrewAIOllamaWrapper(BaseChatModel):
#     ollama_model: Any = Field(...)
#     model_name: str = Field(...)

#     @property
#     def _llm_type(self) -> str:
#         return "ollama_wrapper"

#     def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
#         input_text = "\n".join(
#             [f"{m.type.upper()}: {m.content}" for m in messages if isinstance(m, (HumanMessage, AIMessage))]
#         )
#         response = self.ollama_model.invoke(input_text)
#         return ChatResult(
#             generations=[ChatGeneration(message=AIMessage(content=response.content))]
#         )

#     def bind(self, **kwargs):
#         return self
# from langchain_core.language_models import BaseChatModel
# from langchain_core.outputs import ChatResult, ChatGeneration
# from langchain_core.messages import HumanMessage, AIMessage
# from pydantic import Field, model_validator
# from typing import Any, Dict, List, Optional

# class CrewAIOllamaWrapper(BaseChatModel):
#     ollama_model: Any = Field(..., exclude=True)  # Prevent serialization
#     model_name: str = Field(...)
#     temperature: float = Field(default=0.1)
#     base_url: str = Field(default="http://localhost:11434")
    
#     @model_validator(mode='before')
#     def validate_model(cls, values):
#         """Extract critical params from the underlying model"""
#         if 'ollama_model' in values:
#             model = values['ollama_model']
#             values['temperature'] = getattr(model, 'temperature', 0.1)
#             values['base_url'] = getattr(model, 'base_url', "http://localhost:11434")
#         return values

#     @property
#     def _llm_type(self) -> str:
#         return "ollama"  # Must match CrewAI's expected types

#     def _get_model_params(self) -> Dict:
#         """Expose params for CrewAI serialization"""
#         return {
#             'model': self.model_name,
#             'temperature': self.temperature,
#             'base_url': self.base_url
#         }

#     def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
#         # Convert messages to Ollama format
#         formatted = [
#             {"role": "user" if isinstance(m, HumanMessage) else "assistant", 
#              "content": m.content}
#             for m in messages
#         ]
        
#         response = self.ollama_model.invoke(formatted)
#         return ChatResult(
#             generations=[ChatGeneration(message=AIMessage(content=response.content))]
#         )

#     def bind(self, **kwargs):
#         """Proper binding for CrewAI integration"""
#         new_params = {**self._get_model_params(), **kwargs}
#         return type(self)(
#             ollama_model=self.ollama_model,
#             model_name=self.model_name,
#             **new_params
#         )

#     def __repr__(self):
#         return f"CrewAIOllamaWrapper(model='{self.model_name}')"