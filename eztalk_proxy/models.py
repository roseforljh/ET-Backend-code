from pydantic import BaseModel, Field
from typing import List, Dict, Any, Literal, Optional, Union

class OpenAIToolCallFunction(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None

class OpenAIToolCall(BaseModel):
    index: Optional[int] = None
    id: Optional[str] = None
    type: Optional[Literal["function"]] = "function"
    function: OpenAIToolCallFunction

class ApiMessage(BaseModel):
    role: str
    content: Optional[str] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[OpenAIToolCall]] = None

class ChatRequest(BaseModel):
    api_address: Optional[str] = None
    messages: List[ApiMessage]
    provider: Literal["openai", "google"]
    model: str
    api_key: str
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(None, gt=0)
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    use_web_search: Optional[bool] = Field(None, alias="useWebSearch")
    force_custom_reasoning_prompt: Optional[bool] = Field(None, alias="forceCustomReasoningPrompt")
    custom_model_parameters: Optional[Dict[str, Any]] = Field(None, alias="customModelParameters")
    custom_extra_body: Optional[Dict[str, Any]] = Field(None, alias="customExtraBody")