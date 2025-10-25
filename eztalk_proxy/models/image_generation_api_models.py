from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict

class ImageGenerationRequest(BaseModel):
    model: str = Field(..., description="The model to use for image generation.")
    prompt: str = Field(..., description="A text description of the desired image(s).")
    contents: Optional[List[Dict[str, Any]]] = Field(None, description="The contents for multimodal input.")
    image_size: Optional[str] = Field("1024x1024", description="The size of the generated images.")
    batch_size: Optional[int] = Field(1, description="The number of images to generate.")
    num_inference_steps: Optional[int] = Field(20, description="The number of denoising steps.")
    guidance_scale: Optional[float] = Field(7.5, description="Higher guidance scale encourages to generate images that are closely linked to the text prompt.")
    apiAddress: str = Field(..., description="The address of the downstream API service.")
    apiKey: str = Field(..., description="The API key for the downstream service.")
    provider: Optional[str] = Field("openai compatible", description="The provider/channel type: 'gemini', 'openai compatible', etc.")
    # 新增：与 Google 文档对齐的生成配置（顶层）
    response_modalities: Optional[List[str]] = Field(
        None, description="Preferred response modalities, e.g., ['Image']"
    )
    aspect_ratio: Optional[str] = Field(
        None, description="Output image aspect ratio like '16:9', '3:4', etc."
    )
    # 新增：同时接受 generationConfig（以便从 App 顶层或 generationConfig 两种来源读取）
    generation_config: Optional[Dict[str, Any]] = Field(
        None, alias="generationConfig", description="Raw generation config to pass-through, may contain 'responseModalities' and 'imageConfig.aspectRatio'."
    )
    # 可选：显式前端会话ID，用于后端会话隔离（不提供则回退到IP+model+apiAddress）
    conversation_id: Optional[str] = Field(
        None, alias="conversationId", description="Optional client conversation ID to isolate history between chats."
    )
    # 是否强制使用 Data URI 返回图片（默认关闭，以原图直链/原始字节为主）
    force_data_uri: Optional[bool] = Field(
        False, alias="forceDataUri", description="If true, force images to be returned as Data URI."
    )

class ImageUrl(BaseModel):
    url: str

class Timings(BaseModel):
    inference: int

class ImageGenerationResponse(BaseModel):
    images: List[ImageUrl]
    text: Optional[str] = None
    timings: Timings
    seed: int