import asyncio
from typing import TYPE_CHECKING, cast

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from app.files import FileUtils
from app.logger import base_logger
from app.model_types import AnthropicModelName, ModelProviders, OpenAIModelName

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel


class LLMModel:
    def __init__(self, *, model_name: OpenAIModelName | AnthropicModelName) -> None:
        """Initialize the LLM model."""
        self.logger = base_logger.getChild(self.__class__.__name__)

        model: BaseChatModel
        provider: str

        if isinstance(model_name, OpenAIModelName):
            provider = ModelProviders.OPENAI
            model = ChatOpenAI(model=model_name)
        elif isinstance(model_name, AnthropicModelName):
            provider = ModelProviders.ANTHROPIC
            model = ChatAnthropic(model=model_name)  # pyright: ignore[reportCallIssue]
        else:
            msg = f"Model name {model_name} is not a supported model type"
            self.logger.error(msg)
            raise TypeError(msg)

        self.llm_model_configuration = {"provider": provider, "model": model_name}
        self.model = model

    async def call(self, *, messages: list[HumanMessage]) -> str | list[str | dict]:
        response = await self.model.ainvoke(messages)
        return response.content

    @classmethod
    async def do_ocr(cls, *, images: list[str], prompt: str | None = None) -> str:
        """Method for OCR processing of images.

        Args:
            images (list[str]): List of base64 encoded images.
            prompt (str | None, optional): Prompt for the OCR model. Defaults to "Perform OCR on the following images and extract the text content.".

        Returns:
            str: Extracted text content from the images.

        """
        if prompt is None:
            prompt = "Perform OCR on the following images and extract the text content."
        llm_model = cls(ocr=True)
        image_data = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}} for img in images]
        message = HumanMessage(content=[{"type": "text", "text": prompt}, *image_data])
        return cast(str, await llm_model.call(messages=[message]))


if __name__ == "__main__":
    image = FileUtils.read_image_from_file("Group-1000001696-1.jpeg", return_base64=True)
    if image:
        _response = asyncio.run(LLMModel.do_ocr(images=[cast(str, image)]))
        print(_response)
    else:
        print("No image found")
