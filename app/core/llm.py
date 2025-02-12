import asyncio
from typing import TYPE_CHECKING, TypedDict, cast

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from app.core.files import FileUtils
from app.core.logger import base_logger
from app.core.model_types import AnthropicModelName, ModelProviders, OpenAIModelName

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel


class LLMModel:
    def __init__(self, *, name: OpenAIModelName | AnthropicModelName) -> None:
        """Initialize the LLM model."""
        self.logger = base_logger.getChild(self.__class__.__name__)

        model: BaseChatModel
        provider: str

        if isinstance(name, OpenAIModelName):
            provider = ModelProviders.OPENAI
            model = ChatOpenAI(model=name)
        elif isinstance(name, AnthropicModelName):
            provider = ModelProviders.ANTHROPIC
            model = ChatAnthropic(model=name)  # pyright: ignore[reportCallIssue]
        else:
            msg = f"Model name {name} is not a supported model type"
            self.logger.error(msg)
            raise TypeError(msg)

        self.llm_model_configuration = {"provider": provider, "model": name}
        self.model = model

    async def call(self, *, messages: list[HumanMessage]) -> str | list[str | dict]:
        response = await self.model.ainvoke(messages)
        return response.content

    async def call_with_structured_output(self, *, messages: list[HumanMessage], schema: type[TypedDict]) -> dict:
        return await self.model.with_structured_output(schema).ainvoke(messages)

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
        llm_model = cls(name=OpenAIModelName.GPT_4O)
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
