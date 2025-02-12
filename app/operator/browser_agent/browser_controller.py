import asyncio

from browser_use.agent.views import ActionResult
from browser_use.browser.context import BrowserContext as BrowserUseBrowserContext
from browser_use.controller.service import Controller
from pydantic import BaseModel

from app.core.files import FileUtils
from app.core.llm import LLMModel
from app.core.logger import base_logger
from app.core.utils import chunkify

logger = base_logger.getChild(__name__)


browser_controller = Controller()


class ActionInputTextCoordinatesParam(BaseModel):
    x: int
    y: int
    text: str


@browser_controller.action(
    """Input text into an input element at the specific coordinates:
This is used as a fallback when the input_text action fails with an Action error such as: 'Action error: Error executing action input_text'
x: The x (pixels from the left edge) coordinates to move the mouse to. Required to determine where to click.
y: The y (pixels from the top edge) coordinates to move the mouse to. Required to determine where to click.
text: The text to input into the page.""",
    param_model=ActionInputTextCoordinatesParam,
)
async def input_text_using_coordinates(
    params: ActionInputTextCoordinatesParam,
    browser: BrowserUseBrowserContext,
) -> ActionResult:
    x = params.x
    y = params.y
    text = params.text
    page = await browser.get_current_page()
    await page.mouse.move(x, y)
    await page.mouse.click(x, y, button="left")
    for chunk in chunkify(text, 50):
        await page.keyboard.type(chunk)
    msg = f"Typed '{text}' at coordinates ({x}, {y})"
    logger.info(msg)
    return ActionResult(extracted_content=msg, include_in_memory=True)


class ActionWaitParam(BaseModel):
    seconds: int


@browser_controller.action(
    """Wait for a specified number of seconds for the result to be generated:
seconds: The number of seconds to wait, minimum 5 seconds""",
    param_model=ActionWaitParam,
)
async def wait(params: ActionWaitParam) -> ActionResult:
    seconds = max(5, params.seconds)  # force the minimum to be 5 seconds
    await asyncio.sleep(seconds)
    msg = f"Waited for {seconds} seconds"
    logger.info(msg)
    return ActionResult(extracted_content=msg, include_in_memory=True)


class ActionReadWebPageContentParam(BaseModel):
    pass


@browser_controller.action("Read web page content", param_model=None)
async def read_page_content(browser: BrowserUseBrowserContext) -> ActionResult:
    # NOTE: evaluate may* not work in headless mode
    page = await browser.get_current_page()
    screenshots = []

    last_height = await page.evaluate("document.documentElement.scrollHeight")
    while True:
        screenshots.append(FileUtils.encode_image_to_base64(await page.screenshot()))
        await page.keyboard.press("PageDown")
        await page.wait_for_load_state()
        new_height = await page.evaluate("document.documentElement.scrollHeight")
        current_position = await page.evaluate("window.scrollY")
        if current_position + await page.evaluate("window.innerHeight") >= last_height:
            break
        last_height = new_height
    response = await LLMModel.do_ocr(
        images=screenshots,
        prompt="You are a webpage reader. The images are screenshots of a webpage. Read the webpage and extract the text content.",
    )
    return ActionResult(extracted_content=response, include_in_memory=True)
