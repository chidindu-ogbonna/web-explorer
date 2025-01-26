from datetime import UTC, datetime
from typing import TypedDict, cast

from browser_use import Agent, AgentHistoryList, Browser, BrowserConfig
from browser_use.browser.context import BrowserContext, BrowserContextConfig
from browser_use.browser.views import TabInfo

from app.agent.browser_controller import browser_controller
from app.agent.utils import AGENT_LOG_FOLDER, create_system_prompt_class
from app.files import FileUtils, GIFParams, HistoryMedia
from app.llm import LLMModel
from app.logger import base_logger
from app.model_types import AnthropicModelName


class Output(TypedDict):
    browser_agent_history_gif_url: str | None
    browser_agent_history_gif_video_url: str | None
    browser_agent_history_video_recording_url: str | None
    browser_agent_cookies: list[dict] | None
    image_url: str | None
    message: str | None
    error: str | None


class AgentHistoryMedia(TypedDict):
    screenshot_url: str | None
    gif_url: str | None
    gif_video_url: str | None
    video_recording_url: str | None


class WebExplorerAgent:
    def __init__(self) -> None:
        run_id = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M")
        self.logger = base_logger.getChild(self.__class__.__name__)
        self.CONVERSATION_FILE = f"{AGENT_LOG_FOLDER}/conversation_{run_id}"
        self.COOKIES_FILE = f"{AGENT_LOG_FOLDER}/cookies_{run_id}.json"
        self.AGENT_HISTORY_FILE_NAME = f"{AGENT_LOG_FOLDER}/agent_history_{run_id}"
        self.TRACE_FOLDER = f"{AGENT_LOG_FOLDER}/trace_{run_id}"
        self.AGENT_RECORDING_FOLDER = f"{AGENT_LOG_FOLDER}/agent_recordings_{run_id}"

        self.browser_agent: Agent | None = None
        # NOTE: Swap to any model of your choice here e.g AnthropicModelName | OpenAIModelName
        self.llm_model = LLMModel(model_name=AnthropicModelName.CLAUDE_3_5_LATEST)
        self.model_config = self.llm_model.llm_model_configuration

    def _create_browser_agent(self, *, prompt: str, instruction: str, browser_context: BrowserContext) -> Agent:
        return Agent(
            task=instruction,
            llm=self.llm_model.model,
            save_conversation_path=self.CONVERSATION_FILE,
            browser_context=browser_context,
            system_prompt_class=create_system_prompt_class(prompt=prompt),
            generate_gif=False,  # Generate the gif manually in the _create_history_media method
            controller=browser_controller,
        )

    async def _take_screenshot(self, *, context: BrowserContext) -> str:
        page = await context.get_current_page()
        return await page.screenshot()

    async def _create_history_media(self, *, title: str, context: BrowserContext) -> HistoryMedia | None:
        """Create a GIF from the browser agent's history."""
        if not self.browser_agent:
            msg = "Browser agent is not initialized"
            self.logger.error(msg)
            raise ValueError(msg)
        screenshot_image = await self._take_screenshot(context=context)
        return FileUtils.create_media_from_history_list(
            history_list=self.browser_agent.history,
            screenshots_to_append=[screenshot_image],
            filename=self.AGENT_HISTORY_FILE_NAME,
            output_format=["mp4", "gif"],
            params=GIFParams(
                title_text=title,
                use_logo=False,
            ),
        )

    def _get_open_tabs(self, *, history: AgentHistoryList) -> list[TabInfo]:
        return history.history[-1].state.tabs

    def _read_agent_cookies(self) -> list[dict] | None:
        try:
            return cast(list[dict], FileUtils.read_json_file(self.COOKIES_FILE))
        except Exception:
            msg = f"Error reading cookies from file: {self.COOKIES_FILE}"
            self.logger.error(msg)  # noqa: TRY400
        return None

    def _write_cookies_to_file(self, *, cookies: list[dict] | None) -> None:
        if cookies:
            FileUtils.write_data_to_file(self.COOKIES_FILE, cookies)

    async def run(self, *, prompt: str, title: str, instruction: str) -> Output:
        browser = Browser(config=BrowserConfig(headless=False))

        output: Output
        error: str | None = None
        browser_context_config = BrowserContextConfig(
            minimum_wait_page_load_time=3.0,
            wait_for_network_idle_page_load_time=3.0,
            maximum_wait_page_load_time=10.0,
            # browser_window_size={"width": 1280, "height": 1100},
            # NOTE: using this configuration 1024x768 made it easier to type and stuff like that.
            browser_window_size={"width": 1024, "height": 768},
            cookies_file=self.COOKIES_FILE,
            trace_path=self.TRACE_FOLDER,
            save_recording_path=self.AGENT_RECORDING_FOLDER,
            locale="en-GB",
            highlight_elements=False,
        )
        async with await browser.new_context(config=browser_context_config) as context:
            self.browser_agent = self._create_browser_agent(
                prompt=prompt,
                instruction=instruction,
                browser_context=context,
            )
            browser_agent_history = await self.browser_agent.run()
            browser_agent_cookies = self._read_agent_cookies()
            agent_history_media = await self._create_history_media(context=context, title=title)
            output_message = browser_agent_history.final_result()
            error = browser_agent_history.history[-1].result[-1].error

            output = Output(
                browser_agent_history_gif_url=agent_history_media.get("gif_url"),
                browser_agent_history_gif_video_url=agent_history_media.get("gif_video_url"),
                browser_agent_history_video_recording_url=agent_history_media.get("video_recording_url"),
                message=output_message,
                error=error,
                image_url=agent_history_media.get("screenshot_url"),
                browser_agent_cookies=browser_agent_cookies,
            )

        self.logger.info(output)
        await browser.close()
        return output
