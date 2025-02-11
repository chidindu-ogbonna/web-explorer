"""Define the state structures for the agent."""

from browser_use.browser.browser import BrowserContext
from langgraph.graph import MessagesState


class BrowserState(MessagesState):
    browser_context: BrowserContext | None
    image: str | None  # b64 encoded screenshot
    title: str
    prompt: str
    authentication_instruction: str
    act_instruction: str
    next: str
