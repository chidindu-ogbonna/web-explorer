from typing import TYPE_CHECKING, Literal

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from typing_extensions import TypedDict

from app.core.llm import LLMModel
from app.core.logger import base_logger
from app.core.model_types import AnthropicModelName
from app.operator.browser_agent.main import BrowserAgent, BrowserAgentOutput
from app.operator.configuration import Configuration
from app.operator.state import BrowserState

logger = base_logger.getChild(__name__)

if TYPE_CHECKING:
    from browser_use.browser.browser import BrowserContext

workers = ["authenticate", "act"]


base_system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    f" following workers: {workers}, to achieve a goal."
    " Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)

options = [*workers, "FINISH"]


class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""

    next: Literal[*options]


llm = LLMModel(name=AnthropicModelName.CLAUDE_3_5_LATEST)


async def supervisor(state: BrowserState) -> Command[Literal[*workers, "__end__"]]:
    messages = [{"role": "system", "content": base_system_prompt}] + state["messages"]
    response = await llm.call_with_structured_output(messages=messages, schema=Router)
    # response = llm.with_structured_output(Router).invoke(messages)
    goto = response["next"]
    if goto == "FINISH":
        browser_context: BrowserContext | None = state.get("browser_context")
        if browser_context:
            try:
                await browser_context.close()
                await browser_context.browser.close()
            except Exception:
                logger.exception("Error closing browser")
        goto = END

    return Command(goto=goto, update={"next": goto})


async def authenticate(state: BrowserState) -> Command[Literal["supervisor"]]:
    browser_context: BrowserContext | None = state.get("browser_context")
    browser_agent = BrowserAgent()
    result: BrowserAgentOutput = await browser_agent.run(
        prompt=state["prompt"],
        title=state["title"],
        instruction=state.get("authentication_instruction", ""),
        browser_context=browser_context,
    )
    browser_context = result["browser_context"]
    message = result["message"]
    # NOTE: Support adding images to the message
    # image_data = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}} for img in images]
    # message = HumanMessage(content=[{"type": "text", "text": prompt}, *image_data])

    return Command(
        goto="supervisor",
        update={
            "browser_context": browser_context,
            "messages": [HumanMessage(content=message, name="authenticate")],
        },
    )


async def act(state: BrowserState) -> Command[Literal["supervisor"]]:
    browser_context: BrowserContext | None = state.get("browser_context")
    browser_agent = BrowserAgent()
    result: BrowserAgentOutput = await browser_agent.run(
        prompt=state["prompt"],
        title=state["title"],
        instruction=state.get("act_instruction", ""),
        browser_context=browser_context,
    )
    browser_context = result["browser_context"]
    message = result["message"]
    # NOTE: Support adding images to the message
    # image_data = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}} for img in images]
    # message = HumanMessage(content=[{"type": "text", "text": prompt}, *image_data])
    return Command(
        goto="supervisor",
        update={
            "browser_context": browser_context,
            "messages": [HumanMessage(content=message, name="act")],
        },
    )


builder = StateGraph(BrowserState, config_schema=Configuration)
builder.add_edge(START, "supervisor")
builder.add_node("supervisor", supervisor)
builder.add_node("authenticate", authenticate)
builder.add_node("act", act)
graph = builder.compile(
    interrupt_before=[],  # Add node names here to update state before they're called
    interrupt_after=[],  # Add node names here to update state after they're called
)
graph.name = "Operator"
