import asyncio

from app.agent.main import WebExplorerAgent
from app.logger import base_logger

logger = base_logger.getChild(__name__)


if __name__ == "__main__":
    prompt = "You are web explorer agent"
    title = "Get the latest weather information about Nairobi"
    instruction = "Open google.com and get the latest weather information about Nairobi, Kenya"

    agent = WebExplorerAgent()
    output = asyncio.run(
        agent.run(prompt=prompt, title=title, instruction=instruction),
    )
    logger.info(output)
