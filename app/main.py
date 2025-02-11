import asyncio
import json

from app.browser_agent.main import BrowserAgent
from app.logger import base_logger

if __name__ == "__main__":
    logger = base_logger.getChild(__name__)

    prompt = ""
    title = ""
    instruction = ""

    agent = BrowserAgent()
    output = asyncio.run(
        agent.run(prompt=prompt, title=title, instruction=instruction),
    )
    logger.info(json.dumps(output, indent=4))
