import asyncio
import json

from app.agent.main import WebExplorerAgent
from app.logger import base_logger

if __name__ == "__main__":
    logger = base_logger.getChild(__name__)

    prompt = ""
    title = ""
    instruction = ""

    agent = WebExplorerAgent()
    output = asyncio.run(
        agent.run(prompt=prompt, title=title, instruction=instruction),
    )
    logger.info(json.dumps(output, indent=4))
