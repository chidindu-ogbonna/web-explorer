from browser_use import SystemPrompt

LOG_FOLDER = "logs"

AGENT_LOG_FOLDER = f"{LOG_FOLDER}/agent"


def create_system_prompt_class(prompt: str) -> type[SystemPrompt]:
    class CustomSystemPrompt(SystemPrompt):
        def important_rules(self) -> str:
            existing_rules = super().important_rules()
            custom_rule = f"""
9. MOST IMPORTANT RULE:
{prompt}
"""
            return f"{existing_rules}\n{custom_rule}"

    return CustomSystemPrompt
