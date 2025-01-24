from enum import StrEnum


class ReplicateModelName(StrEnum):
    META_LLAMA_3_70B_INSTRUCT = "meta/meta-llama-3-70b-instruct"


class HuggingFaceModelName(StrEnum):
    QWEN_32B_PREVIEW = "Qwen/QwQ-32B-Preview"
    META_LLAMA_3_3_70B_INSTRUCT = "meta-llama/Llama-3.3-70B-Instruct"


class AnthropicModelName(StrEnum):
    CLAUDE_3_5_LATEST = "claude-3-5-sonnet-20241022"
    CLAUDE_3_5_SONNET_2024_10_22 = "claude-3-5-sonnet-20241022"
    CLAUDE_3_5_SONNET_2024_06_20 = "claude-3-5-sonnet-20240620"


class OpenAIModelName(StrEnum):
    GPT_4O = "gpt-4o"  # alias for gpt-4o-2024-08-06
    GPT_4O_2024_11_20 = "gpt-4o-2024-11-20"
    CHATGPT_4O_LATEST = "chatgpt-4o-latest"


class ModelProviders(StrEnum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    REPLICATE = "replicate"
    HUGGINGFACE = "huggingface"


DEFAULT_MODEL_PROVIDER = ModelProviders.ANTHROPIC
DEFAULT_OCR_MODEL_PROVIDER = ModelProviders.OPENAI
