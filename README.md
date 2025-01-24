# Web Explorer Agent

This is a web explorer agent that can be used to explore the web and get information about a given topic.

## Usage

1. Configure your `prompt`, `title` & `instruction` in the `app/main.py` file

2. Optional: Configure your `model_name` in the `app/main.py` line 45 file. You can swap to any model of your choice here e.g AnthropicModelName | OpenAIModelName

    By default, the model is set to `AnthropicModelName.CLAUDE_3_5_LATEST`

    ```python
    self.llm_model = LLMModel(model_name=AnthropicModelName.CLAUDE_3_5_LATEST)
    ```

3. Run the program

    ```bash
    python -m app.main
    ```
