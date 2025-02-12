# Web Explorer Agent

This is a web explorer agent that can be used to explore the web and get information about a given topic.

## Usage

1. Installation: Setup your environment (you can use venv)

    ```shell
    python -m venv .venv
    ```

    Activate the virtual environment

    ```shell
    source .venv/bin/activate
    ```

    Install the dependencies

    ```shell
    pip install -r requirements.txt
    ```

2. Configure your `prompt`, `title` & `instruction` in the `app/main.py` file

3. Optional: Configure your `name` in the `app/agent/main.py` line 45 file. You can swap to any model of your choice here e.g AnthropicModelName | OpenAIModelName

    By default, the model is set to `AnthropicModelName.CLAUDE_3_5_LATEST`

    ```python
    self.llm_model = LLMModel(name=AnthropicModelName.CLAUDE_3_5_LATEST)
    ```

4. Run the program

    ```bash
    python -m app.main
    ```

## LangSmith

```bash
langgraph dev
```
