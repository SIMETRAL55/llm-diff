# LLM Diff

> Prompt regression testing for RAG pipelines. Like git diff, but for AI.

[![PyPI version](https://img.shields.io/pypi/v/llmdiff)](https://pypi.org/project/llmdiff)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Powered by Groq](https://img.shields.io/badge/Powered%20by-Groq-orange)](https://console.groq.com)

<!-- screenshot placeholder: dark dashboard -->

## Why LLM Diff

- **You changed a prompt. Did it get better?** Find out in 2 minutes.
- **Works with any LLM** — Groq (free), OpenAI, Anthropic, Ollama.
- **Local-first.** No accounts, no cloud, no data leaves your machine.
- **One env var.** Set `GROQ_API_KEY` and you're done.

## Quickstart

```bash
# 1. Install
pip install llmdiff

# 2. Get a free Groq API key at console.groq.com (takes 90 seconds)

# 3. Set the key
export GROQ_API_KEY=your_key_here

# 4. Write a test YAML (or copy from examples/)
cp examples/rag_pipeline.yaml my_tests.yaml

# 5. Compare your prompts
llmdiff compare my_tests.yaml
```

## I have a LangChain RAG app — how do I use this?

If your app looks like:

```python
result = chain.invoke({"question": q, "context": c})
```

Translate it into a YAML test case like this:

```yaml
model: groq/llama3-70b-8192
judge_model: groq/llama3-70b-8192
test_cases:
  - id: my_test
    input: "What is the default chunk size?"
    context: "LangChain's default chunk_size is 1000 characters..."
    criteria:
      - "Answer is factually correct"
      - "Response is concise (under 50 words)"
    prompt_v1: |
      You are a helpful assistant. Context: {context}
      Question: {input}
    prompt_v2: |
      Answer only from context. Be concise.
      Context: {context}
      Question: {input}
```

Then run: `llmdiff compare my_tests.yaml`

## Switching providers

Change 1–2 lines in your YAML — no code changes:

| Provider   | Model string                          | Env var            |
|------------|---------------------------------------|--------------------|
| Groq       | `groq/llama3-70b-8192`                | `GROQ_API_KEY`     |
| Groq fast  | `groq/llama-3.1-8b-instant`           | `GROQ_API_KEY`     |
| OpenAI     | `openai/gpt-4o-mini`                  | `OPENAI_API_KEY`   |
| Anthropic  | `anthropic/claude-3-haiku-20240307`   | `ANTHROPIC_API_KEY`|
| Ollama     | `ollama/llama3`                       | (none)             |

## YAML reference

```yaml
model: groq/llama3-70b-8192        # model for generating outputs
judge_model: groq/llama3-70b-8192  # model for judging (can differ)

test_cases:
  - id: tc_001                      # unique identifier
    input: "user question here"
    context: "RAG context passage"  # optional
    criteria:
      - "Answer is factually correct"
      - "Response is under 100 words"
    prompt_v1: |
      Your baseline prompt. Use {input} and {context} placeholders.
    prompt_v2: |
      Your new prompt. Same placeholders.
```

## CLI reference

| Command | Description | Example |
|---------|-------------|---------|
| `llmdiff compare <yaml>` | Run + print colored diff | `llmdiff compare tests.yaml` |
| `llmdiff run <yaml>` | Run + store (no terminal output) | `llmdiff run tests.yaml` |
| `llmdiff history` | List past runs | `llmdiff history` |
| `llmdiff serve` | Start web UI at localhost:7331 | `llmdiff serve` |
| `llmdiff demo` | Try it without an API key | `llmdiff demo` |

## Docker

```bash
docker-compose up
# Dashboard at http://localhost:7331
# Mount your YAML files into /workspace/tests
```

## Contributing

1. Fork the repo
2. Create a branch: `git checkout -b feat/my-feature`
3. Run tests: `pytest tests/ -v`
4. Open a PR
