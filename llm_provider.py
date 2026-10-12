
import json
import os

from groq import Groq


MODEL_NAME = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b",
)


def _get_api_key():
    """Read the API key from environment variables or Streamlit Secrets."""

    key = os.getenv("GROQ_API_KEY", "").strip()

    if key:
        return key

    try:
        import streamlit as st

        key = str(st.secrets.get("GROQ_API_KEY", "")).strip()

        if key:
            return key

    except Exception:
        pass

    raise RuntimeError(
        "GROQ_API_KEY is missing. Configure it in "
        "Streamlit Community Cloud Secrets or your local environment."
    )


def _get_client():
    """Create the Groq API client."""
    return Groq(api_key=_get_api_key())


def chat(
    model=None,
    messages=None,
    format=None,
    options=None,
    **kwargs,
):
    """Provide an Ollama-compatible response using Groq."""

    if not messages:
        raise ValueError("At least one chat message is required.")

    groq_messages = []

    for message in messages:
        role = message.get("role", "user")

        if role not in {"system", "user", "assistant"}:
            role = "user"

        groq_messages.append({
            "role": role,
            "content": str(message.get("content", "")),
        })

    response_format = None

    if format:
        schema_json = json.dumps(format, ensure_ascii=False)

        instruction = (
            "\n\nReturn only a valid JSON object matching the "
            "requested schema. Do not use Markdown fences.\n"
            f"JSON schema:\n{schema_json}"
        )

        user_indices = [
            i
            for i, message in enumerate(groq_messages)
            if message["role"] == "user"
        ]

        if user_indices:
            groq_messages[user_indices[-1]]["content"] += instruction
        else:
            groq_messages.append({
                "role": "user",
                "content": instruction,
            })

        response_format = {"type": "json_object"}

    options = options or {}

    request = {
        "model": model or MODEL_NAME,
        "messages": groq_messages,
        "temperature": options.get("temperature", 0.2),
    }

    if response_format:
        request["response_format"] = response_format

    for parameter in (
        "max_tokens",
        "max_completion_tokens",
        "top_p",
        "seed",
    ):
        value = kwargs.get(parameter)

        if value is not None:
            request[parameter] = value

    try:
        response = _get_client().chat.completions.create(**request)
        content = response.choices[0].message.content or ""

        return {
            "message": {
                "content": content.strip(),
            }
        }

    except Exception as error:
        raise RuntimeError(
            f"Groq API request failed: {error}"
        ) from error
