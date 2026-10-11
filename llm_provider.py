
import json
import os

from groq import Groq


MODEL_NAME = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b",
)


def _get_api_key():
    """Read the Groq API key from environment variables or Streamlit Secrets."""
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
        "GROQ_API_KEY is missing. Add it to your local environment "
        "or configure it in Streamlit Community Cloud Secrets."
    )


def _get_client():
    """Create a Groq API client."""
    return Groq(api_key=_get_api_key())


def chat(
    model=None,
    messages=None,
    format=None,
    options=None,
    **kwargs,
):
    """
    Compatibility wrapper for existing ollama.chat() calls.

    Returns:
        {"message": {"content": "..."}}
    """
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

        schema_instruction = (
            "\n\nReturn ONLY a valid JSON object matching this JSON "
            "schema. Do not include Markdown fences or commentary.\n"
            f"JSON schema:\n{schema_json}"
        )

        # Add the schema instructions to the last user message.
        user_indices = [
            i
            for i, message in enumerate(groq_messages)
            if message["role"] == "user"
        ]

        if user_indices:
            last_user_index = user_indices[-1]
            groq_messages[last_user_index]["content"] += (
                schema_instruction
            )
        else:
            groq_messages.append({
                "role": "user",
                "content": schema_instruction,
            })

        # JSON mode requests valid JSON syntax. The application must
        # still validate fields and citations after receiving the result.
        response_format = {"type": "json_object"}

    options = options or {}

    request = {
        "model": model or MODEL_NAME,
        "messages": groq_messages,
        "temperature": options.get("temperature", 0.2),
    }

    if response_format:
        request["response_format"] = response_format

    # Forward supported optional parameters when explicitly supplied.
    for parameter in ("max_tokens", "max_completion_tokens", "top_p", "seed"):
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
