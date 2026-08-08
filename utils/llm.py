"""
LLM client abstraction.

Every agent talks to the LLM through call_llm() / call_llm_json() so the
rest of the codebase never needs to know whether we're using Anthropic or
OpenAI under the hood. Swap providers with the LLM_PROVIDER env var.
"""
import os
import json
import streamlit as st

# ============================================
# FIX: Read from st.secrets first, then .env
# ============================================

def get_env_or_secret(key: str, default: str = None) -> str:
    """Get environment variable or Streamlit secret"""
    try:
        # For Streamlit Cloud
        return st.secrets.get(key, os.getenv(key, default))
    except (KeyError, AttributeError):
        # For local development
        return os.getenv(key, default)


PROVIDER = get_env_or_secret("LLM_PROVIDER", "anthropic").lower()


def call_llm(prompt: str, system: str = None, max_tokens: int = 1500, json_mode: bool = False) -> str:
    """Send a prompt to the configured LLM provider and return raw text."""
    if PROVIDER == "anthropic":
        return _call_anthropic(prompt, system, max_tokens, json_mode)
    elif PROVIDER == "openai":
        return _call_openai(prompt, system, max_tokens, json_mode)
    elif PROVIDER == "groq":
        return _call_groq(prompt, system, max_tokens, json_mode)
    raise ValueError(f"Unknown LLM_PROVIDER: {PROVIDER!r}. Use 'anthropic', 'openai', or 'groq'.")


def call_llm_json(prompt: str, system: str = None, max_tokens: int = 1500) -> dict:
    """Call the LLM and parse its response as JSON. Never raises on bad JSON -
    returns {"error": ..., "raw": ...} instead, so callers can fall back gracefully."""
    raw = call_llm(prompt, system=system, max_tokens=max_tokens, json_mode=True)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"error": "Failed to parse LLM JSON output", "raw": raw}


def _call_anthropic(prompt, system, max_tokens, json_mode):
    import anthropic

    client = anthropic.Anthropic()
    model = get_env_or_secret("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")

    sys_prompt = system or ""
    if json_mode:
        sys_prompt = (sys_prompt + "\n\nRespond ONLY with valid JSON. "
                      "No markdown code fences, no preamble, no explanation.").strip()

    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=sys_prompt if sys_prompt else "You are a helpful data analysis assistant.",
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in resp.content if block.type == "text")
    return _strip_fences(text) if json_mode else text


def _call_openai(prompt, system, max_tokens, json_mode):
    from openai import OpenAI

    client = OpenAI()
    model = get_env_or_secret("OPENAI_MODEL", "gpt-4o")

    sys_prompt = system or "You are a helpful data analysis assistant."
    if json_mode:
        sys_prompt += ("\n\nRespond ONLY with valid JSON. No markdown code fences, "
                       "no preamble, no explanation.")

    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    text = resp.choices[0].message.content or ""
    return _strip_fences(text) if json_mode else text


def _call_groq(prompt, system, max_tokens, json_mode):
    # Grox exposes an OpenAI-compatible /chat/completions endpoint
    from openai import OpenAI

    api_key = get_env_or_secret("GROQ_API_KEY")
    if not api_key:
        return "Error: GROQ_API_KEY not found. Please set it in .env or Streamlit secrets."

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )
    model = get_env_or_secret("GROQ_MODEL", "llama-3.3-70b-versatile")

    sys_prompt = system or "You are a helpful data analysis assistant."
    if json_mode:
        sys_prompt += ("\n\nRespond ONLY with valid JSON. No markdown code fences, "
                       "no preamble, no explanation.")

    kwargs = dict(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    if json_mode:
        try:
            resp = client.chat.completions.create(response_format={"type": "json_object"}, **kwargs)
        except Exception:
            resp = client.chat.completions.create(**kwargs)
    else:
        resp = client.chat.completions.create(**kwargs)

    text = resp.choices[0].message.content or ""
    return _strip_fences(text) if json_mode else text


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]
    return text.strip()
