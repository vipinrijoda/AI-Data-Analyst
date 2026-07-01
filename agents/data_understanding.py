"""Agent: Data Understanding
Reads the dataset profile and produces a plain-English description of what
the dataset represents (business domain, what it tracks), instead of just
reporting column counts.
"""
from utils.llm import call_llm
from prompts.prompts import data_understanding_prompt


def understand_dataset(profile: dict) -> str:
    prompt = data_understanding_prompt(profile)
    try:
        return call_llm(prompt, system="You are a senior data analyst giving a first impression of a new dataset.",
                         max_tokens=400)
    except Exception as e:
        return f"(AI understanding unavailable: {e})"
