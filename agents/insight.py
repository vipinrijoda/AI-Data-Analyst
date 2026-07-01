"""Agent: Business Insight
Turns raw statistics into narrative, decision-relevant insights and
recommendations, the way a human analyst would present findings.
"""
from utils.llm import call_llm_json
from prompts.prompts import insight_prompt


def generate_insights(profile: dict, eda_results: dict) -> dict:
    prompt = insight_prompt(profile, eda_results)
    try:
        result = call_llm_json(
            prompt,
            system=("You are a senior business analyst. Respond only in JSON with keys "
                     "'insights' and 'recommendations', each a list of short strings."),
        )
        if "error" in result:
            return {"insights": ["AI insight generation returned an unparseable response."],
                     "recommendations": []}
        result.setdefault("insights", [])
        result.setdefault("recommendations", [])
        return result
    except Exception as e:
        return {"insights": [f"Insight generation failed: {e}"], "recommendations": []}
