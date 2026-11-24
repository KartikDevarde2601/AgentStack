from __future__ import annotations

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from agents import Agent

from app.research.agents.models import WebSearchPlan

PLANNER_AGENT_PROMPT = """\
You are a senior Research Planner Agent.
Given an enhanced research brief, design a set of 4 to 6 distinct, targeted web searches
that together cover all key angles, primary evidence, benchmarks, and tradeoffs needed to
answer the brief thoroughly.

GUIDELINES:
- Each search query should be concrete, search-engine friendly, and non-overlapping.
- Include a brief `reason` explaining what evidence each search will contribute.
- Keep `topic` concise and standalone.
"""


def new_planner_agent(model: str) -> Agent:
    return Agent(
        name="PlannerAgent",
        model=model,
        instructions=PLANNER_AGENT_PROMPT,
        output_type=WebSearchPlan,
    )
