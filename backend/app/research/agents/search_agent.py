from __future__ import annotations

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from agents import Agent, WebSearchTool
    from agents.model_settings import ModelSettings

from app.research.agents.models import SearchSummary

SEARCH_AGENT_PROMPT = """\
You are a Web Search Research Agent.
Given a search query, its research reason, and the desired depth, you MUST search the live
web using the web search tool and distill the results into a concise, evidence-backed summary.

GUIDELINES:
- Always invoke the web search tool first.
- Capture concrete facts, numbers, dates, and technical details; cut fluff.
- Populate `summary` with a tight 1-2 paragraph synthesis (under 250 words).
- Populate `key_points` with 3-5 concrete bullet takeaways (adjusting for concise/standard/deep).
- Populate `sources` with the real URLs or primary source titles returned by the web search.
- Do not fabricate URLs; only include sources actually grounded in the search results.
"""


def new_search_agent(model: str) -> Agent:
    return Agent(
        name="SearchAgent",
        model=model,
        instructions=SEARCH_AGENT_PROMPT,
        tools=[WebSearchTool()],
        model_settings=ModelSettings(tool_choice="required"),
        output_type=SearchSummary,
    )
