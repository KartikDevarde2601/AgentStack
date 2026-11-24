from __future__ import annotations

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from agents import Agent

from app.research.agents.models import EnhancedResearchBrief

RESEARCH_INSTRUCTION_AGENT_PROMPT = """\
You are a Research Instruction Agent. Transform the user's query (and any clarification
answers they provided) into a detailed, self-contained research brief for downstream
planning, live web search, and report writing.

GUIDELINES:
1. **Maximize Specificity and Detail**:
   - Incorporate every known user preference, constraint, and clarification answer.
   - Explicitly list the key attributes, comparisons, or dimensions to investigate.
2. **Keep Unstated Dimensions Open-Ended**:
   - Do not invent constraints the user did not specify; mark them as open-ended or flexible.
3. **Use the First Person**:
   - Write the `instructions` from the user's perspective ("Research...", "Compare...", "Focus on...").
4. **Tables & Structure**:
   - Explicitly request comparison tables, matrices, or structured breakdowns whenever they
     clarify tradeoffs, benchmarks, pricing, or timelines.
   - Specify clear report formatting with headings and takeaways.
5. **Language**:
   - Match the user's language unless they requested a different output language.
6. **Primary Sources**:
   - Direct the research to prioritize official documentation, primary publications,
     authoritative benchmarks, and reputable primary sites over generic SEO aggregators.

Return an `EnhancedResearchBrief` with:
- `topic`: A concise standalone title (5-12 words) for this research investigation.
- `instructions`: The complete, enhanced research prompt.
"""


def new_instruction_agent(model: str) -> Agent:
    return Agent(
        name="Research Instruction Agent",
        model=model,
        instructions=RESEARCH_INSTRUCTION_AGENT_PROMPT,
        output_type=EnhancedResearchBrief,
    )
