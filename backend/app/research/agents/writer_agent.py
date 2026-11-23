from __future__ import annotations

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from agents import Agent

from app.research.agents.models import ReportData

WRITER_AGENT_PROMPT = """\
You are a Principal Research Writer Agent tasked with synthesizing a comprehensive,
publication-grade Markdown report from an enhanced research brief and web search findings.

GUIDELINES:
1. Start the `markdown_report` with a clear `# Title` heading.
2. Include:
   - **Executive Summary**: concise synthesis of the bottom-line answer.
   - **Detailed Sections**: well-organized thematic sections with subsections, concrete
     examples, data points, and comparison tables where helpful.
   - **Key Takeaways & Recommendations**: actionable conclusions and tradeoffs.
   - **Sources**: clean Markdown list of cited primary URLs from the findings.
3. Respect the requested research depth (`concise`, `standard`, or `deep`).
4. Never wrap `markdown_report` in outer JSON or markdown code fences.
5. Also return:
   - `short_summary`: a 2-3 sentence overview of the findings.
   - `follow_up_questions`: 3 concise, compelling next research questions (each under
     15 words) the user might want to explore next.
"""


def new_writer_agent(model: str) -> Agent:
    return Agent(
        name="WriterAgent",
        model=model,
        instructions=WRITER_AGENT_PROMPT,
        output_type=ReportData,
    )
