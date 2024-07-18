from langfuse.decorators import observe
from sentry_sdk.ai.monitoring import ai_track

from seer.automation.agent.agent import AgentConfig, GptAgent
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.components.planner.models import (
    PlanningOutput,
    PlanningOutputPromptXml,
    PlanningRequest,
    RootCausePlanTaskPromptXml,
)
from seer.automation.autofix.components.planner.prompts import PlanningPrompts
from seer.automation.autofix.components.root_cause.models import RootCauseAnalysisItem
from seer.automation.autofix.tools import BaseTools
from seer.automation.component import BaseComponent
from seer.automation.utils import escape_multi_xml


class PlanningComponent(BaseComponent[PlanningRequest, PlanningOutput]):
    context: AutofixContext

    @observe(name="Planning")
    @ai_track(description="Planning")
    def invoke(self, request: PlanningRequest) -> PlanningOutput | None:
        tools = BaseTools(self.context)

        agent = GptAgent(
            tools=tools.get_tools(),
            config=AgentConfig(system_prompt=PlanningPrompts.format_system_msg()),
        )

        task_str = (
            RootCausePlanTaskPromptXml.from_root_cause(request.root_cause_and_fix).to_prompt_str()
            if isinstance(request.root_cause_and_fix, RootCauseAnalysisItem)
            else request.root_cause_and_fix
        )

        response = agent.run(
            PlanningPrompts.format_default_msg(
                event=request.event_details,
                task_str=task_str,
                instruction=request.instruction,
            )
        )

        with self.context.state.update() as cur:
            cur.usage += agent.usage

        if not response:
            return None

        cleaned_response = escape_multi_xml(
            response, ["thoughts", "snippet", "reference_snippet", "new_snippet"]
        )

        return PlanningOutputPromptXml.from_xml(
            f"<planning_output>{cleaned_response}</planning_output>"
        ).to_model()
