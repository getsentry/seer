import textwrap
import xml.etree.ElementTree as ET

from langsmith import traceable

from seer.automation.agent.agent import GptAgent
from seer.automation.agent.models import Message
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.components.planner.models import (
    PlanningOutput,
    PlanningRequest,
    PlanStep,
)
from seer.automation.autofix.components.planner.prompts import PlanningPrompts
from seer.automation.autofix.tools import BaseTools
from seer.automation.autofix.utils import autofix_logger, escape_multi_xml
from seer.automation.component import BaseComponent


class PlanningComponent(BaseComponent):
    context: AutofixContext

    def __init__(self, context: AutofixContext):
        super().__init__(context)

    def _parse(self, response: str) -> PlanningOutput | None:
        parsed_output = ET.fromstring(
            f'<response>{escape_multi_xml(response, ["title", "description", "step"])}</response>'
        )

        try:
            title_element = parsed_output.find(".//*title")
            description_element = parsed_output.find(".//*description")
            steps_element = parsed_output.find(".//*steps")

            if steps_element is None:
                autofix_logger.warning(
                    f"Planning response does not contain steps element: {response}"
                )
                return None

            step_elements = steps_element.findall("step")

            if len(step_elements) == 0:
                autofix_logger.warning(f"Planning response does not contain any steps: {response}")
                return None

            steps = [
                PlanStep(
                    id=i,
                    title=step_element.attrib["title"],
                    text=textwrap.dedent(step_element.text or ""),
                )
                for i, step_element in enumerate(step_elements)
            ]

            title = (
                textwrap.dedent(title_element.text or "").strip()
                if title_element is not None
                else None
            )
            description = (
                textwrap.dedent(description_element.text or "").strip()
                if description_element is not None
                else None
            )

            if title is None or description is None:
                autofix_logger.warning(
                    f"Planning response does not contain a title, description, or plan: {response}"
                )
                return None

            return PlanningOutput(title=title, description=description, steps=steps)
        except AttributeError as e:
            autofix_logger.warning(
                f"Planning response does not contain a title, description, or plan: {e}"
            )
            return None

    @traceable(name="Planning", run_type="llm", tags=["planning:v1.1"])
    def invoke(self, request: PlanningRequest) -> PlanningOutput | None:
        with self.context.state.update() as cur:
            planning_agent_tools = BaseTools(self.context)

            planning_agent = GptAgent(
                name="planner",
                tools=planning_agent_tools.get_tools(),
                memory=[
                    Message(
                        role="system",
                        content=PlanningPrompts.format_system_msg(),
                    )
                ],
            )

            message = PlanningPrompts.format_default_msg(
                err_msg=request.sentry_event.title,
                exceptions=request.sentry_event.exceptions,
                problem=request.problem,
                additional_context=request.additional_context or "",
            )

            planning_response = planning_agent.run(message)

            cur.usage += planning_agent.usage

            if planning_response is None:
                autofix_logger.warning(f"Planning agent did not return a valid response")
                return None

            return self._parse(planning_response)
