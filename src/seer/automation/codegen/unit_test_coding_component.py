import logging

from langfuse.decorators import observe
from sentry_sdk.ai.monitoring import ai_track

from integrations.codecov.codecov_client import CodecovClient
from seer.automation.agent.agent import AgentConfig, LlmAgent, RunConfig
from seer.automation.agent.client import AnthropicProvider, LlmClient
from seer.automation.autofix.components.coding.models import PlanStepsPromptXml
from seer.automation.autofix.components.coding.utils import (
    task_to_file_change,
    task_to_file_create,
    task_to_file_delete,
)
from seer.automation.autofix.tools import BaseTools
from seer.automation.codebase.repo_client import RepoClientType
from seer.automation.codegen.codegen_context import CodegenContext
from seer.automation.codegen.models import CodeUnitTestOutput, CodeUnitTestRequest
from seer.automation.codegen.prompts import CodingUnitTestPrompts
from seer.automation.component import BaseComponent
from seer.automation.models import FileChange
from seer.automation.utils import escape_multi_xml, extract_text_inside_tags
from seer.dependency_injection import inject, injected

logger = logging.getLogger(__name__)


class UnitTestCodingComponent(BaseComponent[CodeUnitTestRequest, CodeUnitTestOutput]):
    context: CodegenContext

    @observe(name="Generate unit tests")
    @ai_track(description="Generate unit tests")
    @inject
    def invoke(
        self, request: CodeUnitTestRequest, llm_client: LlmClient = injected
    ) -> CodeUnitTestOutput | None:
        pass
