from typing import Any

import sentry_sdk

from seer.automation.codegen.codegen_context import CodegenContext
from seer.automation.codegen.models import CodegenStatus
from seer.automation.pipeline import PipelineContext, PipelineStep, PipelineStepTaskRequest
from seer.automation.state import DbStateRunTypes
from seer.automation.utils import make_done_signal


class CodegenStep(PipelineStep):
    context: CodegenContext

    @staticmethod
    def _instantiate_context(
        request: PipelineStepTaskRequest, type: DbStateRunTypes | None = None
    ) -> PipelineContext:
        if type is None:
            type = DbStateRunTypes.UNIT_TEST
        return CodegenContext.from_run_id(request.run_id, type=type)

    def _invoke(self, **kwargs: Any) -> Any:
        sentry_sdk.set_tag("run_id", self.context.run_id)
        super()._invoke(**kwargs)

    def _pre_invoke(self) -> bool:
        done_signal = make_done_signal(self.request.step_id)
        return done_signal not in self.context.signals

    def _get_extra_invoke_kwargs(self) -> dict[str, Any]:
        try:
            current_state = self.context.state.get()
            repo = self.context.repos[0]

            tags = {
                "run_id": current_state.run_id,
                "repo": repo.full_name,
                "repo_id": repo.external_id,
            }

            metadata = {"run_id": current_state.run_id, "repo": repo}
            langfuse_tags = [f"{key}:{value}" for key, value in tags.items() if value is not None]

            return {
                "langfuse_tags": langfuse_tags,
                "langfuse_metadata": metadata,
                "langfuse_session_id": str(current_state.run_id),
                "sentry_tags": tags,
                "sentry_data": metadata,
            }
        except Exception:
            return {}

    def _post_invoke(self, result: Any):
        with self.context.state.update() as current_state:
            signal = make_done_signal(self.request.step_id)
            current_state.signals.append(signal)

    def _handle_exception(self, exception: Exception):
        self.logger.error(f"Failed to run {self.request.step_id}. Error: {str(exception)}")

        with self.context.state.update() as current_state:
            current_state.status = CodegenStatus.ERRORED
            sentry_sdk.set_context("codegen_state", current_state.dict())
            sentry_sdk.capture_exception(exception)
