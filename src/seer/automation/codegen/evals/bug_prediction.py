from seer.automation.codegen.bug_prediction_step import BugPredictionStep, BugPredictionStepRequest
from seer.automation.codegen.models import CodegenRelevantWarningsRequest
from seer.automation.codegen.tasks import create_initial_bug_prediction_run
from seer.automation.models import RepoDefinition
from seer.automation.state import DbStateRunTypes


class BugPredictionEvaluationComponent:
    def invoke(self):
        request = CodegenRelevantWarningsRequest(
            repo=RepoDefinition(
                provider="integrations:github",
                owner="getsentry",
                name="seer",
                external_id="651733415",
            ),
            pr_id=1870,
            callback_url="http://localhost:9091",
            organization_id=1,
            warnings=[],
            commit_sha="4aca350c2b928f8610801ccd0c93ac1911b4f43e",
        )
        state = create_initial_bug_prediction_run(request)
        run_id = state.get().run_id

        data = BugPredictionStepRequest(
            run_id=run_id,
            repo=RepoDefinition(
                provider="integrations:github",
                owner="getsentry",
                name="seer",
                external_id="651733415",
            ),
            pr_id=1870,
            callback_url="http://localhost:9091",
            organization_id=1,
            warnings=[],
            commit_sha="4aca350c2b928f8610801ccd0c93ac1911b4f43e",
        )

        bug_prediction_request = data.to_bug_prediction_request()
        bug_prediction_request.run_id = run_id

        BugPredictionStep(
            request=bug_prediction_request, type=DbStateRunTypes.BUG_PREDICTION
        ).invoke()


class BugPredictionScorerComponent:
    def invoke(self):
        pass
