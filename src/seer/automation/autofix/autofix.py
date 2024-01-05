import logging
import os
import xml.etree.ElementTree as ET

import torch
from github import Auth

from ..agent.agent import GptAgent, Message, Usage
from .agent_context import AgentContext
from .prompts import action_prompt, planning_prompt
from .tools import BaseTools, CodeActionTools
from .types import (
    AutofixAgentsOutput,
    AutofixInput,
    AutofixOutput,
    IssueDetails,
    PlanningOutput,
    SentryEvent,
)

# TODO: Remove this when we stop locking it to a repo
REPO_OWNER = "getsentry"
REPO_NAME = "sentry-mirror-suggested-fix"
REPO_REF = "heads/master"


class Autofix:
    def __init__(self, issue_details: IssueDetails, autofix_input: AutofixInput):
        self.issue_details = issue_details
        self.autofix_input = autofix_input
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        app_id = os.environ.get("GITHUB_APP_ID")
        private_key = os.environ.get("GITHUB_PRIVATE_KEY")

        if app_id is None or private_key is None:
            raise ValueError("GITHUB_APP_ID or GITHUB_PRIVATE_KEY is not set")

        github_auth = Auth.AppAuth(app_id, private_key=private_key)

        self.agent_context = AgentContext(
            repo_owner=REPO_OWNER,
            repo_name=REPO_NAME,
            ref=REPO_REF,
            model="gpt-4-1106-preview",
            github_auth=github_auth,
            gpu_device=self.device,
        )

    def run(self):
        index, documents, nodes = self.agent_context.get_data()

        planning_output = self._run_planning_agent(index, documents, nodes)
        if planning_output is None:
            logger.warning(f"Planning agent did not return a valid output")
            return None

        coding_output, coding_usage = self._run_coding_agent(
            self.agent_context.base_sha,
            planning_output,
            index,
            documents,
            nodes,
        )

        model_dump = planning_output.model_dump()
        model_dump.pop("usage", None)  # Remove the existing usage key if present
        combined_output = AutofixAgentsOutput(
            **model_dump,
            changes=list(coding_output.values()),
            usage=Usage(
                prompt_tokens=planning_output.usage.prompt_tokens + coding_usage.prompt_tokens,
                completion_tokens=planning_output.usage.completion_tokens
                + coding_usage.completion_tokens,
                total_tokens=planning_output.usage.total_tokens + coding_usage.total_tokens,
            ),
        )

        pr = self._create_pr(combined_output)

        self.agent_context.cleanup()

        autofix_output = AutofixOutput(
            title=planning_output.title,
            description=planning_output.description,
            plan=planning_output.plan,
            usage=planning_output.usage,
            pr_url=pr.html_url,
        )

        return autofix_output

    def _create_pr(self, combined_output: AutofixAgentsOutput):
        branch_ref = self.agent_context.create_branch_from_changes(
            file_changes=combined_output.changes, base_commit_sha=self.agent_context.base_sha
        )

        return self.agent_context.create_pr_from_branch(
            branch_ref, combined_output, self.issue_details
        )

    def _run_planning_agent(self, index, documents, nodes) -> PlanningOutput | None:
        planning_agent_tools = BaseTools(index, documents)

        planning_agent = GptAgent(
            tools=planning_agent_tools.get_tools(),
            memory=[
                Message(
                    role="system",
                    content=planning_prompt.format(
                        err_msg=self.issue_details.title,
                        stack_str=self.issue_details.events[-1].build_stacktrace(),
                    ),
                )
            ],
        )

        additional_context_str = (
            (self.autofix_input.additional_context + "\n\n")
            if self.autofix_input.additional_context
            else ""
        )
        planning_response = planning_agent.run(
            f"{additional_context_str}Note: instead of ./app, the correct directory is static/app/..."
        )

        parsed_output = ET.fromstring(f"<response>{planning_response}</response>")

        try:
            title_element = parsed_output.find("title")
            description_element = parsed_output.find("description")
            plan_element = parsed_output.find("plan")

            title = title_element.text if title_element is not None else None
            description = description_element.text if description_element is not None else None
            plan = plan_element.text if plan_element is not None else None

            if title is None or description is None or plan is None:
                logger.warning(
                    f"Planning response does not contain a title, description, or plan: {planning_response}"
                )
                return None

            return PlanningOutput(
                title=title, description=description, plan=plan, usage=planning_agent.usage
            )
        except AttributeError as e:
            logger.warning(f"Planning response does not contain a title, description, or plan: {e}")
            return None

    def _run_coding_agent(
        self,
        base_sha: str,
        planning_output: PlanningOutput,
        index,
        documents,
        nodes,
    ):
        code_action_tools = CodeActionTools(
            self.agent_context, index, documents, base_sha=base_sha, verbose=True
        )
        action_agent = GptAgent(
            tools=code_action_tools.get_tools(),
            memory=[
                Message(
                    role="system",
                    content=action_prompt.format(
                        err_msg=self.issue_details.title,
                        stack_str=self.issue_details.events[-1].build_stacktrace(),
                    ),
                )
            ],
        )

        action_agent.run(planning_output.plan)

        return code_action_tools.file_changes, action_agent.usage


if __name__ == "__main__":
    import json

    logger = logging.getLogger("autofix")
    logger.addHandler(logging.FileHandler("./autofix.log"))
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)

    issue_details = IssueDetails(
        id="4726606035",
        title="TypeError: int() argument must be a string, a bytes-like object or a real number, not 'list'",
        events=[
            SentryEvent(
                entries=json.loads(
                    """[{"data": {"values": [{"type": "TypeError", "value": "int() argument must be a string, a bytes-like object or a real number, not \'list\'", "mechanism": {"type": "celery", "handled": false}, "threadId": null, "module": null, "stacktrace": {"frames": [{"filename": "celery/app/trace.py", "absPath": "/usr/local/lib/python3.10/site-packages/celery/app/trace.py", "module": "celery.app.trace", "package": null, "platform": null, "instructionAddr": null, "symbolAddr": null, "function": "__protected_call__", "rawFunction": null, "symbol": null, "context": [[755, "            stack = self.request_stack"], [756, "            req = stack.top"], [757, "            if req and not req._protected and \\\\"], [758, "                    len(stack) == 1 and not req.called_directly:"], [759, "                req._protected = 1"], [760, "                return self.run(*args, **kwargs)"], [761, "            return orig(self, *args, **kwargs)"], [762, "        BaseTask.__call__ = __protected_call__"], [763, "        BaseTask._stackprotected = True"]], "lineNo": 760, "colNo": null, "inApp": false, "trust": null, "errors": null, "lock": null, "sourceLink": null, "vars": {"args": [], "kwargs": {"__start_time": "1704385240.473305", "public_key": "\'8096fec1d7cadc3a6830e7a2361fc1aa\'", "tmp_scheduled": "1704385240.4726431"}, "orig": "<function Task.__call__ at 0x7f00746a3400>", "req": "<Context: {\'lang\': None, \'task\': \'sentry.tasks.relay.build_project_config\', \'id\': \'fcb3d019-7d83-4b5d-b787-ebc1ed441e4a\', \'root_id\': None, \'parent_id\': None, \'group\': None, \'meth\': None, \'shadow\': None, \'eta\': None, \'expires\': \'2024-01-04T16:21:10.474045+00:00\', \'retries\': 0, \'timelimit\': (10, 5), \'argsrepr\': None, \'kwargsrepr\': None, \'origin\': None, \'sentry-trace\': \'dcdfa9e669854686b49d4aef587efdc6-9c3570724fb9de8d-0\', \'baggage\': \'sentry-trace_id=dcdfa9e669854686b49d4aef587efdc6,sentry-environment=prod,sentry-release=backend%40dce36d6c53daf98f783e6066ced8869602d47851,sentry-public_key=8df96f046b994933aafa8dcfa3345419,sentry-transaction=RelayProjectConfigsEndpoint,sentry-sample_rate=0.0,sentry-sampled=false\', \'sentry-monitor-start-timestamp-s\': \'1704385240.473489523\', \'headers\': {\'sentry-trace\': \'dcdfa9e669854686b49d4aef587efdc6-9c3570724fb9de8d-0\', \'baggage\': \'sentry-trace_id=dcdfa9e669854686b49d4aef587efdc6,sentry-environment=prod,sentry-release=backend%40dce36d6c53daf98f783e6066ced8869602d47851,sentry-p...", "self": "<@task: sentry.tasks.relay.build_project_config of sentry at 0x7f00747f2950>", "stack": "<celery.utils.threads._LocalStack object at 0x7f001a866fb0>"}}, {"filename": "sentry/tasks/base.py", "absPath": "/usr/src/sentry/src/sentry/tasks/base.py", "module": "sentry.tasks.base", "package": null, "platform": null, "instructionAddr": null, "symbolAddr": null, "function": "_wrapped", "rawFunction": null, "symbol": null, "context": [[112, "                scope.set_tag(\\"transaction_id\\", transaction_id)"], [113, ""], [114, "            with metrics.timer(key, instance=instance), track_memory_usage("], [115, "                \\"jobs.memory_change\\", instance=instance"], [116, "            ):"], [117, "                result = func(*args, **kwargs)"], [118, ""], [119, "            return result"], [120, ""], [121, "        # We never use result backends in Celery. Leaving `trail=True` means that if we schedule"], [122, "        # many tasks from a parent task, each task leaks memory. This can lead to the scheduler"]], "lineNo": 117, "colNo": null, "inApp": true, "trust": null, "errors": null, "lock": null, "sourceLink": null, "vars": {"args": [], "func": "<function build_project_config at 0x7f003757f490>", "instance": "\'sentry.tasks.relay.build_project_config\'", "key": "\'jobs.duration\'", "kwargs": {"public_key": "\'8096fec1d7cadc3a6830e7a2361fc1aa\'", "tmp_scheduled": "1704385240.4726431"}, "name": "\'sentry.tasks.relay.build_project_config\'", "record_timing": "False", "scope": "<Scope id=0x7f000b7cb280 name=celery>", "start_time": "1704385240.473305", "transaction_id": "None"}}, {"filename": "sentry/tasks/relay.py", "absPath": "/usr/src/sentry/src/sentry/tasks/relay.py", "module": "sentry.tasks.relay", "package": null, "platform": null, "instructionAddr": null, "symbolAddr": null, "function": "build_project_config", "rawFunction": null, "symbol": null, "context": [[46, "            # In this particular case, where a project key got deleted and"], [47, "            # triggered an update, we know that key doesn\'t exist and we want to"], [48, "            # avoid creating more tasks for it."], [49, "            projectconfig_cache.backend.set_many({public_key: {\\"disabled\\": True}})"], [50, "        else:"], [51, "            config = compute_projectkey_config(key)"], [52, "            projectconfig_cache.backend.set_many({public_key: config})"], [53, ""], [54, "    finally:"], [55, "        # Delete the key in this `finally` block to make sure the debouncing key"], [56, "        # is always deleted. Deleting the key at the end of the task also makes"]], "lineNo": 51, "colNo": null, "inApp": true, "trust": null, "errors": null, "lock": null, "sourceLink": null, "vars": {"ProjectKey": "<class \'sentry.models.projectkey.ProjectKey\'>", "key": "<ProjectKey at 0x7f0012d31ae0: id=3361162, project_id=****************, public_key=\'8096fec1d7cadc3a6830e7a2361fc1aa\'>", "kwargs": {"tmp_scheduled": "1704385240.4726431"}, "public_key": "\'8096fec1d7cadc3a6830e7a2361fc1aa\'"}}, {"filename": "sentry/tasks/relay.py", "absPath": "/usr/src/sentry/src/sentry/tasks/relay.py", "module": "sentry.tasks.relay", "package": null, "platform": null, "instructionAddr": null, "symbolAddr": null, "function": "compute_projectkey_config", "rawFunction": null, "symbol": null, "context": [[188, "    from sentry.relay.config import get_project_config"], [189, ""], [190, "    if key.status != ProjectKeyStatus.ACTIVE:"], [191, "        return {\\"disabled\\": True}"], [192, "    else:"], [193, "        return get_project_config(key.project, project_keys=[key], full_config=True).to_dict()"], [194, ""], [195, ""], [196, "@instrumented_task("], [197, "    name=\\"sentry.tasks.relay.invalidate_project_config\\","], [198, "    queue=\\"relay_config_bulk\\","]], "lineNo": 193, "colNo": null, "inApp": true, "trust": null, "errors": null, "lock": null, "sourceLink": null, "vars": {"ProjectKeyStatus": "<class \'sentry.models.projectkey.ProjectKeyStatus\'>", "get_project_config": "<function get_project_config at 0x7f002f4955a0>", "key": "<ProjectKey at 0x7f0012d31ae0: id=3361162, project_id=****************, public_key=\'8096fec1d7cadc3a6830e7a2361fc1aa\'>"}}, {"filename": "sentry/relay/config/__init__.py", "absPath": "/usr/src/sentry/src/sentry/relay/config/__init__.py", "module": "sentry.relay.config", "package": null, "platform": null, "instructionAddr": null, "symbolAddr": null, "function": "get_project_config", "rawFunction": null, "symbol": null, "context": [[210, "    :return: a ProjectConfig object for the given project"], [211, "    \\"\\"\\""], [212, "    with sentry_sdk.push_scope() as scope:"], [213, "        scope.set_tag(\\"project\\", project.id)"], [214, "        with metrics.timer(\\"relay.config.get_project_config.duration\\"):"], [215, "            return _get_project_config(project, full_config=full_config, project_keys=project_keys)"], [216, ""], [217, ""], [218, "def get_dynamic_sampling_config(project: Project) -> Optional[Mapping[str, Any]]:"], [219, "    if features.has(\\"organizations:dynamic-sampling\\", project.organization):"], [220, "        # For compatibility reasons we want to return an empty list of old rules. This has been done in order to make"]], "lineNo": 215, "colNo": null, "inApp": true, "trust": null, "errors": null, "lock": null, "sourceLink": null, "vars": {"full_config": "True", "project": "<Project at 0x7f0012d31e70: id=****************, team_id=None, name=\'leebet-client\', slug=\'leebet-client\'>", "project_keys": ["<ProjectKey at 0x7f0012d31ae0: id=3361162, project_id=****************, public_key=\'8096fec1d7cadc3a6830e7a2361fc1aa\'>"], "scope": "<Scope id=0x7f000b7caec0 name=celery>"}}, {"filename": "sentry/relay/config/__init__.py", "absPath": "/usr/src/sentry/src/sentry/relay/config/__init__.py", "module": "sentry.relay.config", "package": null, "platform": null, "instructionAddr": null, "symbolAddr": null, "function": "_get_project_config", "rawFunction": null, "symbol": null, "context": [[633, "    with Hub.current.start_span(op=\\"get_event_retention\\"):"], [634, "        event_retention = quotas.backend.get_event_retention(project.organization)"], [635, "        if event_retention is not None:"], [636, "            config[\\"eventRetention\\"] = event_retention"], [637, "    with Hub.current.start_span(op=\\"get_all_quotas\\"):"], [638, "        if quotas_config := get_quotas(project, keys=project_keys):"], [639, "            config[\\"quotas\\"] = quotas_config"], [640, ""], [641, "    return ProjectConfig(project, **cfg)"], [642, ""], [643, ""]], "lineNo": 638, "colNo": null, "inApp": true, "trust": null, "errors": null, "lock": null, "sourceLink": null, "vars": {"cfg": {"config": {"allowedDomains": ["\'*\'"], "breakdownsV2": {"span_ops": "{\\"matches\\":\\"[\'http\', \'db\', \'browser\', \'resource\', \'ui\']\\",\\"type\\":\\"\'spanOperations\'\\"}"}, "datascrubbingSettings": {"excludeFields": "[]", "scrubData": "True", "scrubDefaults": "True", "sensitiveFields": "[]"}, "features": ["\'projects:span-metrics-extraction\'", "\'projects:span-metrics-extraction-resource\'", "\'organizations:transaction-name-mark-scrubbed-as-sanitized\'", "\'organizations:transaction-name-normalize\'", "\'organizations:profiling\'", "\'organizations:session-replay\'", "\'organizations:user-feedback-ingest\'", "\'organizations:session-replay-recording-scrubbing\'", "\'organizations:device-class-synthesis\'"], "metricConditionalTagging": ["{\\"condition\\":\\"{\'op\': \'and\', \'inner\': [{\'op\': \'gt\', \'name\': \'event.duration\', \'value\': 1200}]}\\",\\"tagValue\\":\\"\'frustrated\'\\",\\"targetMetrics\\":\\"(\'s:transactions/user@none\', \'d:transactions/duration@millisecond\', \'d:transactions/measurements.lcp@millisecond\')\\",\\"targetTag\\":\\"\'satisfaction\'\\"}", "{\\"condition\\":\\"{\'op\': \'and\', \'inner\': [{\'op\': \'gt\', \'name\': \'event.duration\', \'value\': 300}]}\\",\\"tagValue\\":\\"\'tolerated\'\\",\\"targetMetrics\\":\\"(\'s:transactions/user@none\', \'d:transactions/duration@millisecond\', \'d:transactions/measurements.lcp@millisecond\')\\",\\"targetTag\\":\\"\'satisfaction\'\\"}", "{\\"condition\\":\\"{\'op\': \'and\', \'inner\': []}\\",\\"tagValue\\":\\"\'satisfied\'\\",\\"targetMetrics\\":\\"(\'s:transactions/user@none\', \'d:transactions/duration@millisecond\', \'d:transactions/measurements.lcp@millisecond\')\\",\\"targetTag\\":\\"\'satisfaction\'\\"}", "{\\"condition\\":\\"{\'op\': \'and\', \'inner\': [{\'op\': \'eq\', \'name\': \'event.contexts.trace.op\', \'value\': \'pageload\'}, {\'op\': \'eq\', \'name\': \'event.platform\', \'value\': \'javascript\'}, {\'op\': \'gte\', \'name\': \'event.duration\', \'value\': 16123.0}]}\\",\\"tagValue\\":\\"\'outlier\'\\",\\"targetMetrics\\":\\"[\'d:transactions/duration@millisecond\']\\",\\"targetTag\\":\\"\'histogram_outlier\'\\"}", "{\\"condition\\":\\"{\'op\': \'and\', \'inner\': [{\'op\': \'eq\', \'name\': \'event.contexts.trace.op\', \'value\': \'pageload\'}, {\'op\': \'eq\', \'name\': \'event.platform\', \'value\': \'javascript\'}, {\'op\': \'gte\', \'name\': \'event.duration\', \'value\': 7941.899538040161}]}\\",\\"tagValue\\":\\"\'outlier\'\\",\\"targetMetrics\\":\\"[\'d:transactions/measurements.lcp@millisecond\']\\",\\"targetTag\\":\\"\'histogram_outlier\'\\"}", "{\\"condition\\":..."]}}}}, {"filename": "sentry/relay/config/__init__.py", "absPath": "/usr/src/sentry/src/sentry/relay/config/__init__.py", "module": "sentry.relay.config", "package": null, "platform": null, "instructionAddr": null, "symbolAddr": null, "function": "get_quotas", "rawFunction": null, "symbol": null, "context": [[181, ""], [182, ""], [183, "def get_quotas(project: Project, keys: Optional[Sequence[ProjectKey]] = None) -> List[str]:"], [184, "    try:"], [185, "        computed_quotas = ["], [186, "            quota.to_json() for quota in quotas.backend.get_quotas(project, keys=keys)"], [187, "        ]"], [188, "    except BaseException:"], [189, "        metrics.incr(\\"relay.config.get_quotas\\", tags={\\"success\\": False}, sample_rate=1.0)"], [190, "        raise"], [191, "    else:"]], "lineNo": 186, "colNo": null, "inApp": true, "trust": null, "errors": null, "lock": null, "sourceLink": null, "vars": {"keys": ["<ProjectKey at 0x7f0012d31ae0: id=3361162, project_id=****************, public_key=\'8096fec1d7cadc3a6830e7a2361fc1aa\'>"], "project": "<Project at 0x7f0012d31e70: id=****************, team_id=None, name=\'leebet-client\', slug=\'leebet-client\'>"}}, {"filename": "getsentry/quotas.py", "absPath": "/usr/src/getsentry/getsentry/quotas.py", "module": "getsentry.quotas", "package": null, "platform": null, "instructionAddr": null, "symbolAddr": null, "function": "get_quotas", "rawFunction": null, "symbol": null, "context": [[393, ""], [394, "        metric_histories = self._get_metric_histories(sub)"], [395, ""], [396, "        # This is going to include project abuse quotas ever since"], [397, "        # that part was moved to sentry, for single-tenant."], [398, "        quotas = super().get_quotas(project, *args, **kwargs)"], [399, ""], [400, "        # Billing quotas"], [401, "        for billing_metric, _ in BILLING_METRIC_USAGE_TALLY_TYPE_CHOICES:"], [402, "            data_category = DataCategory(billing_metric)"], [403, "            quotas.extend("]], "lineNo": 398, "colNo": null, "inApp": true, "trust": null, "errors": null, "lock": null, "sourceLink": null, "vars": {"__class__": "<class \'getsentry.quotas.SubscriptionQuota\'>", "args": [], "kwargs": {"keys": ["<ProjectKey at 0x7f0012d31ae0: id=3361162, project_id=****************, public_key=\'8096fec1d7cadc3a6830e7a2361fc1aa\'>"]}, "metric_histories": {"1": "<BillingMetricHistory at 0x7f0012dc4be0: id=142739595, billing_history_id=672723729, billing_metric=1, total=5251>", "2": "<BillingMetricHistory at 0x7f0011b8cc70: id=142739596, billing_history_id=672723729, billing_metric=2, total=11637>", "4": "<BillingMetricHistory at 0x7f0012dc5330: id=142739598, billing_history_id=672723729, billing_metric=4, total=0>", "7": "<BillingMetricHistory at 0x7f0012dc5cc0: id=142739597, billing_history_id=672723729, billing_metric=7, total=53>"}, "organization": "<Organization at 0x7f0012d30850: id=****************, owner_id=None, name=\'Satoshi Tech\', slug=\'satoshi-tech\'>", "project": "<Project at 0x7f0012d31e70: id=****************, team_id=None, name=\'leebet-client\', slug=\'leebet-client\'>", "self": "<getsentry.quotas.SubscriptionQuota object at 0x7f002c1d10f0>", "sub": "<Subscription at 0x7f0012d30970: id=1697878, organization_id=****************, plan=\'am2_f\', status=\'active\'>"}}, {"filename": "sentry/quotas/redis.py", "absPath": "/usr/src/sentry/src/sentry/quotas/redis.py", "module": "sentry.quotas.redis", "package": null, "platform": null, "instructionAddr": null, "symbolAddr": null, "function": "get_quotas", "rawFunction": null, "symbol": null, "context": [[51, ""], [52, "    def get_quotas(self, project, key=None, keys=None):"], [53, "        if key:"], [54, "            key.project = project"], [55, ""], [56, "        results = [*self.get_abuse_quotas(project.organization)]"], [57, ""], [58, "        with sentry_sdk.start_span(op=\\"redis.get_quotas.get_project_quota\\") as span:"], [59, "            span.set_tag(\\"project.id\\", project.id)"], [60, "            pquota = self.get_project_quota(project)"], [61, "            if pquota[0] is not None:"]], "lineNo": 56, "colNo": null, "inApp": true, "trust": null, "errors": null, "lock": null, "sourceLink": null, "vars": {"key": "None", "keys": ["<ProjectKey at 0x7f0012d31ae0: id=3361162, project_id=****************, public_key=\'8096fec1d7cadc3a6830e7a2361fc1aa\'>"], "project": "<Project at 0x7f0012d31e70: id=****************, team_id=None, name=\'leebet-client\', slug=\'leebet-client\'>", "self": "<getsentry.quotas.SubscriptionQuota object at 0x7f002c1d10f0>"}}, {"filename": "sentry/quotas/base.py", "absPath": "/usr/src/sentry/src/sentry/quotas/base.py", "module": "sentry.quotas.base", "package": null, "platform": null, "instructionAddr": null, "symbolAddr": null, "function": "get_abuse_quotas", "rawFunction": null, "symbol": null, "context": [[431, "            if not limit:"], [432, "                limit = org.get_option(quota.option)"], [433, "            if not limit:"], [434, "                limit = options.get(quota.option)"], [435, ""], [436, "            limit = _limit_from_settings(limit)"], [437, "            if limit is None:"], [438, "                # Unlimited."], [439, "                continue"], [440, ""], [441, "            # Negative limits in config mean a reject-all quota."]], "lineNo": 436, "colNo": null, "inApp": true, "trust": null, "errors": null, "lock": null, "sourceLink": null, "vars": {"abuse_quotas": ["AbuseQuota(id=\'pae\', option=\'project-abuse-quota.error-limit\', categories=[<DataCategory.DEFAULT: 0>, <DataCategory.ERROR: 1>, <DataCategory.SECURITY: 3>], scope=<QuotaScope.PROJECT: 2>, compat_option_org=\'sentry:project-error-limit\', compat_option_sentry=\'getsentry.rate-limit.project-errors\')", "AbuseQuota(id=\'pati\', option=\'project-abuse-quota.transaction-limit\', categories=[<DataCategory.TRANSACTION_INDEXED: 9>], scope=<QuotaScope.PROJECT: 2>, compat_option_org=\'sentry:project-transaction-limit\', compat_option_sentry=\'getsentry.rate-limit.project-transactions\')", "AbuseQuota(id=\'paa\', option=\'project-abuse-quota.attachment-limit\', categories=[<DataCategory.ATTACHMENT: 4>], scope=<QuotaScope.PROJECT: 2>, compat_option_org=None, compat_option_sentry=None)", "AbuseQuota(id=\'pas\', option=\'project-abuse-quota.session-limit\', categories=[<DataCategory.SESSION: 5>], scope=<QuotaScope.PROJECT: 2>, compat_option_org=None, compat_option_sentry=None)", "AbuseQuota(id=\'oam\', option=\'organization-abuse-quota.metric-bucket-limit\', categories=[<DataCategory.METRIC_BUCKET: 15>], scope=<QuotaScope.ORGANIZATION: 1>, compat_option_org=None, compat_option_sentry=None)"], "abuse_window": "10", "global_abuse_window": "10", "limit": ["<BillingMetricHistory at 0x7f0011e71570: id=139993042, billing_history_id=672032613, billing_metric=2, total=11412770525>", "<BillingMetricHistory at 0x7f0011e71870: id=139993043, billing_history_id=672032613, billing_metric=7, total=401607>", "<BillingMetricHistory at 0x7f0011e70bb0: id=139993041, billing_history_id=672032613, billing_metric=1, total=17391001>", "<BillingMetricHistory at 0x7f0011e70e80: id=139993044, billing_history_id=672032613, billing_metric=4, total=235089477>"], "org": "<Organization at 0x7f0012d30850: id=****************, owner_id=None, name=\'Satoshi Tech\', slug=\'satoshi-tech\'>", "quota": "AbuseQuota(id=\'paa\', option=\'project-abuse-quota.attachment-limit\', categories=[<DataCategory.ATTACHMENT: 4>], scope=<QuotaScope.PROJECT: 2>, compat_option_org=None, compat_option_sentry=None)", "reason_codes": {"QuotaScope.ORGANIZATION": "\'org_abuse_limit\'", "QuotaScope.PROJECT": "\'project_abuse_limit\'"}, "self": "<getsentry.quotas.Su..."}}, {"filename": "sentry/quotas/base.py", "absPath": "/usr/src/sentry/src/sentry/quotas/base.py", "module": "sentry.quotas.base", "package": null, "platform": null, "instructionAddr": null, "symbolAddr": null, "function": "_limit_from_settings", "rawFunction": null, "symbol": null, "context": [[193, "    \\"\\"\\""], [194, "    limit=0 (or any falsy value) in database means \\"no limit\\". Convert that to"], [195, "    limit=None as limit=0 in code means \\"reject all\\"."], [196, "    \\"\\"\\""], [197, ""], [198, "    return int(x or 0) or None"], [199, ""], [200, ""], [201, "@dataclass"], [202, "class SeatAssignmentResult:"], [203, "    assignable: bool"]], "lineNo": 198, "colNo": null, "inApp": true, "trust": null, "errors": null, "lock": null, "sourceLink": null, "vars": {"x": ["<BillingMetricHistory at 0x7f0011e71570: id=139993042, billing_history_id=672032613, billing_metric=2, total=11412770525>", "<BillingMetricHistory at 0x7f0011e71870: id=139993043, billing_history_id=672032613, billing_metric=7, total=401607>", "<BillingMetricHistory at 0x7f0011e70bb0: id=139993041, billing_history_id=672032613, billing_metric=1, total=17391001>", "<BillingMetricHistory at 0x7f0011e70e80: id=139993044, billing_history_id=672032613, billing_metric=4, total=235089477>"]}}], "framesOmitted": null, "registers": null, "hasSystemFrames": true}, "rawStacktrace": null}], "hasSystemFrames": true, "excOmitted": null}, "type": "exception"}, {"data": {"values": [{"type": "default", "timestamp": "2024-01-04T16:20:40.478662Z", "level": "info", "message": "[Filtered]", "category": "query", "data": null, "event_id": null, "messageFormat": "sql", "messageRaw": "[Filtered]"}, {"type": "default", "timestamp": "2024-01-04T16:20:40.480437Z", "level": "info", "message": "SELECT sentry_project.id, sentry_project.slug, sentry_project.name,\\n       sentry_project.forced_color, sentry_project.organization_id, sentry_project.public,\\n       sentry_project.date_added, sentry_project.status, sentry_project.first_event,\\n       sentry_project.flags, sentry_project.platform\\nFROM sentry_project\\nWHERE sentry_project.id = %s\\nLIMIT 21", "category": "query", "data": null, "event_id": null, "messageFormat": "sql", "messageRaw": "SELECT \\"sentry_project\\".\\"id\\", \\"sentry_project\\".\\"slug\\", \\"sentry_project\\".\\"name\\", \\"sentry_project\\".\\"forced_color\\", \\"sentry_project\\".\\"organization_id\\", \\"sentry_project\\".\\"public\\", \\"sentry_project\\".\\"date_added\\", \\"sentry_project\\".\\"status\\", \\"sentry_project\\".\\"first_event\\", \\"sentry_project\\".\\"flags\\", \\"sentry_project\\".\\"platform\\" FROM \\"sentry_project\\" WHERE \\"sentry_project\\".\\"id\\" = %s LIMIT 21"}, {"type": "redis", "timestamp": "2024-01-04T16:20:40.543642Z", "level": "info", "message": "DEL \'relayconfig-debounce:k:8096fec1d7cadc3a6830e7a2361fc1aa\'", "category": "redis", "data": {"db.operation": "DEL", "redis.command": "DEL", "redis.is_cluster": true, "redis.key": "relayconfig-debounce:k:8096fec1d7cadc3a6830e7a2361fc1aa"}, "event_id": null}]}, "type": "breadcrumbs"}]"""
                )
            )
        ],
    )

    print(issue_details.json())
