import logging
import os
import xml.etree.ElementTree as ET

import torch
from github import Auth, Github

from ..agent.agent import GptAgent, Message, Usage
from .agent_context import AgentContext
from .prompts import action_prompt, planning_prompt
from .tools import BaseTools, CodeActionTools
from .types import AutofixInput, AutofixOutput, IssueDetails, PlanningOutput, SentryEvent

REPO_NAME = "getsentry/sentry-mirror-suggested-fix"
auth = Auth.Token(os.getenv("GITHUB_TOKEN"))

logger = logging.getLogger(__name__)


class Autofix:
    def __init__(self, issue_details: IssueDetails, autofix_input: AutofixInput):
        self.issue_details = issue_details
        self.autofix_input = autofix_input
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")
        self.agent_context = AgentContext(
            repo_name=REPO_NAME,
            ref="heads/master",
            model="gpt-4-1106-preview",
            github_auth=auth,
            persist_path="/Users/jennmueng/Documents/code/tmp-indices/indices",
            tmp_dir="/Users/jennmueng/Documents/code/tmp",
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

        autofix_output = AutofixOutput(
            **planning_output.model_dump(),
            changes=coding_output.values(),
            usage=Usage(
                prompt_tokens=planning_output.usage.prompt_tokens + coding_usage.prompt_tokens,
                completion_tokens=planning_output.usage.completion_tokens
                + coding_usage.completion_tokens,
                total_tokens=planning_output.usage.total_tokens + coding_usage.total_tokens,
            ),
        )

        self._create_pr(autofix_output)

        return autofix_output

    def _create_pr(self, autofix_output: AutofixOutput):
        branch_ref = self.agent_context.create_branch_from_changes(
            autofix_output.changes, self.agent_context.base_sha
        )

        self.agent_context.create_pr_from_branch(branch_ref, autofix_output, self.issue_details)

    def _run_planning_agent(self, index, documents, nodes) -> PlanningOutput:
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
            title = parsed_output.find("title").text
            description = parsed_output.find("description").text
            plan = parsed_output.find("plan").text

            return PlanningOutput(
                title=title, description=description, plan=plan, usage=planning_agent.usage
            )
        except AttributeError:
            logger.warning(f"Planning response does not contain a title, description, or plan")
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
    import logging

    logging.basicConfig(level=logging.DEBUG)

    issue_details = IssueDetails(
        id="4595888126",
        title="TypeError: 'int' object is not iterable",
        events=[
            SentryEvent(
                entries=json.loads(
                    '[{"data": {"values": [{"type": "TypeError", "value": "\'int\' object is not iterable", "mechanism": {"type": "celery", "handled": false}, "threadId": null, "module": null, "stacktrace": {"frames": [{"filename": "celery/app/trace.py", "absPath": "/usr/local/lib/python3.10/site-packages/celery/app/trace.py", "module": "celery.app.trace", "package": null, "platform": null, "instructionAddr": null, "symbolAddr": null, "function": "__protected_call__", "rawFunction": null, "symbol": null, "context": [[755, "            stack = self.request_stack"], [756, "            req = stack.top"], [757, "            if req and not req._protected and \\\\"], [758, "                    len(stack) == 1 and not req.called_directly:"], [759, "                req._protected = 1"], [760, "                return self.run(*args, **kwargs)"], [761, "            return orig(self, *args, **kwargs)"], [762, "        BaseTask.__call__ = __protected_call__"], [763, "        BaseTask._stackprotected = True"]], "lineNo": 760, "colNo": null, "inApp": false, "trust": null, "errors": null, "lock": null, "sourceLink": null, "vars": {"args": [], "kwargs": {"__start_time": "1704328598.143453", "public_key": "\'7d3b3369fc12f0fd74c961d7c7944a97\'", "tmp_scheduled": "1704328598.142824"}, "orig": "<function Task.__call__ at 0x7f81245aae60>", "req": "<Context: {\'lang\': None, \'task\': \'sentry.tasks.relay.build_project_config\', \'id\': \'619a29c5-1fa7-4ec0-af84-712fd5a1e292\', \'root_id\': None, \'parent_id\': None, \'group\': None, \'meth\': None, \'shadow\': None, \'eta\': None, \'expires\': \'2024-01-04T00:37:08.143666+00:00\', \'retries\': 0, \'timelimit\': (10, 5), \'argsrepr\': None, \'kwargsrepr\': None, \'origin\': None, \'sentry-trace\': \'13589d3a3a204c3dbf2f629ce0453584-8f786cb12ccf78ce-0\', \'baggage\': \'sentry-trace_id=13589d3a3a204c3dbf2f629ce0453584,sentry-environment=prod,sentry-release=backend%40f769def0c5d3b148a4b8f18789d06a270c0b57d9,sentry-public_key=8df96f046b994933aafa8dcfa3345419,sentry-transaction=RelayProjectConfigsEndpoint,sentry-sample_rate=0.0,sentry-sampled=false\', \'sentry-monitor-start-timestamp-s\': \'1704328598.143541098\', \'headers\': {\'sentry-trace\': \'13589d3a3a204c3dbf2f629ce0453584-8f786cb12ccf78ce-0\', \'baggage\': \'sentry-trace_id=13589d3a3a204c3dbf2f629ce0453584,sentry-environment=prod,sentry-release=backend%40f769def0c5d3b148a4b8f18789d06a270c0b57d9,sentry-p...", "self": "<@task: sentry.tasks.relay.build_project_config of sentry at 0x7f812454c1f0>", "stack": "<celery.utils.threads._LocalStack object at 0x7f80c0e3aad0>"}}, {"filename": "sentry/tasks/base.py", "absPath": "/usr/src/sentry/src/sentry/tasks/base.py", "module": "sentry.tasks.base", "package": null, "platform": null, "instructionAddr": null, "symbolAddr": null, "function": "_wrapped", "rawFunction": null, "symbol": null, "context": [[112, "                scope.set_tag(\\"transaction_id\\", transaction_id)"], [113, ""], [114, "            with metrics.timer(key, instance=instance), track_memory_usage("], [115, "                \\"jobs.memory_change\\", instance=instance"], [116, "            ):"], [117, "                result = func(*args, **kwargs)"], [118, ""], [119, "            return result"], [120, ""], [121, "        # We never use result backends in Celery. Leaving `trail=True` means that if we schedule"], [122, "        # many tasks from a parent task, each task leaks memory. This can lead to the scheduler"]], "lineNo": 117, "colNo": null, "inApp": true, "trust": null, "errors": null, "lock": null, "sourceLink": null, "vars": {"args": [], "func": "<function build_project_config at 0x7f80e9d33010>", "instance": "\'sentry.tasks.relay.build_project_config\'", "key": "\'jobs.duration\'", "kwargs": {"public_key": "\'7d3b3369fc12f0fd74c961d7c7944a97\'", "tmp_scheduled": "1704328598.142824"}, "name": "\'sentry.tasks.relay.build_project_config\'", "record_timing": "False", "scope": "<Scope id=0x7f80b0feba00 name=celery>", "start_time": "1704328598.143453", "transaction_id": "None"}}, {"filename": "sentry/tasks/relay.py", "absPath": "/usr/src/sentry/src/sentry/tasks/relay.py", "module": "sentry.tasks.relay", "package": null, "platform": null, "instructionAddr": null, "symbolAddr": null, "function": "build_project_config", "rawFunction": null, "symbol": null, "context": [[46, "            # In this particular case, where a project key got deleted and"], [47, "            # triggered an update, we know that key doesn\'t exist and we want to"], [48, "            # avoid creating more tasks for it."], [49, "            projectconfig_cache.backend.set_many({public_key: {\\"disabled\\": True}})"], [50, "        else:"], [51, "            config = compute_projectkey_config(key)"], [52, "            projectconfig_cache.backend.set_many({public_key: config})"], [53, ""], [54, "    finally:"], [55, "        # Delete the key in this `finally` block to make sure the debouncing key"], [56, "        # is always deleted. Deleting the key at the end of the task also makes"]], "lineNo": 51, "colNo": null, "inApp": true, "trust": null, "errors": null, "lock": null, "sourceLink": null, "vars": {"ProjectKey": "<class \'sentry.models.projectkey.ProjectKey\'>", "key": "<ProjectKey at 0x7f80b0dddf30: id=3359512, project_id=****************, public_key=\'7d3b3369fc12f0fd74c961d7c7944a97\'>", "kwargs": {"tmp_scheduled": "1704328598.142824"}, "public_key": "\'7d3b3369fc12f0fd74c961d7c7944a97\'"}}, {"filename": "sentry/tasks/relay.py", "absPath": "/usr/src/sentry/src/sentry/tasks/relay.py", "module": "sentry.tasks.relay", "package": null, "platform": null, "instructionAddr": null, "symbolAddr": null, "function": "compute_projectkey_config", "rawFunction": null, "symbol": null, "context": [[188, "    from sentry.relay.config import get_project_config"], [189, ""], [190, "    if key.status != ProjectKeyStatus.ACTIVE:"], [191, "        return {\\"disabled\\": True}"], [192, "    else:"], [193, "        return get_project_config(key.project, project_keys=[key], full_config=True).to_dict()"], [194, ""], [195, ""], [196, "@instrumented_task("], [197, "    name=\\"sentry.tasks.relay.invalidate_project_config\\","], [198, "    queue=\\"relay_config_bulk\\","]], "lineNo": 193, "colNo": null, "inApp": true, "trust": null, "errors": null, "lock": null, "sourceLink": null, "vars": {"ProjectKeyStatus": "<class \'sentry.models.projectkey.ProjectKeyStatus\'>", "get_project_config": "<function get_project_config at 0x7f80d9c14ee0>", "key": "<ProjectKey at 0x7f80b0dddf30: id=3359512, project_id=****************, public_key=\'7d3b3369fc12f0fd74c961d7c7944a97\'>"}}, {"filename": "sentry/relay/config/__init__.py", "absPath": "/usr/src/sentry/src/sentry/relay/config/__init__.py", "module": "sentry.relay.config", "package": null, "platform": null, "instructionAddr": null, "symbolAddr": null, "function": "get_project_config", "rawFunction": null, "symbol": null, "context": [[210, "    :return: a ProjectConfig object for the given project"], [211, "    \\"\\"\\""], [212, "    with sentry_sdk.push_scope() as scope:"], [213, "        scope.set_tag(\\"project\\", project.id)"], [214, "        with metrics.timer(\\"relay.config.get_project_config.duration\\"):"], [215, "            return _get_project_config(project, full_config=full_config, project_keys=project_keys)"], [216, ""], [217, ""], [218, "def get_dynamic_sampling_config(project: Project) -> Optional[Mapping[str, Any]]:"], [219, "    if features.has(\\"organizations:dynamic-sampling\\", project.organization):"], [220, "        # For compatibility reasons we want to return an empty list of old rules. This has been done in order to make"]], "lineNo": 215, "colNo": null, "inApp": true, "trust": null, "errors": null, "lock": null, "sourceLink": null, "vars": {"full_config": "True", "project": "<Project at 0x7f80b0dddb70: id=****************, team_id=None, name=\'i18n-smartling-sync\', slug=\'i18n-smartling-sync\'>", "project_keys": ["<ProjectKey at 0x7f80b0dddf30: id=3359512, project_id=****************, public_key=\'7d3b3369fc12f0fd74c961d7c7944a97\'>"], "scope": "<Scope id=0x7f80aed22440 name=celery>"}}, {"filename": "sentry/relay/config/__init__.py", "absPath": "/usr/src/sentry/src/sentry/relay/config/__init__.py", "module": "sentry.relay.config", "package": null, "platform": null, "instructionAddr": null, "symbolAddr": null, "function": "_get_project_config", "rawFunction": null, "symbol": null, "context": [[633, "    with Hub.current.start_span(op=\\"get_event_retention\\"):"], [634, "        event_retention = quotas.backend.get_event_retention(project.organization)"], [635, "        if event_retention is not None:"], [636, "            config[\\"eventRetention\\"] = event_retention"], [637, "    with Hub.current.start_span(op=\\"get_all_quotas\\"):"], [638, "        if quotas_config := get_quotas(project, keys=project_keys):"], [639, "            config[\\"quotas\\"] = quotas_config"], [640, ""], [641, "    return ProjectConfig(project, **cfg)"], [642, ""], [643, ""]], "lineNo": 638, "colNo": null, "inApp": true, "trust": null, "errors": null, "lock": null, "sourceLink": null, "vars": {"cfg": {"config": {"allowedDomains": ["\'*\'"], "breakdownsV2": {"span_ops": "{\\"matches\\":\\"[\'http\', \'db\', \'browser\', \'resource\', \'ui\']\\",\\"type\\":\\"\'spanOperations\'\\"}"}, "datascrubbingSettings": {"excludeFields": "[\\"\'fullStoryUrl\'\\",\\"\'heapReplayUrl\'\\"]", "scrubData": "True", "scrubDefaults": "True", "sensitiveFields": "[Filtered]"}, "features": ["\'projects:span-metrics-extraction\'", "\'projects:span-metrics-extraction-resource\'", "\'organizations:transaction-name-mark-scrubbed-as-sanitized\'", "\'organizations:transaction-name-normalize\'", "\'organizations:profiling\'", "\'organizations:session-replay\'", "\'organizations:user-feedback-ingest\'", "\'organizations:session-replay-recording-scrubbing\'", "\'organizations:device-class-synthesis\'"], "metricConditionalTagging": ["{\\"condition\\":\\"{\'op\': \'and\', \'inner\': [{\'op\': \'gt\', \'name\': \'event.duration\', \'value\': 1200}]}\\",\\"tagValue\\":\\"\'frustrated\'\\",\\"targetMetrics\\":\\"(\'s:transactions/user@none\', \'d:transactions/duration@millisecond\', \'d:transactions/measurements.lcp@millisecond\')\\",\\"targetTag\\":\\"\'satisfaction\'\\"}", "{\\"condition\\":\\"{\'op\': \'and\', \'inner\': [{\'op\': \'gt\', \'name\': \'event.duration\', \'value\': 300}]}\\",\\"tagValue\\":\\"\'tolerated\'\\",\\"targetMetrics\\":\\"(\'s:transactions/user@none\', \'d:transactions/duration@millisecond\', \'d:transactions/measurements.lcp@millisecond\')\\",\\"targetTag\\":\\"\'satisfaction\'\\"}", "{\\"condition\\":\\"{\'op\': \'and\', \'inner\': []}\\",\\"tagValue\\":\\"\'satisfied\'\\",\\"targetMetrics\\":\\"(\'s:transactions/user@none\', \'d:transactions/duration@millisecond\', \'d:transactions/measurements.lcp@millisecond\')\\",\\"targetTag\\":\\"\'satisfaction\'\\"}", "{\\"condition\\":\\"{\'op\': \'and\', \'inner\': [{\'op\': \'eq\', \'name\': \'event.contexts.trace.op\', \'value\': \'pageload\'}, {\'op\': \'eq\', \'name\': \'event.platform\', \'value\': \'javascript\'}, {\'op\': \'gte\', \'name\': \'event.duration\', \'value\': 16123.0}]}\\",\\"tagValue\\":\\"\'outlier\'\\",\\"targetMetrics\\":\\"[\'d:transactions/duration@millisecond\']\\",\\"targetTag\\":\\"\'histogram_outlier\'\\"}", "{\\"condition\\":\\"{\'op\': \'and\', \'inner\': [{\'op\': \'eq\', \'name\': \'event.contexts.trace.op\', \'value\': \'pageload\'}, {\'op\': \'eq\', \'name\': \'event.platform\', \'value\': \'javascript\'}, {\'op\': \'gte\', \'name\': \'event.duration\', \'value\': 7941.899538040161}]}\\",\\"tagValue\\":\\"\'outlier\'\\",\\"targetMetrics\\":\\"[\'d:transactions/me..."]}}}}, {"filename": "sentry/relay/config/__init__.py", "absPath": "/usr/src/sentry/src/sentry/relay/config/__init__.py", "module": "sentry.relay.config", "package": null, "platform": null, "instructionAddr": null, "symbolAddr": null, "function": "get_quotas", "rawFunction": null, "symbol": null, "context": [[181, ""], [182, ""], [183, "def get_quotas(project: Project, keys: Optional[Sequence[ProjectKey]] = None) -> List[str]:"], [184, "    try:"], [185, "        computed_quotas = ["], [186, "            quota.to_json() for quota in quotas.backend.get_quotas(project, keys=keys)"], [187, "        ]"], [188, "    except BaseException:"], [189, "        metrics.incr(\\"relay.config.get_quotas\\", tags={\\"success\\": False}, sample_rate=1.0)"], [190, "        raise"], [191, "    else:"]], "lineNo": 186, "colNo": null, "inApp": true, "trust": null, "errors": null, "lock": null, "sourceLink": null, "vars": {"keys": ["<ProjectKey at 0x7f80b0dddf30: id=3359512, project_id=****************, public_key=\'7d3b3369fc12f0fd74c961d7c7944a97\'>"], "project": "<Project at 0x7f80b0dddb70: id=****************, team_id=None, name=\'i18n-smartling-sync\', slug=\'i18n-smartling-sync\'>"}}, {"filename": "getsentry/quotas.py", "absPath": "/usr/src/getsentry/getsentry/quotas.py", "module": "getsentry.quotas", "package": null, "platform": null, "instructionAddr": null, "symbolAddr": null, "function": "get_quotas", "rawFunction": null, "symbol": null, "context": [[420, "                subscription=sub,"], [421, "            )"], [422, "        )"], [423, ""], [424, "        quotas.extend("], [425, "            self._set_spike_protection(organization=organization, project=project, subscription=sub)"], [426, "        )"], [427, ""], [428, "        return quotas"], [429, ""], [430, "    def get_event_retention(self, organization):"]], "lineNo": 425, "colNo": null, "inApp": true, "trust": null, "errors": null, "lock": null, "sourceLink": null, "vars": {"_": "\'profiles\'", "args": [], "billing_metric": "6", "kwargs": {"keys": ["<ProjectKey at 0x7f80b0dddf30: id=3359512, project_id=****************, public_key=\'7d3b3369fc12f0fd74c961d7c7944a97\'>"]}, "metric_histories": {"1": "<BillingMetricHistory at 0x7f80b937dff0: id=143459193, billing_history_id=672904809, billing_metric=1, total=273424>", "2": "<BillingMetricHistory at 0x7f80b937ea10: id=143459194, billing_history_id=672904809, billing_metric=2, total=686422>", "4": "<BillingMetricHistory at 0x7f80b937c460: id=143459196, billing_history_id=672904809, billing_metric=4, total=0>", "7": "<BillingMetricHistory at 0x7f80b937f5e0: id=143459195, billing_history_id=672904809, billing_metric=7, total=0>"}, "organization": "<Organization at 0x7f80b0ddfdf0: id=19233, owner_id=None, name=\'Klaviyo\', slug=\'klaviyo-1\'>", "project": "<Project at 0x7f80b0dddb70: id=****************, team_id=None, name=\'i18n-smartling-sync\', slug=\'i18n-smartling-sync\'>", "quotas": ["<sentry.quotas.base.QuotaConfig object at 0x7f80ae9db0a0>", "<sentry.quotas.base.QuotaConfig object at 0x7f80ae9db400>", "<sentry.quotas.base.QuotaConfig object at 0x7f80aec6dba0>"], "self": "<getsentry.quotas.SubscriptionQuota object at 0x7f80d8317130>", "sub": "<Subscription at 0x7f80b937c2e0: id=10406, organization_id=19233, plan=\'am2_business_ent_auf\', status=\'active\'>"}}, {"filename": "getsentry/quotas.py", "absPath": "/usr/src/getsentry/getsentry/quotas.py", "module": "getsentry.quotas", "package": null, "platform": null, "instructionAddr": null, "symbolAddr": null, "function": "_set_spike_protection", "rawFunction": null, "symbol": null, "context": [[687, "        if ProjectOption.objects.get_value("], [688, "            project, \\"quotas:spike-protection-disabled\\", default=False"], [689, "        ):"], [690, "            return quotas"], [691, ""], [692, "        metric_histories = self._get_metric_histories(subscription)"], [693, "        for billing_metric in get_spike_protected_categories(subscription=subscription):"], [694, "            metric_history = metric_histories.get(billing_metric)"], [695, ""], [696, "            if not metric_history or metric_history.reserved == UNLIMITED_QUOTA:"], [697, "                # Skip if the quotas are unlimited"]], "lineNo": 692, "colNo": null, "inApp": true, "trust": null, "errors": null, "lock": null, "sourceLink": null, "vars": {"organization": "<Organization at 0x7f80b0ddfdf0: id=19233, owner_id=None, name=\'Klaviyo\', slug=\'klaviyo-1\'>", "project": "<Project at 0x7f80b0dddb70: id=****************, team_id=None, name=\'i18n-smartling-sync\', slug=\'i18n-smartling-sync\'>", "quotas": [], "self": "<getsentry.quotas.SubscriptionQuota object at 0x7f80d8317130>", "subscription": "<Subscription at 0x7f80b937c2e0: id=10406, organization_id=19233, plan=\'am2_business_ent_auf\', status=\'active\'>"}}, {"filename": "getsentry/quotas.py", "absPath": "/usr/src/getsentry/getsentry/quotas.py", "module": "getsentry.quotas", "package": null, "platform": null, "instructionAddr": null, "symbolAddr": null, "function": "_get_metric_histories", "rawFunction": null, "symbol": null, "context": [[446, "        cached = cache.get(cache_key)"], [447, "        if cached is not None:"], [448, "            metrics.incr(\\"quotas.metric_histories.cache\\", tags={\\"hit\\": \\"true\\"})"], [449, "            # Rebuild the defaultdict as we couldn\'t pickle the dict."], [450, "            hydrated = defaultdict(lambda: BillingMetricHistory())"], [451, "            for obj in cached:"], [452, "                hydrated[obj.billing_metric] = obj"], [453, "            return hydrated"], [454, ""], [455, "        # Store only the values as the container is a defaultdict"], [456, "        # and we can\'t pickle the lambda default value."]], "lineNo": 451, "colNo": null, "inApp": true, "trust": null, "errors": null, "lock": null, "sourceLink": null, "vars": {"cache_key": "\'1:quota_metric_history:10406\'", "cache_ttl": "120", "cached": "0", "hydrated": {}, "self": "<getsentry.quotas.SubscriptionQuota object at 0x7f80d8317130>", "subscription": "<Subscription at 0x7f80b937c2e0: id=10406, organization_id=19233, plan=\'am2_business_ent_auf\', status=\'active\'>"}}], "framesOmitted": null, "registers": null, "hasSystemFrames": true}, "rawStacktrace": null}], "hasSystemFrames": true, "excOmitted": null}, "type": "exception"}, {"data": {"values": [{"type": "default", "timestamp": "2024-01-04T00:36:38.146629Z", "level": "info", "message": "[Filtered]", "category": "query", "data": null, "event_id": null, "messageFormat": "sql", "messageRaw": "[Filtered]"}, {"type": "default", "timestamp": "2024-01-04T00:36:38.148756Z", "level": "info", "message": "SELECT sentry_project.id, sentry_project.slug, sentry_project.name,\\n       sentry_project.forced_color, sentry_project.organization_id, sentry_project.public,\\n       sentry_project.date_added, sentry_project.status, sentry_project.first_event,\\n       sentry_project.flags, sentry_project.platform\\nFROM sentry_project\\nWHERE sentry_project.id = %s\\nLIMIT 21", "category": "query", "data": null, "event_id": null, "messageFormat": "sql", "messageRaw": "SELECT \\"sentry_project\\".\\"id\\", \\"sentry_project\\".\\"slug\\", \\"sentry_project\\".\\"name\\", \\"sentry_project\\".\\"forced_color\\", \\"sentry_project\\".\\"organization_id\\", \\"sentry_project\\".\\"public\\", \\"sentry_project\\".\\"date_added\\", \\"sentry_project\\".\\"status\\", \\"sentry_project\\".\\"first_event\\", \\"sentry_project\\".\\"flags\\", \\"sentry_project\\".\\"platform\\" FROM \\"sentry_project\\" WHERE \\"sentry_project\\".\\"id\\" = %s LIMIT 21"}, {"type": "redis", "timestamp": "2024-01-04T00:36:43.340587Z", "level": "info", "message": "DEL \'relayconfig-debounce:k:7d3b3369fc12f0fd74c961d7c7944a97\'", "category": "redis", "data": {"db.operation": "DEL", "redis.command": "DEL", "redis.is_cluster": true, "redis.key": "relayconfig-debounce:k:7d3b3369fc12f0fd74c961d7c7944a97"}, "event_id": null}]}, "type": "breadcrumbs"}]'
                )
            )
        ],
    )

    autofix = Autofix(issue_details, AutofixInput(additional_context=None))

    changes = autofix.run()

    print(changes)
