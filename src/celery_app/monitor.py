import contextlib
import dataclasses
import datetime
from collections import defaultdict
from functools import cached_property
from typing import ContextManager, Mapping

import sentry_sdk
import sentry_sdk.metrics as metrics
from celery import Celery
from celery.events import Event, EventReceiver, state
from flask import Flask

from seer.bootup import bootup, bootup_celery


@dataclasses.dataclass
class CeleryMonitor:
    app: Celery
    last_publish: datetime.datetime = dataclasses.field(default_factory=datetime.datetime.now)
    publish_interval: datetime.timedelta = dataclasses.field(default=datetime.timedelta(seconds=10))

    @cached_property
    def state(self) -> state.State:
        return self.app.events.State()

    def handle_event(self, event: Event):
        self.state.event(event)
        group, _, subject = event["type"].partition("-")
        if group == "task":
            if "runtime" in event:
                task: state.Task = self.state.tasks[event["uuid"]]
                runtime = event["runtime"]
                metrics.timing(
                    "celery.tasks.runtime.seconds",
                    runtime,
                    unit="seconds",
                    tags=dict(task_name=task.name, broker_url=self.app.conf.broker_url),
                )

        if self.last_publish + self.publish_interval > datetime.datetime.now():
            self.last_publish = datetime.datetime.now()
            self.publish_metrics()

    def publish_metrics(self):
        state_tagset: dict[str, str] = {"broker_url": self.app.conf.broker_url}

        with sentry_sdk.start_transaction(
            op="seer.celery_app.monitor.publish_metrics",
            description="Publishing metrics for the celery monitor",
        ):
            workers: Mapping[str, state.Worker] = self.state.workers
            task_count_by_state = self.tasks_by_state()

            sum_alive_workers = sum(1 for worker in workers.values() if worker.alive)
            metrics.gauge("celery.workers.alive.count", sum_alive_workers, tags=state_tagset)

            for s, count in task_count_by_state.items():
                metrics.gauge(f"celery.tasks.{s}.count", count, tags=state_tagset)

            self.state.clear()

    def tasks_by_state(self):
        tasks: Mapping[str, state.Task] = self.state.tasks
        task_count_by_state: dict[str, int] = defaultdict(lambda: 0)
        for task in tasks.values():
            task_count_by_state[task.state] += 1
        return task_count_by_state

    def run(self) -> ContextManager[EventReceiver]:
        @contextlib.contextmanager
        def inner():
            with self.app.connection_for_read() as conn:
                receiver = EventReceiver(conn, {"*": self.handle_event})
                yield receiver

        return inner()


if __name__ == "__main__":
    bootup(Flask(__name__), [], init_db=False, async_load_models=True, init_migrations=False)

    try:
        with CeleryMonitor(app=bootup_celery()).run() as receiver:
            receiver.capture()
    except Exception as e:
        sentry_sdk.capture_exception(e)
        raise
