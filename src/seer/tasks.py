import abc
import asyncio
import dataclasses
import datetime
from asyncio import Future, Task
from typing import Any, Callable, Coroutine, Protocol, TypeVar

import celery.result
from dateutil.relativedelta import relativedelta
from sentry_sdk.integrations.asyncio import AsyncioIntegration
from sqlalchemy.ext.asyncio import async_sessionmaker

from seer.db import ProcessRequest

_A = TypeVar("_A")

AsyncSession = async_sessionmaker(expire_on_commit=False)


class QConsumer(Protocol):
    def get(self, block: bool = True, timeout: int | None = None) -> Any:
        ...


@dataclasses.dataclass(frozen=True, order=True)
class Period:
    period_start: datetime.date
    period: relativedelta

    @property
    def period_end(self) -> datetime.date:
        return self.period_start + self.period - datetime.timedelta(days=1)

    def fit_to(self, date: datetime.date) -> "Period":
        result = self

        while date < result.period_start:
            result = result.prev()

        while date > result.period_end:
            result = result.next()

        return result

    def next(self) -> "Period":
        return Period(period_start=self.period_start + self.period, period=self.period)

    def prev(self) -> "Period":
        return Period(period_start=self.period_start - self.period, period=self.period)


class AsyncTaskFactory(abc.ABC):
    @abc.abstractmethod
    def matches(self, process_request: ProcessRequest) -> bool:
        pass

    @abc.abstractmethod
    async def invoke(self, process_request: ProcessRequest) -> None:
        pass

    def await_celery_job(self, result: celery.result.AsyncResult):
        pass


_async_task_factories: list[Callable[[], AsyncTaskFactory]] = []


def async_task_factory(f: Callable[[], AsyncTaskFactory]) -> Callable[[], AsyncTaskFactory]:
    _async_task_factories.append(f)
    return f


@dataclasses.dataclass
class AsyncApp:
    end_event: asyncio.Event = dataclasses.field(default_factory=lambda: asyncio.Event())
    num_consumers: int = 10
    queue: asyncio.Queue = dataclasses.field(default_factory=lambda: asyncio.Queue())
    task_factories: list[Callable[[], AsyncTaskFactory]] = dataclasses.field(
        default_factory=lambda: _async_task_factories,
    )
    consumer_sleep: int = 5

    async def run_or_end(self, c: Future[_A] | Coroutine[Any, Any, _A]) -> tuple[_A] | None:
        end_task = asyncio.create_task(self.end_event.wait())
        task: Future[_A] | Task[_A]
        if not isinstance(c, Future):
            task = asyncio.create_task(c)
        else:
            task = c
        parts: list[Future | Task] = [end_task, task]
        await asyncio.wait(parts, return_when=asyncio.FIRST_COMPLETED)
        if end_task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            return None
        else:
            end_task.cancel()
        if exc := task.exception():
            raise exc
        return (task.result(),)

    async def select_from_db(self) -> None:
        while not self.end_event.is_set():
            async with AsyncSession() as session:
                result = await self.run_or_end(
                    session.run_sync(
                        lambda session: ProcessRequest.acquire_work(
                            10, datetime.datetime.utcnow(), session=session
                        )
                    )
                )
            if result is not None:
                for item in result[0]:
                    await self.run_or_end(self.queue.put(item))
                    if self.end_event.is_set():
                        break
            else:
                await self.run_or_end(asyncio.sleep(self.consumer_sleep))

    async def producer_loop(self):
        await asyncio.gather(self.select_from_db())

    async def consumer_loop(self):
        while not self.end_event.is_set():
            result = await self.run_or_end(self.queue.get())
            if result is None:
                continue

            item: ProcessRequest = result[0]
            for factory in self.task_factories:
                task = factory()
                accept = task.matches(item)
                if accept:
                    await task.invoke(item)

                    # If this was persisted work, clean it up.
                    if item.id:
                        async with AsyncSession() as session:
                            await session.execute(item.mark_completed_stmt())
                            await session.commit()
                    break

    async def run(self):
        producer_task = asyncio.create_task(self.producer_loop())
        consumer_tasks: list[asyncio.Task[Any]] = []
        while not self.end_event.is_set():
            if producer_task.done():
                # Self terminate if the producer dies for any reason.
                self.end_event.set()
                continue

            task: asyncio.Task
            for task in [*consumer_tasks]:
                if task.done():
                    # Unexpected death of a task?  Reboot it.
                    consumer_tasks.remove(task)

            while len(consumer_tasks) < self.num_consumers:
                consumer_tasks.append(asyncio.create_task(self.consumer_loop()))
            await self.run_or_end(asyncio.sleep(1))

        all_tasks = [producer_task, *consumer_tasks]
        async with asyncio.timeout(10):
            await asyncio.gather(*all_tasks)


def async_main():
    from seer.bootup import bootup

    bootup(__name__, [AsyncioIntegration()])
    app = AsyncApp()
    asyncio.run(app.run())
