import abc
import asyncio
import contextlib
import dataclasses
import datetime
import hashlib
from asyncio import Future, Task
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Full, Queue
from threading import Event
from typing import Any, Callable, Coroutine, Protocol, TypeVar

import celery.result
import sqlalchemy
from dateutil.relativedelta import relativedelta
from sentry_sdk.integrations.asyncio import AsyncioIntegration
from sqlalchemy import func, select
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
    async def invoke(self, process_request: ProcessRequest):
        pass

    async def async_celery_job(self, cb: Callable[[], celery.result.AsyncResult]):
        loop = asyncio.get_running_loop()
        q: Queue = Queue(1)

        with ThreadPoolExecutor() as pool:
            ar = await loop.run_in_executor(pool, cb)

            def run():
                def on_message(raw: Any):
                    # Keep trying to put the item into the queue by removing existing item until we get in.
                    while True:
                        try:
                            q.put_nowait(raw)
                            return
                        except Full:
                            try:
                                q.get_nowait()
                            except Empty:
                                pass

                ar.get(on_message=on_message, propagate=True)

            complete = loop.run_in_executor(pool, run)

            try:
                while not complete.done():
                    get = loop.run_in_executor(pool, q.get)
                    await asyncio.wait([get, complete], return_when=asyncio.FIRST_COMPLETED)
                    if get.done():
                        v = await get
                        if v:
                            if v["status"] == "PROGRESS":
                                try:
                                    yield v["result"]
                                except Exception as e:
                                    if not complete.done():
                                        await loop.run_in_executor(
                                            pool,
                                            lambda: ar.revoke(
                                                terminate=True, signal="SIGUSR1", wait=False
                                            ),
                                        )
                                    raise e
                            # if v['status'] == 'FAILURE':
                            #     raise v['result']
            finally:
                loop.run_in_executor(pool, lambda: q.put_nowait(None))
        await complete


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

    async def invoke_task(self, task: AsyncTaskFactory, process_request: ProcessRequest):
        async with AsyncSession() as session, acquire_x_lock(
            process_request.name, session
        ) as acquired:
            if acquired:
                await task.invoke(process_request)
            return acquired

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
                    result = await self.run_or_end(self.invoke_task(task, item))

                    if result and result[0] is not False and item.id:
                        async with AsyncSession() as session:
                            await session.execute(item.mark_completed_stmt())
                            await session.commit()

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


@contextlib.asynccontextmanager
async def acquire_x_lock(name: str, session: sqlalchemy.ext.asyncio.AsyncSession):
    m = hashlib.sha256()
    m.update(name.encode("utf8"))
    key = int.from_bytes(m.digest()[:8], byteorder="big", signed=True)

    rows = await session.execute(select(func.pg_try_advisory_lock(key)))
    acquired = next(rows)[0]
    try:
        yield acquired
    finally:
        if acquired:
            await session.execute(select(func.pg_advisory_unlock(key)))


def async_main():
    from seer.bootup import bootup

    bootup(__name__, [AsyncioIntegration()])
    app = AsyncApp()
    asyncio.run(app.run())
