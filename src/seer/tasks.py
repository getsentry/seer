import asyncio
import dataclasses
import datetime
import multiprocessing
from asyncio import CancelledError, Future, Task
from concurrent.futures import ThreadPoolExecutor
from queue import Empty
from typing import Any, Callable, Coroutine, Protocol, TypeVar

from pydantic import BaseModel
from sentry_sdk.integrations.asyncio import AsyncioIntegration
from sqlalchemy.ext.asyncio import async_sessionmaker

from seer.db import ProcessRequest

_Request = TypeVar("_Request", bound=BaseModel)
_A = TypeVar("_A")

AsyncSession = async_sessionmaker(expire_on_commit=False)


class QConsumer(Protocol):
    def get(self, block: bool = True, timeout: int | None = None) -> Any:
        ...


class QProducer(Protocol):
    def put(self, obj: Any, block: bool = True, timeout: int | None = None) -> None:
        ...


class TaskFactory(Protocol[_Request]):
    """
    Represents a potential factory for mapping a process request to a local job to be invoked.
    """

    def from_process_request(self, process_request: ProcessRequest) -> _Request | None:
        """
        Selects a process request, usually by matching its name, to this task, returning a request
        object from that job if it is a match.
        """
        pass

    async def invoke(self, request: _Request):
        pass


_async_task_factories: list[TaskFactory[BaseModel]] = []


def async_task_factory(f: Callable[[], TaskFactory]) -> Callable[[], TaskFactory]:
    _async_task_factories.append(f())
    return f


@dataclasses.dataclass
class AsyncApp:
    end_event: asyncio.Event
    queue: asyncio.Queue
    io_work: QConsumer = dataclasses.field(default_factory=multiprocessing.Queue)
    cpu_work: QProducer = dataclasses.field(default_factory=multiprocessing.Queue)
    num_consumers: int = 10
    task_factories: list[TaskFactory[BaseModel]] = dataclasses.field(
        default_factory=lambda: [],
    )
    # workers = 2 => one for consuming io work, one for producing cpu_work
    threaded_executor: ThreadPoolExecutor = dataclasses.field(
        default_factory=lambda: ThreadPoolExecutor(max_workers=2)
    )

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
                await self.run_or_end(asyncio.sleep(5))

    def get_io_work(self) -> ProcessRequest:
        while not self.end_event.is_set():
            try:
                return self.io_work.get(True, 1)
            except Empty:
                pass
        raise CancelledError()

    async def select_from_local_work(self) -> None:
        while not self.end_event.is_set():
            result = await self.run_or_end(
                asyncio.get_running_loop().run_in_executor(self.threaded_executor, self.get_io_work)
            )
            if result is not None:
                await self.run_or_end(self.queue.put(result[0]))

    async def producer_loop(self):
        await asyncio.gather(self.select_from_db(), self.select_from_local_work())

    async def consumer_loop(self):
        while not self.end_event.is_set():
            result = await self.run_or_end(self.queue.get())
            if result is None:
                continue

            item: ProcessRequest = result[0]
            for factory in self.task_factories:
                request = factory.from_process_request(item)
                if request is not None:
                    await factory.invoke(request)

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


def main():
    from seer.bootup import bootup

    bootup(__name__, [AsyncioIntegration()])
    app = AsyncApp(
        end_event=asyncio.Event(),
        queue=asyncio.Queue(),
    )
    asyncio.run(app.run())


if __name__ == "__main__":
    main()
