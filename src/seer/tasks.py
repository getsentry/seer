import asyncio
import dataclasses
import datetime
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol, TypeVar

from pydantic import BaseModel
from sentry_sdk import start_span
from sentry_sdk.integrations.asyncio import AsyncioIntegration
from sqlalchemy.ext.asyncio import async_sessionmaker

from seer.db import ProcessRequest

_Request = TypeVar("_Request", bound=BaseModel)
_A = TypeVar("_A")

AsyncSession = async_sessionmaker(expire_on_commit=False)


class QConsumer(Protocol[_A]):
    def get(self) -> _A:
        ...


class QProducer(Protocol[_A]):
    def put(self, obj: _A) -> None:
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


@dataclasses.dataclass
class AsyncApp:
    end_event: asyncio.Event
    queue: asyncio.Queue
    io_work: QConsumer[ProcessRequest] = dataclasses.field(default_factory=multiprocessing.Queue)
    cpu_work: QProducer[ProcessRequest] = dataclasses.field(default_factory=multiprocessing.Queue)
    num_consumers: int = 10
    task_factories: list[TaskFactory[BaseModel]] = dataclasses.field(
        default_factory=lambda: [],
    )
    # workers = 2 => one for consuming io work, one for producing cpu_work
    threaded_executor: ThreadPoolExecutor = dataclasses.field(
        default_factory=lambda: ThreadPoolExecutor(max_workers=2)
    )

    async def select_from_db(self) -> None:
        while not self.end_event.is_set():
            loop = asyncio.get_event_loop()
            loop.run_in_executor()
            async with AsyncSession() as session:
                work: list[ProcessRequest] = await session.run_sync(
                    lambda session: ProcessRequest.acquire_work(
                        100, datetime.datetime.utcnow(), session=session
                    )
                )
                for item in work:
                    await self.queue.put(item)

            await asyncio.sleep(1)

    async def select_from_local_work(self) -> None:
        end_task = asyncio.create_task(self.end_event.wait())
        while not self.end_event.is_set():
            get_task = asyncio.get_running_loop().run_in_executor(
                self.threaded_executor, self.io_work.get
            )
            await asyncio.wait([end_task, get_task], return_when=asyncio.FIRST_COMPLETED)
            if end_task.done():
                get_task.cancel()
                await asyncio.gather(get_task, return_exceptions=True)
                continue
            item: ProcessRequest = get_task.result()
            await self.queue.put(item)

    async def producer_loop(self):
        await asyncio.gather(self.select_from_db(), self.select_from_local_work())

    async def consumer_loop(self):
        end_task = asyncio.create_task(self.end_event.wait())
        while not self.end_event.is_set():
            get_task = asyncio.create_task(self.queue.get())
            await asyncio.wait([get_task, end_task], return_when=asyncio.FIRST_COMPLETED)
            if end_task.done():
                get_task.cancel()
                await asyncio.gather(get_task, return_exceptions=True)
                continue

            item: ProcessRequest = get_task.result()
            for factory in self.task_factories:
                request = factory.from_process_request(item)
                if request is not None:
                    # with hub.start_span(op=OP.FUNCTION, description=get_name(coro)):
                    start_span
                    await factory.invoke(request)

                    # If this was persisted work, clean it up.
                    if item.id:
                        async with AsyncSession() as session:
                            await session.execute(item.mark_completed_stmt())
                            await session.commit()
                    break

    async def run(self):
        producer_task = asyncio.create_task(self.producer_loop())
        consumer_tasks: list[asyncio.Task[...]] = []
        while not self.end_event.is_set():
            if producer_task.done():
                # Self terminate if the producer dies for any reason.
                self.end_event.set()

            task: asyncio.Task[...]
            for task in [*consumer_tasks]:
                if task.done():
                    # Unexpected death of a task?  Reboot it.
                    consumer_tasks.remove(task)

            while len(consumer_tasks) < self.num_consumers:
                consumer_tasks.append(asyncio.create_task(self.consumer_loop()))
            await asyncio.sleep(1)

        all_tasks = [producer_task, *consumer_tasks]
        async with asyncio.timeout(10):
            await asyncio.gather(*all_tasks)


def main(work_ready: multiprocessing.Event = multiprocessing.Event()):
    from seer.bootup import bootup

    bootup(__name__, [AsyncioIntegration()])
    app = AsyncApp(
        end_event=asyncio.Event(),
        queue=asyncio.Queue(),
        work_ready=work_ready,
    )
    asyncio.run(app.run())


if __name__ == "__main__":
    main()
