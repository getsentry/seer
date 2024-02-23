import asyncio
import dataclasses
import datetime
from typing import Annotated, Callable, Self

import pytest
from pydantic import BaseModel
from sqlalchemy import select

from seer.db import ProcessRequest, Session
from seer.generator import Examples, gen, ints, parameterize, printable_strings
from seer.tasks import AsyncApp, AsyncSession, TaskFactory


@dataclasses.dataclass
class ScheduledWork:
    process_request: ProcessRequest

    def save(self) -> Self:
        with Session() as session:
            session.add(self.process_request)
            session.commit()
        return self


@dataclasses.dataclass
class UpdatedWork:
    scheduled_work: ScheduledWork
    new_payload: Annotated[dict, Examples(({"this-unique-payload": i} for i in ints))]

    @property
    def original_process_request(self):
        return self.scheduled_work.process_request

    @property
    def current_process_request_by_name(self):
        with Session() as session:
            return session.scalar(
                select(ProcessRequest).where(
                    ProcessRequest.name == self.original_process_request.name
                )
            )

    def save(self, now: datetime.datetime) -> Self:
        with Session() as session:
            self.scheduled_work.save()
            session.execute(
                ProcessRequest.schedule_stmt(
                    self.original_process_request.name, self.new_payload, now=now
                )
            )
            session.commit()
            return self


@parameterize
def test_schedule_is_idempotent(updated: UpdatedWork):
    updated.save(datetime.datetime.now())
    assert updated.current_process_request_by_name.payload == updated.new_payload
    assert (
        updated.current_process_request_by_name.scheduled_from
        > updated.original_process_request.scheduled_from
    )
    assert (
        updated.current_process_request_by_name.scheduled_for
        == updated.current_process_request_by_name.scheduled_for
    )


@parameterize
def test_mark_complete_does_not_erase_concurrent_work(updated: UpdatedWork):
    updated.save(datetime.datetime.now() + datetime.timedelta(seconds=1))
    with Session() as session:
        session.execute(updated.original_process_request.mark_completed_stmt())

    assert updated.current_process_request_by_name is not None

    with Session() as session:
        session.execute(updated.current_process_request_by_name.mark_completed_stmt())
        session.commit()

    assert updated.current_process_request_by_name is None


@parameterize
def test_next_schedule(
    scheduled: tuple[ScheduledWork, ScheduledWork, ScheduledWork, ScheduledWork]
):
    for s in scheduled:
        s.save()

    work_1 = ProcessRequest.acquire_work(2, datetime.datetime.now())
    assert len(work_1) == 2

    work_2 = ProcessRequest.acquire_work(2, datetime.datetime.now())
    assert len(work_2) == 2

    assert len(ProcessRequest.acquire_work(4, datetime.datetime.now())) == 0

    assert (
        len(ProcessRequest.acquire_work(4, datetime.datetime.now() + datetime.timedelta(minutes=1)))
        == 4
    )
    assert (
        len(ProcessRequest.acquire_work(4, datetime.datetime.now() + datetime.timedelta(minutes=1)))
        == 0
    )

    assert (
        len(ProcessRequest.acquire_work(4, datetime.datetime.now() + datetime.timedelta(minutes=2)))
        == 0
    )
    assert (
        len(ProcessRequest.acquire_work(4, datetime.datetime.now() + datetime.timedelta(minutes=3)))
        == 4
    )


@parameterize
def test_next_schedule(scheduled: ScheduledWork):
    scheduled.save()
    proc = scheduled.process_request
    now = proc.scheduled_from

    proc.scheduled_for = next_time = proc.next_schedule(now)
    assert next_time - now == datetime.timedelta(minutes=2)

    proc.scheduled_for = next_time = proc.next_schedule(now)
    assert next_time - now == datetime.timedelta(minutes=4)

    proc.scheduled_for = next_time = proc.next_schedule(now)
    assert next_time - now == datetime.timedelta(minutes=8)

    proc.scheduled_for = next_time = proc.next_schedule(now)
    assert next_time - now == datetime.timedelta(minutes=16)


class TestRequest(BaseModel):
    my_special_value: str


class TestTaskFactory(TaskFactory[TestRequest]):
    acceptible_names: frozenset[str] = frozenset(["job-1", "job-2", "job-3", "job-4"])
    side_effect: Callable[[str], None] = lambda _: None

    def from_process_request(self, process_request: ProcessRequest) -> TestRequest | None:
        if process_request.name in self.acceptible_names:
            return TestRequest(**process_request.payload)
        return None

    async def invoke(self, request: TestRequest):
        self.side_effect(request.my_special_value)


@dataclasses.dataclass
class ScheduleAsyncTest:
    acceptable_name: Annotated[str, Examples(gen.one_of(TestTaskFactory.acceptible_names))]
    unacceptable_name: Annotated[
        str, Examples((s for s in printable_strings if s not in TestTaskFactory.acceptible_names))
    ]
    payload: TestRequest
    side_effect_calls: list[str] = dataclasses.field(default_factory=list)
    end_event: asyncio.Event = dataclasses.field(default_factory=asyncio.Event)

    def create_app(self) -> AsyncApp:
        test_task = TestTaskFactory()
        test_task.side_effect = lambda v: self.side_effect_calls.append(v)

        return AsyncApp(
            end_event=self.end_event,
            queue=asyncio.Queue(),
            task_factories=[
                test_task,
            ],
            num_consumers=2,
        )

    async def current_process_request_by_name(self, name: str):
        async with AsyncSession() as session:
            return await session.scalar(select(ProcessRequest).where(ProcessRequest.name == name))

    async def new_side_effect(self):
        self.side_effect_calls = []
        for _ in range(20):
            await asyncio.sleep(0.1)
            if self.side_effect_calls:
                return self.side_effect_calls
        raise ValueError(f"side effect of invoke was not invoked for 2 seconds")


@pytest.mark.asyncio
@parameterize
async def test_next_schedule_async(test: ScheduleAsyncTest, now: datetime.datetime):
    async with AsyncSession() as session:
        for name in [test.acceptable_name, test.unacceptable_name]:
            await session.execute(
                ProcessRequest.schedule_stmt(name, test.payload.model_dump(), now=now)
            )
        await session.commit()

    process = await test.current_process_request_by_name(test.acceptable_name)
    assert process is not None

    run_task = asyncio.create_task(test.create_app().run())
    processed = await test.new_side_effect()
    test.end_event.set()
    await run_task
    assert processed == [test.payload.my_special_value]

    process = await test.current_process_request_by_name(test.acceptable_name)
    assert process is None

    process = await test.current_process_request_by_name(test.unacceptable_name)
    assert process is not None
