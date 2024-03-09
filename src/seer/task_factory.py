import abc
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Full, Queue
from typing import Any, Callable

import celery.result

from seer.db import ProcessRequest

logger = logging.getLogger("asyncrunner")


class AsyncTaskFactory(abc.ABC):
    @abc.abstractmethod
    def matches(self, process_request: ProcessRequest) -> bool:
        pass

    @abc.abstractmethod
    async def invoke(self, process_request: ProcessRequest):
        pass

    async def async_celery_job(self, cb: Callable[[], celery.result.AsyncResult]):
        logger.info("Starting async celery job")
        loop = asyncio.get_running_loop()
        q: Queue = Queue(1)

        with ThreadPoolExecutor() as pool:
            ar = await loop.run_in_executor(pool, cb)

            def run():
                def on_message(raw: Any):
                    # Keep trying to put the item into the queue by removing existing item until we get in.
                    logger.info("Received response from celery job")
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
                                        logger.warning("SIGUSR1 on job, generator failed")
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
                logger.info("async celery job completing")
                loop.run_in_executor(pool, lambda: q.put_nowait(None))
        await complete


_async_task_factories: list[Callable[[], AsyncTaskFactory]] = []


def async_task_factory(f: Callable[[], AsyncTaskFactory]) -> Callable[[], AsyncTaskFactory]:
    logger.info(f"@async_task_factory registering task factory: {f}")
    _async_task_factories.append(f)
    return f
