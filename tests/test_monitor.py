import datetime
import threading
import time

from celery import Celery
from celery.apps.worker import Worker

from celery_app.monitor import CeleryMonitor


def test_monitor_integration(celery_app: Celery, celery_worker: Worker):
    @celery_app.task(time_limit=60)
    def my_happy_task():
        return

    @celery_app.task(time_limit=60)
    def my_sad_task():
        raise Exception("oh no!")

    @celery_app.task(time_limit=1)
    def my_slow_task():
        time.sleep(100)

    @celery_app.task(time_limit=1)
    def my_dead_task():
        import sys

        sys.exit(6)

    celery_worker.reload()

    monitor = CeleryMonitor(app=celery_app, publish_interval=datetime.timedelta(seconds=3))
    with monitor.run() as receiver:
        t = threading.Thread(target=lambda: receiver.capture())
        t.start()

        my_happy_task.apply_async()
        my_sad_task.apply_async()
        my_slow_task.apply_async()
        t = my_dead_task.apply_async()

        t.join()
