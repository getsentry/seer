from celery_app.app import app as celery_app

# pool.apply_async()


@celery_app.task(time_limit=60 * 2)
def schedule_process_requests() -> None:
    pass
