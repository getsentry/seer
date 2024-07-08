from celery import Celery

# singleton celery app used to register tasks.
# configuration occurs via the celery_app.config.injector object
# celery workers are started via importing the tasks module
app = Celery("seer")
