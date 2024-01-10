from celery import Celery

app = Celery("seer", broker="redis://redis:6379/0", backend="redis://redis:6379/0")
app.conf.task_serializer = "json"
app.conf.result_serializer = "json"
app.conf.accept_content = ["json"]
app.conf.enable_utc = True
app.autodiscover_tasks(["seer.tasks"])
