from celery import Celery

app = Celery("seer", broker="memory://")
app.conf.task_serializer = "json"
app.conf.result_serializer = "json"
app.conf.accept_content = ["json"]
app.conf.enable_utc = True
