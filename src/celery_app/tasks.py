import seer.automation.autofix.tasks
import seer.automation.codebase.tasks

# Import runs the `bootup_celery` and `bootup` helpers
from celery_app.app import app as celery_app  # noqa
