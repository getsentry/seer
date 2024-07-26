# All tasks are part of the import dependency tree of starting from seer endpoints.
# See the test_seer:test_detected_celery_jobs test
import seer.app  # noqa: F401
from celery_app.app import celery_app as celery  # noqa: F401
