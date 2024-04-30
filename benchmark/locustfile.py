import random
import string

from locust import HttpUser, task


#  pip install locust and see https://docs.locust.io/en/stable/quickstart.html
class SeerBenchmarkUser(HttpUser):
    @task
    def similarity_embedding(self):
        random_stacktrace = "".join(random.choices(string.ascii_letters + string.digits, k=500))
        self.client.post(
            "/v0/issues/similarity-embedding-benchmark",
            json={
                "group_id": 2,
                "project_id": 1,
                "stacktrace": random_stacktrace,
                "message": "message",
                "stacktrace_hash": "QYK7aNYNnp5FgSev9Np1soqb1SdtyahD",
                "k": 1,
                "threshold": 0.01,
            },
        )
