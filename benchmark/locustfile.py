import random

from locust import HttpUser, task


#  pip install locust and see https://docs.locust.io/en/stable/quickstart.html
class SeerBenchmarkUser(HttpUser):
    @task
    def similarity_embedding(self):
        real_looking_stacktrace = generate_stacktrace(10)
        self.client.post(
            "/v0/issues/similarity-embedding-benchmark",
            json={
                "project_id": 1,
                "hash": "abc",
                "stacktrace": real_looking_stacktrace,
                "message": "message",
                "k": 1,
                "threshold": 0.01,
            },
        )

    @task
    def bulk_request(self):
        grouping_record_data = generate_grouping_record_data(10)
        self.client.post(
            "/v0/issues/similar-issues/grouping-record",
            json=grouping_record_data,
        )


def generate_stacktrace(depth):
    filenames = ["main.py", "utils.py", "handler.py", "database.sql"]
    functions = [
        "initializeApp",
        "fetchData",
        "processData",
        "saveToDatabase",
        "logError",
        "sendNotification",
        "validateUserInput",
    ]
    errors = [
        "SyntaxError: invalid syntax",
        "NameError: name 'undefined_var' is not defined",
        "TypeError: unsupported operand type(s) for +: 'int' and 'str'",
    ]

    stacktrace = ""
    for i in range(depth):
        file = random.choice(filenames)
        function = random.choice(functions)
        error = errors[i % len(errors)] if i == depth - 1 else ""
        line = random.randint(1, 100)
        stacktrace += f'  File "{file}", line {line}\n    {function}\n'
        if error:
            stacktrace += f"    {error}\n"
    return stacktrace.strip()


def generate_grouping_record_data(num_records):
    records = []
    stacktraces = []
    for i in range(num_records):
        stacktrace = generate_stacktrace(10)
        stacktraces.append(stacktrace)
        records.append(
            {
                "group_id": i + 1,
                "hash": f"hash_{i}",
                "project_id": 1,
                "message": f"message_{i}",
            }
        )
    return {"data": records, "stacktrace_list": stacktraces}
