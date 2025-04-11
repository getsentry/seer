
# Seer

<img src="seer.png" alt="Seer Logo" width="128">

Seer is a service that provides AI capabilities to Sentry by running inference on Sentry issues and providing user insights.

> 📣 Seer is currently in early development and not yet compatible with self-hosted Sentry instances. Stay tuned for updates!

## Setup

These instructions require access to internal Sentry resources and are intended for internal Sentry employees.

### Prerequisites

1. Install [direnv](https://direnv.net/) or a similar tool
2. Install [pyenv](https://github.com/pyenv/pyenv) and configure Python 3.11
```bash
pyenv install 3.11
pyenv local 3.11
```
3. Install [Docker](https://www.docker.com/get-started). Note that if you want to install Docker from brew instead of Docker Desktop, then you would need to install docker-compose as well.
4. Install [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) and authenticate.
### Environment Setup

1. Clone the repository and navigate to the project root
2. Run `direnv allow` to set up the Python environment
3. Create a `.env` file based on `.env.example` and set the required values
4. (Optional) Add `SENTRY_AUTH_TOKEN=<your token>` to your `.env` file

### Model Artifacts

Download model artifacts:

```bash
gsutil cp -r gs://sentry-ml/seer/models .
```
If you see a prompt "Reauthentication required. Please insert your security key and press enter...", re-authenticate using the command `gcloud auth login` and set the project id to the one for Seer.
### Running Seer

1. Start the development environment:

   ```bash
   make dev
   ```

2. If you encounter database errors, run:

   ```bash
   make update
   ```
3. If you encounter authentication errors, run:

   ```bash
   gcloud auth application-default login
   ```


## Integrating with Local Sentry

1. Expose port 9091 in your local Sentry configuration
2. Add the following to `~/.sentry/sentry.conf.py`:

   ```python
   SEER_RPC_SHARED_SECRET = ["seers-also-very-long-value-haha"]
   SENTRY_FEATURES['projects:ai-autofix'] = True
   SENTRY_FEATURES['organizations:issue-details-autofix-ui'] = True
   ```

3. For local development, you may need to bypass certain checks in the Sentry codebase
4. Restart both Sentry and Seer

> [!NOTE]
> Set `NO_SENTRY_INTEGRATION=1` in `.env` to ignore Local Sentry Integration

## Development Commands

* Apply database migrations: `make update`
* Create new migrations: `make migration`
* Run type checker: `make mypy`
* Run tests: `make test`
* Open a shell: `make shell`
* Update `requirements.txt` based on `requirements-constraints.txt`: `make upgrade-package-versions`

### Reset Development Environment

To start fresh:

```bash
bash
docker compose down --volumes
make update && make dev
```

## Running Multiple Instances of Seer

To run multiple instances of Seer, you should set unique port values for each instance in the `.env` file.

```
RABBITMQ_PORT=...
RABBITMQ_CONFIG_PORT=...
DB_PORT=...
APP_PORT=...
```

## Langfuse Integration

To enable Langfuse tracing, set these environment variables:

```bash
LANGFUSE_SECRET_KEY=...
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_HOST=...
```

## Autofix

Autofix is an AI agent that identifies root causes of Sentry issues and suggests fixes.

### Running Evaluations

Send a POST request to `/v1/automation/autofix/evaluations/start` with the following JSON body:

```jsonc
{
"dataset_name": "string", // Name of the dataset to run on (currently only internal datasets available)
"run_name": "string", // Custom name for your evaluation run...MAKE SURE THIS NAME IS UNIQUE EVEN IF YOU DELETED A PREVIOUS RUN! LANGFUSE KEYS ON THIS!!
"run_description": "string", // Description of your evaluation run
"run_type": "full | root_cause | execution", // Type of evaluation to perform
"test": boolean, // Set to true to run on a single item (for testing)
"random_for_test": boolean, // Set to true to use a random item when testing (requires "test": true)
"run_on_item_id": "string", // Specific item ID to run on (optional)
"n_runs_per_item": int // Number of runs to perform per item (optional, default 1)
}
```

Note: Currently, only internal datasets are available.

## Staging Sandbox

It is possible to run and deploy seer to a sandbox staging environment.
An example of such a deployment is in [this PR](https://github.com/getsentry/terraform-sandboxes.private/pull/128/files).

To get started, use the `#proj-tf-sandbox` channel and request direction or help on scaffolding a new sandbox in the
[sandbox repo](https://github.com/getsentry/terraform-sandboxes.private).

You then can use the [iap](https://github.com/getsentry/terraform-sandboxes.private/pull/128/files#diff-6c91d750c658a5427482946cfcdce9e1eb6347ebed778bffc4b1813311dc9a79),
and [seer-staging](https://github.com/getsentry/terraform-sandboxes.private/pull/128/files#diff-716adf4cf1da8b035369fe6219f11fb09b86e1757b91f0c03187d58e5f0bc5bd)
modules to scaffold a public load balancer pointing to a compute running the `docker-compose.staging.yml` file.


After scaffolding your environment, you'll want to set your `SBX_PROJECT` environment variable in your `.env` file, and
run `make push-staging` to submit a cloud build for your image.

!!!!NOTE!!!!
The staging cloud build uses your current local environment to build the image, not CI, which means it will use all your
src files and your local `.env` file to configure the image that will be hosted in your sandbox.  Make sure you don't
accidentally include any sensitive personal files in your source tree before using this.


Each time you push with `make push-staging` there will be a period of time while the VM polls and unpacks the new image
before it is loaded.  If you have a `SENTRY_DSN` and `SENTRY_ENVIRONMENT` set, a release will be created by the push,
allowing you to track when the server has loaded that release version.


## Running Tests

You can run all tests with `make test`.

### Running Individual Tests

Make sure you have the test database running when running individual tests, do that via `docker compose up --remove-orphans -d test-db`.

To run a single test, make sure you're in a shell, by doing `make shell`, and then run `pytest tests/path/to/test.py::test_name`.


### VCRs

VCRs are a way to record and replay HTTP requests.  They are useful for recording requests from external services that you don't control instead of mocking them.

To use VCRs, add the `@pytest.mark.vcr()` decorator to your test.

To record new VCRs, delete the existing cassettes and run the test.  Subsequent test runs will use the cassette instead of making requests.

#### VCR Encryption

You must not commit the raw VCRs to the repo.  Instead, you must encrypt them using `make vcr-encrypt` and decrypt them using `make vcr-decrypt`.

Before first time running encryption or decryption, you will need to run `make vcr-encrypt-prep` to install the required libraries and authenticate with GCP.

##### Using encrypted VCRs

Before committing the VCRs, you must run `make vcr-encrypt` to encrypt them.
> By default, the `CLEAN=1` flag is set, so the encrypted cassettes will match exactly what you have in your local. If you want to not overwrite files, run `make vcr-encrypt CLEAN=0`.

If you want to run tests with VCRs enabled, you must run `make vcr-decrypt` to decrypt them.
> By default, the `CLEAN` flag is not set, so files in your local that don't exist in the repo will not be deleted. If you want your local to match exactly what is in the encrypted cassettes, run with `CLEAN=1`.

### Split Tests in CI

In CI, we split the tests into groups to run in parallel. The groups are divided by the test durations.

The durations are stored in the `.test_durations` file.

To update the durations, run in a `make shell`:
```bash
pip install pytest-split
pytest --store-durations
```

The durations should be updated every once in a while or especially when you add potentially slow tests.

# Production

## Celery Worker Queue

You can set the queue that the celery worker listens on via the `CELERY_WORKER_QUEUE` environment variable.

If not set, the default queue name is `"seer"`.

## Debugging Celery with Flower

When `DEV=1`, the flower dashboard is available at `http://localhost:5555`. Use it to debug Celery.
