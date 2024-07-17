# seer

## Using Seer

### Autofix

#### Codebase Index Storage

Autofix creates and uses a ChromaDB for every separate codebase. This can either be stored in a Google Cloud Storage (GCS) bucket, in your local filesystem, or you can make your own storage adapter.

##### Google Cloud Storage

To use GCS, you need to set the following environment variables:

```txt
GOOGLE_APPLICATION_CREDENTIALS=<path to your GCS credentials file>
GOOGLE_CLOUD_PROJECT=<your GCS project ID>
```

Then, you can define your storage bucket with:

```txt
CODEBASE_STORAGE_TYPE=gcs
CODEBASE_GCS_STORAGE_BUCKET=<your GCS bucket name>
CODEBASE_GCS_STORAGE_DIR=<path within gcs bucket>
```

##### Local Filesystem

To use local filesystem, you can set the following environment variable:

```txt
CODEBASE_STORAGE_TYPE=filesystem
CODEBASE_STORAGE_DIR=<path to your local directory>
```

## Local Development

### Setup

Use `direnv` or a similar tool that sources `.envrc`. It will check your python version and setup virtualenv for you:

```bash
direnv allow
```

Recommended to use `pyenv` or similar python environment manager so as to be able to use differing python versions between sentry projects.

> You will need pyenv installed and configured for python 3.11 before running `direnv allow`.

### Environment variables

Create `.env` in the root of the project and set the environment variables that are in `.env.example` to the actual values.

> The example shows `GITHUB_PRIVATE_KEY` and `GITHUB_APP_ID`. You can also use just `GITHUB_TOKEN` instead, which you can get from GitHub Settings/Developer Setttings/Personal Access Tokens.
> For local development, set `NO_SENTRY_INTEGRATION=1` and set `NO_REAL_MODELS=1`.

In your rc file, add `export SENTRY_AUTH_TOKEN=<your sentry auth token>`.

### Install GCloud CLI

Refer <https://cloud.google.com/sdk/docs/install>. We use the `super-big-data` project.

### Model Artifacts

You will need model artifacts to run inference in seer, get them from gcs by:

```bash
gsutil cp -r gs://sentry-ml/seer/models ./models
```

### Running

To run for development locally in one go, including building the docker image and rabbitmq container, run:

```bash
make dev # runs docker compose up --build
```

If you see database-related errors, try `make update` (see the Applying Migrations section below).

#### Running Autofix with Sentry

For the full Autofix experience, you'll want to use the UI in `sentry`.

Port `9091` will be exposed which is what the local `sentry` application will look for to connect to the service.

In `~/.sentry/sentry.conf.py`, make sure you have the below to access Autofix and Seer in your local `sentry` instance:

```yaml
SEER_RPC_SHARED_SECRET=["seers-also-very-long-value-haha"]
SENTRY_FEATURES['projects:ai-autofix'] = True
SENTRY_FEATURES['organizations:issue-details-autofix-ui'] = True
```

You will not be able to get GenAI data consent locally, so you may need to hardcode [this line](https://github.com/getsentry/sentry/blob/c4848fa48c92a9dd40649a4f94072c4154d6d564/static/app/components/events/autofix/useAutofixSetup.tsx#L50-L54) in `sentry` to `Boolean(true)`.

You may also have to comment out [this GitHub check](https://github.com/getsentry/sentry/blob/3f6b07dbd53386c8b8bb44a84fbffcdd5d59f16f/src/sentry/api/endpoints/group_ai_autofix.py#L199-L203) in order to use Autofix from the UI locally, if no GitHub repositories are linked to your project.

Make sure to restart `sentry` and `seer`.

Now in the Sentry interface, you should be able to start Autofix from an event.

### Database Stuff

#### Applying Migrations

```bash
make update
```

This will apply all migrations to the database and create new image.

#### Creating Migrations

```bash
make migration
```

### Running Mypy type checker

```bash
make mypy
```

### Running Tests

```bash
make test
```

### Opening a shell

```bash
make shell
```

### Resetting Dev Environment

If you run into any data issue or connection issue and want to start from scratch, run the following set of commands from the command shell inside the seer repo:

```
docker compose down --volumes
make update && make dev
```

### Using Langfuse

We publish our Langfuse traces to our Langfuse instance. Set the following environment variables:

```bash
LANGFUSE_SECRET_KEY=...
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_HOST=...
```

to use it. Otherwise leaving them unset will disable the instrumentation.
