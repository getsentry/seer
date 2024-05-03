# seer

## Using Seer

### Autofix

#### Codebase Index Storage

Autofix creates and uses a ChromaDB for every separate codebase. This can either be stored in a Google Cloud Storage (GCS) bucket, in your local filesystem, or you can make your own storage adapter.

##### Local Workspace

Autofix will need a local folder to use as its workspace, by default it uses `data/chroma/workspaces`.

You can set the location in filesystem where Autofix will use as a workspace with `CODEBASE_WORKSPACE_DIR`.

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

Set the environment variables that are in `.env.example` with the actual values, save this as `.env` in the root of the project.

> The example shows `GITHUB_PRIVATE_KEY` and `GITHUB_APP_ID`. You can also use just `GITHUB_TOKEN` instead.

Add `export SENTRY_AUTH_TOKEN=<your sentry auth token>` to your rc file.

### Install GCloud CLI

Refer <https://cloud.google.com/sdk/docs/install>. We use the `super-big-data` project.

### Model Artifacts

You will need model artifacts to run inference in seer, get them from gcs by:

```bash
gsutil cp -r gs://sentry-ml/seer/models ./models
```

### Running

To run for development locally in one ago including building the docker image and rabbitmq container:

```bash
make dev # runs docker compose up --build
```

Port `9091` will be exposed which is what the local sentry application will look for to connect to the service.

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
