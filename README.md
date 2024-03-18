# seer

## Local Development

### Setup

Use `direnv` or a similar tool that sources `.envrc`.  It will check your python version and setup virtualenv for you:

```bash
direnv allow
```

Recommended to use `pyenv` or similar python environment manager so as to be able to use differing python versions between sentry projects.

### Environment variables

Set the environment variables that are in `.env.example` with the actual values, save this as `.env` in the root of the project.

### Model Artifacts

You will need model artifacts to run inference in seer, get them from gcs by:

```bash
gsutil cp -r gs://tmp_tillman/models ./models
```

### Running

To run for development locally in one ago including building the docker image and rabbitmq container:

```bash
make dev # runs docker-compose up --build
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
