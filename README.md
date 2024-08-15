
# Seer

<img src="seer.png" alt="Seer Logo" width="128">

Seer is a service that provides AI capabilities to Sentry, running inference on Sentry issues and providing insights to users.

> 📣 Seer is currently in early development and not yet compatible with self-hosted Sentry instances. Stay tuned for updates!

## Setup

These instructions require access to internal Sentry resources, and are intended for internal Sentry employees.

### Prerequisites

1. Install [direnv](https://direnv.net/) or a similar tool
2. Install [pyenv](https://github.com/pyenv/pyenv) and configure Python 3.11
3. Install [Docker](https://www.docker.com/get-started)
4. Install [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)

### Environment Setup

1. Clone the repository and navigate to the project root
2. Run `direnv allow` to set up the Python environment
3. Create a `.env` file based on `.env.example` and set the required values
4. (Optional) Add `export SENTRY_AUTH_TOKEN=<your token>` to your shell's RC file

### Model Artifacts

Download model artifacts:

```bash
gsutil cp -r gs://sentry-ml/seer/models ./models
```


### Running Seer

1. Start the development environment:

   ```bash
   make dev
   ```

2. If you encounter database errors, run:

   ```bash
   make update
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

- Apply database migrations: `make update`
- Create new migrations: `make migration`
- Run type checker: `make mypy`
- Run tests: `make test`
- Open a shell: `make shell`

### Reset Development Environment

To start fresh:

```bash
bash
docker compose down --volumes
make update && make dev
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
"run_name": "string", // Custom name for your evaluation run
"run_description": "string", // Description of your evaluation run
"run_type": "full | root_cause | execution", // Type of evaluation to perform
"test": boolean, // Set to true to run on a single item (for testing)
"random_for_test": boolean, // Set to true to use a random item when testing (requires "test": true)
"run_on_item_id": "string", // Specific item ID to run on (optional)
"n_runs_per_item": int // Number of runs to perform per item (optional, default 1)
}
```

Note: Currently, only internal datasets are available.
