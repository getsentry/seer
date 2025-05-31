makefile:=$(lastword $(MAKEFILE_LIST))
project_name:=$(shell basename $(shell dirname $(realpath $(makefile))))
tmpdir:=$(shell mktemp -d)

default: help

.PHONY: help
help:
	@echo
	@echo make
	@grep -E '^[a-zA-Z0-9 _ -]+:.*#'  $(makefile) | sort | while read -r l; do printf "  \033[1;32m$$(echo $$l | cut -f 1 -d':')\033[00m:$$(echo $$l | cut -f 2- -d'#')\n"; done

.PHONY: pip
pip: # Runs pip install with the requirements.txt file
	pip install -r requirements.txt

.PHONY: shell
shell: .env # Opens a bash shell in the context of the project
	docker compose run app bash

.PHONY: update
update: .env # Updates the project's docker compose image.
	docker compose build
	docker compose run app flask db history
	docker compose run app flask db upgrade

.PHONY: db-downgrade
db-downgrade: .env # Downgrades the db by one upgrade script each time it is run.
	docker compose run app flask db downgrade

.PHONY: db-reset
db-reset: .env  # Reinitializes the database. You need to run make update to apply all migrations after a reset.
	docker compose down --volumes

.PHONY: dev
dev: .env # Starts the webserver based on the current src on port 9091
	docker compose up --build

.PHONY: dev-cuda
dev-cuda: .env # Starts the webserver based on the current src on port 9091 with cuda enabled
	docker compose -f docker-compose.yml -f docker-compose.cuda.yml up --build

.PHONY: test
test: # Executes all tests in the baked image file.  Requires models/
	docker compose up -d test-db
	docker compose run $(EXTRA_COMPOSE_TEST_OPTIONS) app pytest --cov . --cov-report="xml:.artifacts/coverage.xml" -vv

.PHONY: mypy
mypy: # Runs mypy type checking
	docker compose run app mypy

.PHONY: schemas
schemas: # Generates json files
	#docker run --rm -v ./src/seer/schemas:/app/src/seer/schemas $(project_name):latest python src/seer/generate_schemas.py
	docker compose run app python src/seer/generate_schemas.py
	git clone --depth 1 https://github.com/getsentry/sentry-data-schemas.git $(tmpdir)
	docker run --rm -t \
	  -v $(tmpdir):/sentry-data-schemas:ro \
	  -v $$(pwd)/src/:/src:ro \
	  tufin/oasdiff breaking /sentry-data-schemas/seer/seer_api.json /src/seer/schemas/seer_api.json

.PHONY: migration
migration: .env # Creates a new migration script for DB changes
	docker compose run app flask db migrate -m 'Migration'

.PHONY: check-no-pending-migrations
check-no-pending-migrations: .env # Checks for any pending db migrations
	docker compose run app flask db check

.PHONY: merge-migrations
merge-migrations: .env # Merges all db migrations into a single migration file
	docker compose run app flask db merge heads

.env:
	cp .env.example .env

gocd: ## Build GoCD pipelines
	rm -rf ./gocd/generated-pipelines
	mkdir -p ./gocd/generated-pipelines
	cd ./gocd/templates && jb install && jb update

  # Format
	find . -type f \( -name '*.libsonnet' -o -name '*.jsonnet' \) -print0 | xargs -n 1 -0 jsonnetfmt -i
  # Lint
	find . -type f \( -name '*.libsonnet' -o -name '*.jsonnet' \) -print0 | xargs -n 1 -0 jsonnet-lint -J ./gocd/templates/vendor
	# Build
	cd ./gocd/templates && find . -type f \( -name '*.jsonnet' \) -print0 | xargs -n 1 -0 jsonnet --ext-code output-files=true -J vendor -m ../generated-pipelines

  # Convert JSON to yaml
	cd ./gocd/generated-pipelines && find . -type f \( -name '*.yaml' \) -print0 | xargs -n 1 -0 yq -p json -o yaml -i
.PHONY: gocd

upgrade-package-versions: ## Upgrade package versions in requirements.txt from constraints in requirements-constraints.txt
	pip-compile --upgrade --strip-extras --quiet requirements-constraints.txt -o requirements.txt

HEAD_SHA:=$(shell git rev-parse --short HEAD)
TIME:=$(shell date +%F.%T)
SEER_STAGING_VERSION_SHA:=$(HEAD_SHA).$(TIME)
export SENTRY_ORG:=sentry
export SENTRY_PROJECT:=seer
push-staging:
	# Ensure the google authentication helper is working.  If this fails, https://cloud.google.com/artifact-registry/docs/docker/authentication#gcloud-helper
	gcloud auth configure-docker us-west1-docker.pkg.dev > /dev/null
	# Setup your SBX_PROJECT in .env from the sandbox project name
	docker build . --platform linux/amd64 --build-arg SEER_VERSION_SHA=$(SEER_STAGING_VERSION_SHA) --build-arg SENTRY_ENVIRONMENT=staging -t us-west1-docker.pkg.dev/$(SBX_PROJECT)/staging/seer
	docker build . --platform linux/amd64 -f Compose.Dockerfile --build-arg SBX_PROJECT=$(SBX_PROJECT) -t us-west1-docker.pkg.dev/$(SBX_PROJECT)/staging/seer.compose
	docker push	us-west1-docker.pkg.dev/$(SBX_PROJECT)/staging/seer
	docker push	us-west1-docker.pkg.dev/$(SBX_PROJECT)/staging/seer.compose
	sentry-cli releases new "${SEER_STAGING_VERSION_SHA}"
	sentry-cli releases deploys "${SEER_STAGING_VERSION_SHA}" new -e staging
	sentry-cli releases finalize "${SEER_STAGING_VERSION_SHA}"
	sentry-cli releases set-commits "${SEER_STAGING_VERSION_SHA}" --auto || true


.PHONY: vcr-encrypt-prep
vcr-encrypt-prep:
	pip install -r scripts/requirements.txt
	gcloud auth application-default login

.PHONY: vcr-encrypt
CLEAN:=1
vcr-encrypt: # Encrypts all vcr cassettes
	python3 ./scripts/encrypt.py --mode=encrypt --kek_uri=gcp-kms://projects/ml-ai-420606/locations/global/keyRings/seer_cassette_encryption/cryptoKeys/seer_cassette_encryption $(if $(filter 0,$(CLEAN)),,--clean)

.PHONY: vcr-decrypt
vcr-decrypt: # Decrypts all vcr cassettes. Use make vcr-decrypt CLEAN=1 to include --clean flag
	python3 ./scripts/encrypt.py --mode=decrypt --kek_uri=gcp-kms://projects/ml-ai-420606/locations/global/keyRings/seer_cassette_encryption/cryptoKeys/seer_cassette_encryption $(if $(CLEAN) = 1,--clean,)
