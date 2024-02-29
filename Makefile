makefile:=$(lastword $(MAKEFILE_LIST))
project_name:=$(shell basename $(shell dirname $(realpath $(makefile))))
tmpdir:=$(shell mktemp -d)

default: help

.PHONY: help
help:
	@echo
	@echo make
	@grep -E '^[a-zA-Z0-9 -]+:.*#'  $(makefile) | sort | while read -r l; do printf "  \033[1;32m$$(echo $$l | cut -f 1 -d':')\033[00m:$$(echo $$l | cut -f 2- -d'#')\n"; done

.PHONY: pip
pip: # Runs pip install with the requirements.txt file
	pip install -r requirements.txt

.PHONY: shell
shell: .env # Opens a bash shell in the context of the project
	docker-compose run app bash

.PHONY: update
update: .env # Updates the project's docker-compose image.
	docker-compose build
	docker-compose run app flask db upgrade

.PHONY: dev
dev: .env # Starts the webserver based on the current src on port 9091
	docker-compose up --build

.PHONY: test
test: # Executes all tests in the baked image file.  Requires models/
	docker-compose run app mypy
	docker-compose run app pytest
#	docker run --rm pgvector/pgvector:pg14
#	docker run --rm -v ./tests:/app/tests -v ./src:/app/src $(project_name):latest pytest

.PHONY: mypy
mypy: # Runs mypy type checking
	docker run --rm -v ./tests:/app/tests -v ./src:/app/src $(project_name):latest mypy

.PHONY: schemas
schemas: # Generates json files
	#docker run --rm -v ./src/seer/schemas:/app/src/seer/schemas $(project_name):latest python src/seer/generate_schemas.py
	docker-compose run app python src/seer/generate_schemas.py
	git clone --depth 1 https://github.com/getsentry/sentry-data-schemas.git $(tmpdir)
	docker run --rm -t \
	  -v $(tmpdir):/sentry-data-schemas:ro \
	  -v $$(pwd)/src/:/src:ro \
	  tufin/oasdiff breaking /sentry-data-schemas/seer/seer_api.json /src/seer/schemas/seer_api.json

.PHONY: migration
migration: .env
	docker-compose run app flask db migrate -m 'Migration'

.PHONY: merge-migrations
merge-migrations: .env
	docker-compose run app flask db merge heads

.env:
	cp .env.example .env
