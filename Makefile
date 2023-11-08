makefile:=$(lastword $(MAKEFILE_LIST))
project_name:=$(shell basename $(shell dirname $(realpath $(makefile))))

default: help

.PHONY: help
help:
	@echo
	@echo make
	@grep -E '^[a-zA-Z0-9 -]+:.*#'  $(makefile) | sort | while read -r l; do printf "  \033[1;32m$$(echo $$l | cut -f 1 -d':')\033[00m:$$(echo $$l | cut -f 2- -d'#')\n"; done

.PHONY: pip
pip: # Runs pip install with the requirements.txt file
	pip install -r requirements.txt

.PHONY: image
image: # Builds the dockerfile image of the project
	docker build . --tag $(project_name):latest

.PHONY: shell
shell: # Opens a bash shell in the context of the project
	docker-compose run app bash

.PHONY: update
update: # Updates the project's docker-compose image.
	docker-compose build

.PHONY: run
run: # Starts the webserver based on the current src on port 8900
	docker run --rm --env PORT=8900 $(project_name):latest

.PHONY: test
test: # Executes all tests in the baked image file.  Requires models/
	docker run --rm -v ./tests:/app/tests $(project_name):latest pytest
