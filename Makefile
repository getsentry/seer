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
shell: image # Opens a bash shell in the context of the project
	docker run --rm -v $(PWD)/models:/app/models -v $(PWD)/src:/app/src -it $(project_name):latest bash

.PHONY: run
run: image # Starts the webserver based on the current src on port 8900
	docker run --rm --env PORT=8900 $(project_name):latest

.PHONY: test
test: image # Executes all tests in the baked image file.  Requires models/
	docker run --rm $(project_name):latest pytest
