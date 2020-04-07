.PHONY=help build run

help: ## This help.
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

build:  ## Build docker image
	docker-compose build

run:  ## Run docker image
	docker-compose up -d

pull: ## Pull image image from DockerHub and Run
	docker-compose pull covasim

pull-run: pull run  ## Pull Image from DockerHub and Run