SERVICE_NAME=ofs_env

.PHONY: build_image
build_image:
	docker compose build --no-cache --build-arg UID="`id -u`" --build-arg GID="`id -g`"

.PHONY: up
up:
	docker compose up -d

.PHONY: start
start:
	docker compose start

.PHONY: exec
exec:
	docker compose exec ${SERVICE_NAME} /bin/bash

.PHONY: root_exec
root_exec:
	docker compose exec --user root ${SERVICE_NAME} /bin/bash

.PHONY: all
all:
	$(MAKE) build_image
	$(MAKE) up
	$(MAKE) exec