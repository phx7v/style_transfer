COMPOSE = docker compose -f docker/docker-compose.yml
COMPOSE_RUN_DEV = $(COMPOSE) run --rm app-dev


# App up and down
dev_up:
	$(COMPOSE) --profile dev up

prod_up:
	$(COMPOSE) --profile prod up -d

dev_down:
	$(COMPOSE) --profile dev down

prod_down:
	$(COMPOSE) --profile prod down


# Tests, lineters, formatters
test:
	$(COMPOSE_RUN_DEV) pytest

lint:
	$(COMPOSE_RUN_DEV) ruff check .

lint_fix:
	$(COMPOSE_RUN_DEV) ruff check . --fix

format:
	$(COMPOSE_RUN_DEV) ruff format --check

format_diff:
	$$(COMPOSE_RUN_DEV) ruff format --check --diff

format_done:
	$(COMPOSE_RUN_DEV) ruff format

final_check:
	$(COMPOSE_RUN_DEV) bash -lc 'ruff check . & ruff format --check & wait; pytest'

final_check_fixed:
	$(COMPOSE_RUN_DEV) bash -lc 'ruff check . --fix & ruff format --check & wait; pytest'


# Poetry
PKG ?=
VER ?=

poetry_lock:
	$(COMPOSE_RUN_DEV) poetry lock

poetry_update:
	$(COMPOSE_RUN_DEV) poetry update $(PKG)

poetry_add_prod:
	@if [ -z '$(PKG)' ]; then \
		echo 'Error: PKG is not set. Usage: make poetry_add_prod PKG=<name> [VER=<version>]'; \
		exit 1; \
	fi
	$(COMPOSE_RUN_DEV) poetry add $(PKG)$(VER)

poetry_add_dev:
	@if [ -z '$(PKG)' ]; then \
		echo 'Error: PKG is not set. Usage: make poetry_add_dev PKG=<name> [VER=<version>]'; \
		exit 1; \
	fi
	$(COMPOSE_RUN_DEV) poetry add --dev $(PKG)$(VER)

poetry_remove_prod:
	@if [ -z '$(PKG)' ]; then \
		echo 'Error: PKG is not set. Usage: make poetry_remove_prod PKG=<name>'; \
		exit 1; \
	fi
	$(COMPOSE_RUN_DEV) poetry remove $(PKG)

poetry_remove_dev:
	@if [ -z '$(PKG)' ]; then \
		echo 'Error: PKG is not set. Usage: make poetry_remove_dev PKG=<name>'; \
		exit 1; \
	fi
	$(COMPOSE_RUN_DEV) poetry remove --dev $(PKG)

export_to_onnx:
	@if [ -z '$(WEIGHTS)' ] || [ -z '$(ONNX)' ]; then \
		echo 'Error: WEIGHTS and ONNX must be set.'; \
		echo 'Usage: make export_to_onnx WEIGHTS=<weights_path> ONNX=<onnx_path>'; \
		exit 1; \
	fi
	$(COMPOSE_RUN_DEV) 	python -m cli.export_to_onnx --weights $(WEIGHTS) --onnx $(ONNX)
