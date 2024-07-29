.PHONY: black-check
black-check:
	black --check src tests

.PHONY: black
black:
	black src tests

.PHONY: flake8
flake8:
	flake8 src tests

.PHONY: isort-check
isort-check:
	isort --check-only src tests

.PHONY: isort
isort:
	isort src tests

.PHONY: mdformat
mdformat:
	mdformat *.md

.PHONY: mdformat-check
mdformat-check:
	mdformat --check *.md

.PHONY: mypy
mypy:
	mypy src

.PHONY: test
test:
	pytest tests --cov=src --cov-report term-missing --durations 5

.PHONY: format
format:
	$(MAKE) black
	$(MAKE) isort
	$(MAKE) mdformat

.PHONY: lint
lint:
	$(MAKE) black-check
	$(MAKE) isort-check
	$(MAKE) mdformat-check
	$(MAKE) flake8
	$(MAKE) mypy

.PHONY: test-all
test-all:
	$(MAKE) lint
	$(MAKE) test