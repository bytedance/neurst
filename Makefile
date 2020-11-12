SHELL = bash

all: install_dev isort isort_check lint test
check: isort_check lint test

install_dev:
	@pip install -e .[dev] >/dev/null 2>&1

isort:
	@isort -s venv -s venv_py -s .tox -rc --atomic .

isort_check:
	@isort -rc -s venv -s venv_py -s .tox -c .

lint:
	@flake8

test:
	@tox

clean:
	@rm -rf .pytest_cache .tox bytedmypackage.egg-info
	@rm -rf tests/*.pyc tests/__pycache__

.IGNORE: install_dev
.PHONY: all check install_dev isort isort_check lint test
