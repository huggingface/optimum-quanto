.PHONY: check test style

check_dirs := optimum test bench examples

check:
	ruff check --show-fixes ${check_dirs}
	ruff format ${check_dirs} --diff

style:
	ruff check ${check_dirs} --fix
	ruff format ${check_dirs}

test:
	python -m pytest -sv test
