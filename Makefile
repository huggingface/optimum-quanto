.PHONY: check test style

check_dirs := optimum test bench examples

check:
	ruff check ${check_dirs}

style:
	ruff check ${check_dirs} --fix

test:
	python -m pytest -sv test
