.PHONY: check test style

check_dirs := quanto test bench examples

check:
	black --check ${check_dirs}
	ruff ${check_dirs}

style:
	black ${check_dirs}
	ruff ${check_dirs} --fix

test:
	python -m pytest -sv test
