.PHONY: test
test:
	python -m pytest tests -s -v

.PHONY: style
style:
	isort . && ruff format ./