SRC = src

.DEFAULT_GOAL := run

.PHONY: run train lint fmt clean

run:
	@uv run $(SRC)/main.py

train:
	@uv run $(SRC)/train.py

lint:
	@uvx ruff check $(SRC)

fmt:
	@uvx black $(SRC)

clean:
	@rm -rf .ruff_cache
	@rm -rf $(SRC)/**/__pycache__
