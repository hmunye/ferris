.DEFAULT_GOAL := inference

.PHONY: inference train lint fmt clean

inference:
	@uv run src/main.py

train:
	@uv run src/train.py

lint:
	@uvx ruff check src/

fmt:
	@uvx black src/

clean:
	@rm -rf .ruff_cache
