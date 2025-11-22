.DEFAULT_GOAL := run

.PHONY: run train ift lint fmt clean

run:
	@uv run src/main.py

train:
	@uv run src/train.py $(ARGS)

ift:
	@uv run src/ift.py $(ARGS)

lint:
	@uvx ruff check src/

fmt:
	@uvx black src/

clean:
	@rm -rf .ruff_cache
