.DEFAULT_GOAL := run

.PHONY: run train ift lint fmt clean help

run:
	@uv run src/main.py $(ARGS)

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

help:
	@echo "Available targets:"
	@echo "  run      - Run the model inference script"
	@echo "  train    - Run the model pretraining script"
	@echo "  ift      - Run the model instruction fine-tuning script"
	@echo "  lint     - Run the linter against scripts within src/"
	@echo "  fmt      - Run the formatter against scripts within src/"
	@echo "  clean    - Remove all generated files and directories"
	@echo "  help     - Show this help message"
