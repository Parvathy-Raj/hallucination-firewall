.PHONY: install test eval traces lint serve

install:
	uv sync --extra ml --extra eval

test:
	uv run pytest -q

lint:
	uv run ruff check src tests evals

# Rebuild the UAE traces from your local copy of the UAE regulatory RAG corpus
traces:
	uv run python evals/custom_traces/build.py $(UAE_CHUNKS)

# Runs every benchmark and rewrites the README results table
eval:
	uv run python -m evals.run

serve:
	uv run hfw serve
