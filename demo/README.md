# Demo

This folder contains a minimal runnable demo for the FalkorDB LangChain
integration.

## Run

1. Install dependencies:

   ```bash
   cd demo
   uv sync
   ```

2. Start FalkorDB (for example with Docker):

   ```bash
   docker run -p 6379:6379 -it falkordb/falkordb:latest
   ```

3. Run the demo:

   ```bash
   uv run python basic_usage.py
   ```

Optional flags:

```bash
uv run python basic_usage.py --host localhost --port 6379 --database langchain_falkordb_demo
```
