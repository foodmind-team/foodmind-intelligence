# FoodMind local AI agents

Local environment variables live in this directory at `app/agents/.env`.
Individual agent directories must not keep their own `.env` files.

The canonical topology is Chatbot 8001, ML Inference 8002, Cooking Agent 8003,
and Recommendation Agent 8004. These are the actual listeners both on the host
and inside containers. To start the services with one DeepSeek key:

1. Build the local runtime package from the sibling ML repository:
   `python scripts/build_runtime_package.py`.
2. Copy `.env.example` to `.env` and set `DEEPSEEK_API_KEY`. Chatbot, Cooking,
   and Recommendation use it; Inference never does.
3. Run `docker compose up --build` in this directory.
4. Start the Backend with its checked-in `.env.example` values.

Recommendation always gets ranking and model metadata from Inference. DeepSeek
may only render bounded explanations from approved reason facts. Chatbot uses
delegation-scoped Backend tools for read-only search/reference resolution.
Do not use the checked-in local-only service-token defaults outside development.

Each service also has its own `.env.example` for running directly with `uv`.
No real API key belongs in Git.
