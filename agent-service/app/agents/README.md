# FoodMind local AI agents

The Spring Boot local defaults expect Recommendation on port 8001, Cooking on
8003, and Chat on 8004. To start the matching services with one DeepSeek key:

1. Copy `.env.example` to `.env` and set `DEEPSEEK_API_KEY`.
2. Run `docker compose up --build` in this directory.
3. Start the Backend with its checked-in `.env.example` values.

The compose file maps each service's internal port to the ports expected by
Spring Boot and injects matching local-only service tokens. Do not use these
known token values outside local development.

Each service also has its own `.env.example` for running directly with `uv`.
No real API key belongs in Git.

