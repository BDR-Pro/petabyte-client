<!-- Source of truth: the petabyte CLI is developed in the (private) petabyte monorepo and
     mirrored here by scripts/build_cli_package.py. Open issues/PRs against this repo for the
     client; the server is closed-source. -->

# Petabyte CLI & Dashboard

## CLI
Installed from PyPI, the `petabyte` command is a thin client (only needs `httpx` — it just talks
to the API over HTTPS, so it never pulls in the server):
```bash
pip install petabyte-client                          # the command it installs is `petabyte`
export PETABYTE_API_URL=https://petabyte.market     # default; or pass --api / omit for localhost
export PETABYTE_API_KEY=pk_...                        # your account key — sign in on the web (Google),
                                                     # create an 'account'-scoped key; no passwords
petabyte deposit 100
petabyte specs                                       # a readable, cheapest-first GPU table
petabyte launch ollama --hours 2                     # one-click app: cheapest verified GPU, started
petabyte run hello.ipynb --gpu H100 --hours 1        # run a notebook/.py on a rented GPU
petabyte ask "explain attention" --model llama3.2    # pay-per-token Inference API (OpenAI-compatible)
petabyte wallet
```
`petabyte ask` uses an `inference`-scoped API key — pass `--key`, set `PETABYTE_API_KEY`, or
save it as `api_key` in your CLI config; the answer prints to stdout (the token/cost line to stderr).
Model management is included — `petabyte pull <publisher/model>`, `petabyte model list/inspect/remove`,
and `petabyte run <model-id>` work straight from the pip install (the model hub is pure standard
library, so it adds no dependency beyond httpx):
```bash
petabyte pull Qwen/Qwen3-8B          # verified, resumable download into ~/.petabyte
petabyte model list                  # what's in your local cache
petabyte run Qwen/Qwen3-8B           # start a model runtime
```
The package is built from the repo-root `pyproject.toml` (`name = "petabyte-client"`; the command
stays `petabyte`), which bundles this
CLI module plus the `modelhub` package. From a source checkout you can also run it directly with
`python cli/petabyte.py <cmd>`, or `pip install .` from the repo root.
`run` books the cheapest matching GPU, escrows funds, dispatches the notebook,
polls, and prints the result. `.ipynb` (code cells) and `.py` files are supported.

### Output & config
- **Human output:** semantic colour (green=ok, yellow=pending, red=error, cyan=info) with
  aligned tables. Colour turns **off** automatically when stdout is not a TTY or `NO_COLOR`
  is set — safe for scripts and CI. The buyer `petabyte` client uses its own small inline
  colour helpers (no dependency beyond `httpx`); it does **not** import `cli_ui.py`.
- `PETABYTE_API_URL` (or `--api`) selects the API; `PETABYTE_CONFIG=/path/cli.json`
  isolates the saved token/API (handy in CI or tests).

> Note: a stable `--json`/`PETABYTE_JSON` machine-readable mode and a `doctor`
> health-gate command exist in the **seller agent CLI** (`lumaris_agent/agent_cli.py`),
> not in this buyer client. The shared `cli_ui.py` presentation layer is likewise used
> by the agent and desktop app, not by `petabyte.py`.

## Dashboard
Served by the API at `/` (same-origin, no CORS setup). Start the API and open
`http://localhost:8000/` — live nodes/jobs/GMV stats, wallet + deposit, the GPU
inventory with a live $/hr-vs-AWS savings column, and one-click job runs.

Both need an attested, online seller node (run the agent) to actually execute jobs.
