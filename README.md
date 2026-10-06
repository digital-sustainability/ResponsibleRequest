# responsible-request

Responsible LLM requests for research, against any OpenAI-compatible endpoint (LiteLLM, OpenRouter, vLLM, Ollama, OpenAI, …). It works as a drop-in for `openai.AsyncOpenAI`.

- **Polite**: load-aware throttling, so large experiments don't starve the other users of a shared inference server.
- **Reproducible**: every request is logged (parameters, messages, responses, tokens, timing, provider, system fingerprint) to JSONL and/or SQLite, each client writes a run manifest (versions, git commit, configuration), and `rr.reproducible()` pins the decoding parameters.
- **Cost-safe**: a response cache means a restarted script only sends what is still missing, and per-request cost tracking with an optional budget stops spending before it gets out of hand, even across restarts.

## Load-aware throttling in a nutshell

Large experiments can starve the other users of a shared inference server. Hard RPM limits are the usual answer, but they leave the server idle at night and still hurt others at peak times. `responsible-request` instead **watches the latency of your own requests**:

- While latency stays close to its baseline, the server is idle, so the client ramps up to `max_rpm`.
- As soon as latency reaches 3× the baseline (or the server returns 429/5xx/timeouts), other users are active, so the client drops to `min_rpm` at once.
- Once latency is back to normal, the client ramps up again.

For commercial APIs with generous rate limits of their own (OpenAI, OpenRouter, …), load-aware throttling is not needed: use a plain fixed-rate limiter, `rr.ThrottleConfig.fixed(rpm)`. In fixed mode a 429 does not change the rate: it pauses only the affected model (and endpoint) for the time in the `Retry-After` header, or for an exponential backoff (`backoff_initial_s`, doubling up to `backoff_max_s`) if there is none, plus a random jitter of up to `backoff_jitter_s`. The adaptive throttle instead treats a 429 as a sign of other users and drops to `min_rpm` for at least `cooldown_s`.

## Installation

```bash
pip install git+https://github.com/digital-sustainability/ResponsibleRequest.git
# or: uv add git+https://github.com/digital-sustainability/ResponsibleRequest.git
pip install "responsible-request[pandas,progress] @ git+https://github.com/digital-sustainability/ResponsibleRequest.git"
```

Requires Python ≥ 3.10 and `openai` ≥ 1.40 (it works with both the `httpx`-based 1.x/2.x SDKs and the `httpx2`-based 3.x SDK).

## Quickstart

```python
import responsible_request as rr

client = rr.AsyncOpenAI(                       # same arguments as openai.AsyncOpenAI
    base_url="https://inference.example.org/api/v1",
    api_key="sk-...",
    throttle=rr.ThrottleConfig(max_rpm=15, min_rpm=1),
    log=rr.LogConfig(sqlite="requests.db"),
    default_params=rr.reproducible(seed=42),   # temperature=0, top_p=1, seed=42 unless set per call
)

response = await client.chat.completions.create(model="gpt-oss:120b", messages=[...])
print(client.throttle.stats())
await client.close()
```

As with the OpenAI SDK, `base_url` and `api_key` can come from `OPENAI_BASE_URL` / `OPENAI_API_KEY` instead. For the examples in this repo, copy `.env.example` to `.env` and run them with `uv run --env-file .env python examples/quickstart.py`.

For existing code, only the HTTP client has to change:

```python
import openai, responsible_request as rr

client = openai.AsyncOpenAI(http_client=rr.http_client(rr.ThrottleConfig(max_rpm=15)))
```

It also works as a plain fixed-rate limiter for everyday use: `rr.ThrottleConfig.fixed(60)`.

With OpenRouter (or any other commercial endpoint), a typical setup is:

```python
client = rr.AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    throttle=rr.ThrottleConfig.fixed(120),
    log=rr.LogConfig(sqlite="requests.db"),
    cache=True,                                 # a re-run only sends what is missing
    cost=rr.CostConfig(budget_usd=5),           # stop at $5, counting earlier runs in requests.db
    run=rr.RunConfig(name="ablation-3"),
    default_params=rr.reproducible(seed=42),
)
```

See `examples/openrouter.py`.

## Helpers

```python
# Many requests, paced by the throttle, results in input order (exceptions returned, not raised)
results = await rr.run_batch(client, [{"model": m, "messages": msgs} for msgs in dataset], progress=True)
results = rr.run_batch_sync(client, requests)        # from a plain script

# Structured output validated with Pydantic; the validation error is fed back to the model on failure
class Answer(BaseModel):
    label: str
    confidence: float

answer = await rr.structured(client, model=m, messages=msgs, schema=Answer, retries=2)
# (if `content` is empty, the answer is read from `reasoning`, then `reasoning_content`)
rr.response_format_from_model(Answer)                 # just the strict json_schema response_format

# System prompt marked for OpenRouter's prompt cache (cache_control: ephemeral)
messages = [rr.cached_system_message(long_instructions), {"role": "user", "content": question}]

# Attach metadata to every request record created inside the block
with rr.tags(experiment="ablation-3", fold=2):
    ...

# Measure and pin the baseline explicitly, e.g. at night with a representative request
await rr.calibrate(client, "gpt-oss:120b", n=10, messages=representative_messages)
```

`rr.cached_system_message()` only asks the provider to cache the prompt prefix (cheaper input tokens, the model still runs), whereas `cache=True` replays complete stored responses from your SQLite or JSONL log without sending the request.

## How the throttle works

```
             ratio >= high_ratio (3.0) or 429/5xx/timeout
   ┌────────────────────────────────────────────────────────────┐
   │                                                            ▼
WARMUP ──baseline known──► NORMAL ◄──at max_rpm── RECOVERING ◄── THROTTLED (min_rpm)
start_rpm                  ramp ×1.5 / 30 s        ramp ×1.5 / 30 s   cooldown ≥ 120 s and
                           up to max_rpm                              ratio < recover_ratio (1.5)
```

- **Pacing**: requests are spaced evenly at the current rate with no bursts, and `max_concurrency` caps the number in flight. One `Throttle` can be shared by several clients, also across threads that each run their own event loop (e.g. `asyncio.run` per thread), and then enforces one rate for all of them:

  ```python
  throttle = rr.Throttle(rr.ThrottleConfig.fixed(60))
  def worker(items):  # runs in its own thread
      asyncio.run(work(rr.AsyncOpenAI(throttle=throttle, ...), items))
  ```
- **Per model**: every (endpoint, model) pair has its own lane, because load on one backend says nothing about another. Only `/chat/completions` and `/embeddings` feed the latency estimate (`observe_paths`). Other endpoints (audio, …) are paced and react to errors only. Requests with server-side tools (MCP, web search) are ignored for load estimation, because their latency includes external calls.
- **Signal**: by default, latency per completion token (`latency / max(completion_tokens, 16)`), so long answers and reasoning traces don't look like load. For streamed requests the signal is the time to first byte. The current value is the median of the last `window` requests.
- **Baseline**: the 10th percentile of the signal over the last 30 minutes, needing `warmup_requests` samples first. Samples taken while throttled are excluded, so a long busy period does not become the new "normal". You can pin it with `ThrottleConfig(baseline=...)` or `rr.calibrate(...)`, and reset it with `client.throttle.reset_baseline()`.
- **Dead band**: between `recover_ratio` and `high_ratio` the rate stays where it is, which prevents oscillation.

`examples/simulate_load.py` runs the whole loop against a simulated server (time-compressed) where other users appear for a while:

```
 t [s] others     RPM       state   load
   8.5      0    1800      normal   1.00
  12.5     48    1800      normal   1.00
  13.5     48      60   throttled   3.56
  ...
  30.6      0      60  recovering   1.00
  41.6      0    1800      normal   1.00
```

### Configuration (`ThrottleConfig`)

| Parameter | Default | Meaning |
|---|---|---|
| `max_rpm` | 15 | rate when the endpoint is idle |
| `min_rpm` | 1 | rate while others are active (also the probing rate) |
| `start_rpm` | 2 | rate while the baseline is being established |
| `max_concurrency` | 32 | max requests in flight per model |
| `high_ratio` | 3.0 | throttle when latency ≥ this × baseline |
| `recover_ratio` | 1.5 | ramp up only when latency < this × baseline |
| `cooldown_s` | 120 | minimum time at `min_rpm` after the last high-load signal |
| `ramp_factor` / `ramp_interval_s` | 1.5 / 30 | multiplicative ramp-up step and its interval |
| `metric` | `"latency_per_token"` | `"latency"`, `"ttfb"`, or a callable `RequestRecord -> float` |
| `window` | 10 | requests in the rolling median |
| `warmup_requests` | 20 | samples needed before the baseline is trusted |
| `baseline` | None | pin the baseline instead of estimating it |
| `baseline_percentile` / `baseline_window_s` | 10 / 1800 | baseline estimator |
| `observe_paths` | chat, embeddings | endpoints whose latency is used |
| `estimator_factory` | None | plug in your own `LoadEstimator` (e.g. reading queue depth from Prometheus) |
| `backoff_initial_s` / `backoff_max_s` | 1 / 60 | fixed mode: pause after a 429 without `Retry-After`, doubling per further 429 |
| `backoff_jitter_s` | 1 | fixed mode: random extra time added to every 429 pause |

## Logging

The package logs through [loguru](https://github.com/Delgan/loguru) and, as loguru recommends for libraries, stays silent until a client is created with `log=` (the default `log=True` enables console messages only).

- **Console**: rate changes (INFO), switches to throttling (WARNING), and a per-model summary every `summary_interval_s`. Individual requests are never printed. With `LogConfig(console_level="INFO")` the package replaces loguru's default DEBUG handler with a quieter one.
- **Request records**: one record per HTTP attempt (SDK retries are separate records with `attempt` > 0) goes to `LogConfig(jsonl=...)` and/or `LogConfig(sqlite=...)` (table `requests`). Records are emitted at loguru's TRACE level with `extra["rr_record"]`, so you can also attach your own sinks.

| Group | Fields |
|---|---|
| core (always) | `request_id`, `timestamp`, `method`, `path`, `model`, `stream`, `attempt`, `status_code`, `error`, `tags`, `cache_key`, `cache_hit`, `run_id` |
| `timing` | `sent_at`, `first_byte_at`, `finished_at`, `wait_s` (time spent throttled), `ttfb_s`, `latency_s` |
| `usage` | `prompt_tokens`, `completion_tokens`, `total_tokens`, `cached_tokens`, `reasoning_tokens`, `cost_usd` |
| `throttle` | `rpm`, `state`, `baseline`, `load_ratio`, `in_flight` |
| `response_meta` | `response_id`, `response_model`, `finish_reason`, `system_fingerprint`, `provider`, `gateway_request_id`, `upstream_duration_ms`, `provider_meta` (see below) |
| `params` | all request parameters except messages/input |
| `request_body` / `response_body` | full request and response (for streams: the concatenated content) |

All groups are logged by default. Turn groups off with `LogConfig(fields={"request_body": False})`, or use `LogConfig.minimal()` (no params or bodies). **Request and response bodies can be large and may contain sensitive data.**

```python
df = rr.load_records("requests.db")   # pandas DataFrame if pandas is installed, else list of dicts
```

JSONL logs rotate at `rotation="500 MB"` (any loguru rotation). LLM logs are repetitive and compress well, so finished segments can be compressed with `compression="gz"`, `"bz2"`, `"xz"` or `"zst"` (zstd needs Python 3.14+ or `pip install responsible-request[zstd]`); the file being written stays plain. Without rotation, the file is compressed when the client's sinks are removed. `load_records("requests.jsonl")`, the cache and the cost budget read all segments, compressed or not, and `load_records` / `CacheConfig(jsonl=...)` also accept a single archive such as `requests.2026-10-06_12-00-00_000000.jsonl.zst`.

```python
log = rr.LogConfig(jsonl="requests.jsonl", rotation="100 MB", compression="zst")
```

### Provider metadata

Gateways and routers report different extra information. *Extractors* map it to the generic `response_meta` fields and put everything else into the `provider_meta` JSON column. All built-in extractors run on every response and only fill in what they find, so nothing has to be configured:

| Extractor | Reads |
|---|---|
| `openai_compat` | `x-request-id` / `request-id` → `gateway_request_id`, `openai-processing-ms` → `upstream_duration_ms` |
| `litellm` | `x-litellm-call-id` → `gateway_request_id`, `x-litellm-response-duration-ms` → `upstream_duration_ms`, `x-litellm-response-cost` → `cost_usd`, all other `x-litellm-*` headers → `provider_meta["litellm"]` |
| `openrouter` | `provider` → `provider`, `usage.cost` → `cost_usd`, `usage.is_byok`/`cost_details` and `native_finish_reason` → `provider_meta["openrouter"]` |

Add your own for other endpoints:

```python
def my_gateway(record, headers, body):       # body: parsed JSON, stream summary, or None
    if "x-queue-ms" in headers:
        rr.providers.meta(record, "my_gateway")["queue_ms"] = float(headers["x-queue-ms"])

client = rr.AsyncOpenAI(..., extractors=[my_gateway])
```

Databases written by version 0.1 keep their `litellm_*` columns; new records leave them empty.

### Runs

Every client is a *run* with its own `run_id`, which every record carries. With a JSONL or SQLite log, the first request also writes a run manifest: the run name and `metadata`, start time, endpoint (scheme and host only, never the API key), versions of `responsible-request`, `openai` and Python, platform, hostname, command line, working directory, git commit and dirty flag, and the complete throttle, log, cache and cost configuration plus `default_params`.

```python
client = rr.AsyncOpenAI(..., run=rr.RunConfig(name="ablation-3", metadata={"dataset": "v2"}))
# or simply run="ablation-3"

runs = rr.load_runs("requests.db")            # the `runs` table, or requests.runs.jsonl for a JSONL log
df = rr.load_records("requests.db").merge(runs, on="run_id", suffixes=("", "_run"))
```

## Caching

With `cache=True`, a request that was already answered successfully is served from the logged records instead of being sent again. Re-running a script (e.g. after a crash) then only sends the requests that are still missing:

```python
client = rr.AsyncOpenAI(
    log=rr.LogConfig(sqlite="requests.db"),
    cache=True,                                  # or rr.CacheConfig(...)
)
```

- **Key**: a hash of the URL and the complete JSON request body (model, messages and all parameters, after `default_params` are applied). Any change to the prompt or parameters is a miss.
- **Source**: the records the client writes itself (`LogConfig.sqlite`, else `LogConfig.jsonl`; `response_body` must be logged), or another database or file via `rr.CacheConfig(sqlite=...)` / `rr.CacheConfig(jsonl=...)`.
- **What is served**: only non-streamed requests whose record has status 200 and no error. Streamed requests are always sent.
- **Hits** don't wait for or affect the throttle. They are logged with `cache_hit=True` (filter them out when analysing latency or token usage), and the response carries an `x-rr-cache: hit` header. `client.cache.stats()` counts hits and misses.
- **Repeated sampling**: identical requests share one cached answer. To draw several independent samples, name the tags that belong to the key:

```python
client = rr.AsyncOpenAI(log=..., cache=rr.CacheConfig(key_tags=("sample",)))
for i in range(5):
    with rr.tags(sample=i):                      # 5 different keys; a re-run gets the same 5 answers
        await client.chat.completions.create(model=m, messages=msgs, temperature=0.7)
```

Other tags (e.g. `experiment=...`) are not part of the key. Records are written asynchronously, so an identical request sent a few milliseconds after the first one may still go to the server.

## Cost and budget

With `cost=True` or a `rr.CostConfig`, every record gets a `cost_usd`:

1. the cost the provider reports (OpenRouter `usage.cost`, LiteLLM `x-litellm-response-cost`, or a custom extractor), else
2. the cost computed from token usage and `CostConfig(prices={"model": rr.Price(input=..., output=..., cached_input=...)})` (USD per million tokens; there is no built-in price list, since it would go stale), else
3. `None`.

Cache hits cost `0.0`. `client.cost.stats()` shows what was spent, and the per-model summary line and `client.throttle.stats()` include it.

```python
client = rr.AsyncOpenAI(..., log=rr.LogConfig(sqlite="requests.db"), cost=rr.CostConfig(budget_usd=5))
```

- Once `budget_usd` has been spent, new requests raise `rr.BudgetExceeded` instead of being sent (`run_batch` returns it for each remaining item). Cache hits are still served, so re-running a finished experiment works with an exhausted budget.
- With `include_logged=True` (the default), the spend already recorded in the log counts too, so restarting a script after a crash does not reset the budget. This sums **all** records in the log file, so use one database per experiment or budget.
- Requests already in flight when the budget runs out still complete, so the budget can be exceeded by their cost.
- If a budget is set but a model's responses carry no cost and no price is configured, a warning is logged once: the budget cannot see those requests.
- With `openai<2`, the SDK retries requests after an exception, so a blocked request is attempted `max_retries` more times (each blocked again, without a network call) before `rr.BudgetExceeded` is raised.

## Caveats

- **Your own load raises latency too.** If `max_rpm` is high enough to saturate the server by itself, the client will throttle itself. The low-percentile baseline and the dead band absorb moderate self-load. Choose `max_rpm`/`max_concurrency` so that you alone don't saturate the backend.
- **One process, one event loop per client.** Throttle state is not shared between processes. Several scripts running in parallel each throttle independently.
- **Detection latency.** At `min_rpm=1` and `window=10`, noticing that others have left takes about ten minutes plus the cooldown. This is deliberate: the client stays polite longer than strictly needed.
- If the backend becomes permanently slower (e.g. a model is redeployed on smaller hardware), call `client.throttle.reset_baseline()`.

## Development

```bash
uv sync
uv run pytest
uv run ruff check . && uv run mypy src
uv run python examples/simulate_load.py
```
