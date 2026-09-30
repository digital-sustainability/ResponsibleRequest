# responsible-request

Load-aware rate limiting and request logging for OpenAI-compatible LLM gateways (LiteLLM, vLLM, Ollama, …).

Large experiments can starve the other users of a shared inference server. Hard RPM limits are the usual answer, but they leave the server idle at night and still hurt others at peak times. `responsible-request` instead **watches the latency of your own requests**:

- While latency stays close to its baseline, the server is idle, so the client ramps up to `max_rpm`.
- As soon as latency reaches 3× the baseline (or the server returns 429/5xx/timeouts), other users are active, so the client drops to `min_rpm` at once.
- Once latency is back to normal, the client ramps up again.

It works as a drop-in for `openai.AsyncOpenAI` and logs every request (tokens, timing, throttle state, and optionally full messages) to JSONL and/or SQLite.

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
    throttle=rr.ThrottleConfig(max_rpm=300, min_rpm=5),
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

client = openai.AsyncOpenAI(http_client=rr.http_client(rr.ThrottleConfig(max_rpm=300)))
```

It also works as a plain fixed-rate limiter for everyday use: `rr.ThrottleConfig.fixed(60)`.

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
rr.response_format_from_model(Answer)                 # just the strict json_schema response_format

# Attach metadata to every request record created inside the block
with rr.tags(experiment="ablation-3", fold=2):
    ...

# Measure and pin the baseline explicitly, e.g. at night with a representative request
await rr.calibrate(client, "gpt-oss:120b", n=10, messages=representative_messages)
```

## How the throttle works

```
             ratio >= high_ratio (3.0) or 429/5xx/timeout
   ┌────────────────────────────────────────────────────────────┐
   │                                                            ▼
WARMUP ──baseline known──► NORMAL ◄──at max_rpm── RECOVERING ◄── THROTTLED (min_rpm)
start_rpm                  ramp ×1.5 / 30 s        ramp ×1.5 / 30 s   cooldown ≥ 120 s and
                           up to max_rpm                              ratio < recover_ratio (1.5)
```

- **Pacing**: requests are spaced evenly at the current rate with no bursts ([aiolimiter](https://github.com/mjpieters/aiolimiter)), and `max_concurrency` caps the number in flight.
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
| `max_rpm` | 300 | rate when the endpoint is idle |
| `min_rpm` | 5 | rate while others are active (also the probing rate) |
| `start_rpm` | 30 | rate while the baseline is being established |
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

## Logging

The package logs through [loguru](https://github.com/Delgan/loguru) and, as loguru recommends for libraries, stays silent until a client is created with `log=` (the default `log=True` enables console messages only).

- **Console**: rate changes (INFO), switches to throttling (WARNING), and a per-model summary every `summary_interval_s`. Individual requests are never printed. With `LogConfig(console_level="INFO")` the package replaces loguru's default DEBUG handler with a quieter one.
- **Request records**: one record per HTTP attempt (SDK retries are separate records with `attempt` > 0) goes to `LogConfig(jsonl=...)` and/or `LogConfig(sqlite=...)` (table `requests`). Records are emitted at loguru's TRACE level with `extra["rr_record"]`, so you can also attach your own sinks.

| Group | Fields |
|---|---|
| core (always) | `request_id`, `timestamp`, `method`, `path`, `model`, `stream`, `attempt`, `status_code`, `error`, `tags` |
| `timing` | `sent_at`, `first_byte_at`, `finished_at`, `wait_s` (time spent throttled), `ttfb_s`, `latency_s` |
| `usage` | `prompt_tokens`, `completion_tokens`, `total_tokens`, `cached_tokens`, `reasoning_tokens` |
| `throttle` | `rpm`, `state`, `baseline`, `load_ratio`, `in_flight` |
| `response_meta` | `response_id`, `response_model`, `finish_reason`, LiteLLM call/model id and duration headers |
| `params` | all request parameters except messages/input |
| `request_body` / `response_body` | full request and response (for streams: the concatenated content) |

All groups are logged by default. Turn groups off with `LogConfig(fields={"request_body": False})`, or use `LogConfig.minimal()` (no params or bodies). **Request and response bodies can be large and may contain sensitive data.**

```python
df = rr.load_records("requests.db")   # pandas DataFrame if pandas is installed, else list of dicts
```

## Caveats

- **Your own load raises latency too.** If `max_rpm` is high enough to saturate the server by itself, the client will throttle itself. The low-percentile baseline and the dead band absorb moderate self-load. Choose `max_rpm`/`max_concurrency` so that you alone don't saturate the backend.
- **One process, one event loop per client.** Throttle state is not shared between processes. Several scripts running in parallel each throttle independently.
- **Detection latency.** At `min_rpm=5` and `window=10`, noticing that others have left takes about two minutes plus the cooldown. This is deliberate: the client stays polite longer than strictly needed.
- If the backend becomes permanently slower (e.g. a model is redeployed on smaller hardware), call `client.throttle.reset_baseline()`.

## Development

```bash
uv sync
uv run pytest
uv run ruff check . && uv run mypy src
uv run python examples/simulate_load.py
```
