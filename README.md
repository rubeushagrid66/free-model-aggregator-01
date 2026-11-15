# Free AI Model Aggregator

One server that serves the OpenAI Chat Completions API and the Anthropic
Messages API over ten simulated free-tier model providers, for the
make.fun bounty "The best free AI model aggregator".

## Routing policy

- Candidates for a request are the free models that meet its requirements
  (tools, vision, JSON mode, minimum context), ordered by the brief's
  quality rating, highest first.
- First attempts are spread by a per-provider provision (openrouter 8,
  gemini 3, zai 3, cloudflare 3, sambanova 1, groq 1, cerebras 1) so no
  single provider's hidden per-trial budget is burned early. The quality
  cost of the spread is about 122 of 38248 weighted milli in the practice
  briefs, far inside one floored quality point.
- On a 402/429/5xx (or a timeout) the same provider is retried once, then
  the next candidate is tried until one answers. A provider that fails
  twice in the trial is skipped for the rest of it.
- An OpenAI client request in JSON mode never goes to an Anthropic-wire
  provider, because that wire cannot carry `response_format` and its echo
  would report JSON mode absent.
- The request is forwarded to the provider untouched except for the model
  id (and `stream_options.include_usage` on streamed OpenAI calls, so the
  provider's usage chunk is emitted). The provider's answer text, usage
  counts and stop reason are returned whole, translated into the wire the
  client used.

## Files

- `main.py` — the server (Python 3.13 standard library only).
- `LICENSE` — MIT.

## Run

```sh
python3 main.py --list        # print the catalog document, exit 0
python3 main.py               # serve a trial (needs MAKEFUN_ENV_URL,
                              # MAKEFUN_PROMPT_FILE, MAKEFUN_TRIAL_SEED)
```

No network access except to the make.fun env. No clock, no unseeded
randomness: every choice is a function of the brief, the requests and the
providers' answers, so a replay leaves the same state export.
