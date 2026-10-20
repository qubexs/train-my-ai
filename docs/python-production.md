# Production Python Reference

## Async/concurrency

`asyncio.gather` fan-out (ordered!) + `create_task` background + `Queue`
backpressure + `as_completed` first-done; aiohttp sessions pooled (never
per-request!) + timeouts/retries; httpx sync/async dual; trio nurseries
(structured, all-children-finish!) + AnyIO portable; threads for blocking I/O
(GIL!), processes for CPU (picklable! `__main__` guard Windows!); locks minimal
sections, events coordinate.

## Typing/data

Hints at boundaries (`int | str`, `X | None`, `list[int]`, `Callable`),
inference inside; `Protocol` structural, `@overload` shapes, `Literal`
commands, `TypedDict` JSON, `TypeVar` generics; `match` dispatches (guards
refine!); walrus removes double-eval; f-strings specs + `=` debug; dataclasses
(`field` factories!, frozen/slots/kw_only); attrs vs pydantic (schema!) vs
msgspec (speed!) per boundary.

## CLI/TUI/logging

argparse subparsers (typed, required!); Click prompts/groups + CliRunner tests;
Typer hints-are-CLI; Rich markup/tables/progress (transient!); Loguru
zero-config + bound context; stdlib `logging.dictConfig` versioned (rotating
bounds!); structlog bound flows (humans AND machines!).

## Testing/packaging

pytest fixtures scoped (cost-matched!) + parametrize named + mock boundaries
+ asyncio mode + tmp_path; Hypothesis properties (shrinks!) + examples pin;
tox/nox matrices + coverage gates (not vanity!); pip editable/frozen/hashed;
`pipdeptree`/`audit` scheduled; Poetry groups + locked; uv speed + pip compat;
conda exported + base never polluted.

## Full apps (training seeds)

Async scraper (semaphore!), Typer subcommands, Rich progress+table, FastAPI
CRUD (422 auto!), pytest fixtures+mocks, pydantic-settings (fail fast!),
ProcessPool (picklable!), JSON rotating logs, `src/` packaging layout,
flock contextmanager, ethical scraper (robots!), argparse git-style.
