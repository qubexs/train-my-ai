# Production Laravel Reference

## Lifecycle

Request → global middleware → route (+route middleware) → controller
(FormRequest validated first!) → policy authorize → action → API Resource →
response. `bootstrap/app.php` wires exceptions/routes/middleware (no Kernel
file in modern Laravel). `php artisan route:list` audits truth.

## Routing/controllers/requests

`apiResource` (JSON, no create/edit) vs `resource`; `__invoke` singles;
implicit binding (`{user:slug}`) + `missing()` 404s; named routes + `route()`
helper (never hardcode URLs); groups share prefix/name/middleware; throttles
(`throttle:60,1`) + signed URLs (`signedRoute`) for trust.

## Blade/views

`{{ }}` escaped always, `{!! !!}` almost never; components + slots
(`$attributes->merge`); `@push/@stack` assets; `@can/@error/@auth` mirror
backend (cosmetic only!); `@csrf` + `@method` + `old()` forms survive
round-trips; `@vite` assets with HMR.

## Eloquent/data

Relationships map domain shape (pivot alphabetical, morphs for shared);
`with()` kills N+1 (verify via Telescope); scopes name rules; new-style
Attribute accessors; SoftDeletes + scoped uniques; `$fillable` explicit;
migrations reversible (`down()` tested!); factories+states+sequences;
`RefreshDatabase` tests; `chunk/cursor/lazy` millions; `upsert` races;
`firstOrCreate` idempotent seeds.

## Auth/queues/events

Breeze (starter) / Fortify (headless actions) / Sanctum (SPA cookies + mobile
tokens with abilities/expiry) / Socialite (provider-ID linking); gates +
policies + `$authorize` (UI only mirrors); jobs idempotent with `tries/backoff`,
chains/batches with catch, `failed()` alerts; events fan out (mail/stock/
analytics independent); broadcast privates (auth channels!) + `toOthers()`;
notifications per-channel, queued; cache `remember` + tags (redis only);
`Storage::disk` per env; markdown mail queued; scheduler single-cron
(`withoutOverlapping`, `onOneServer`, `emailOutputOnFailure`).

## Ops

`optimize` (config/route/view/event cache — clear every deploy, `.env` edits
ignored when cached!); Telescope local-only + pruned; Horizon supervises
(scale on waits); Octane needs stateless code; Pint/Larastan/Shift automate
style/analysis/upgrades; Forge provisioned vs Vapor serverless per traffic;
Livewire (server reactivity, validate server-side!) / Inertia (no-API SPA) /
Filament (policy-enforced admin) per team shape.
