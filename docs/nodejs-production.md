# Production Node.js Reference

## Runtime

`node --version` + `.nvmrc` pin runtime; `node --watch` dev reload;
`node --test --test-coverage` zero-dep suites; `--env-file=.env` (v20.6+)
loads env natively; `--permission` least-privilege; `--max-old-space-size`
heap ceiling; SEA single binaries for distribution.

## Express/Fastify/Koa/Nest

- Express: middleware order is the framework (`use` → routes → 4-arg errors);
  wrap async routes (or v5); `Router()` per module; helmet+cors+rate-limit first.
- Fastify: schemas validate+serialize (2x throughput), encapsulated plugins,
  built-in logger. Koa: `await next()` onion for timing/logging wrappers.
- NestJS: controllers/providers/modules (DI), guards→interceptors→pipes→handler.

## Auth/Data/Jobs

- JWT: access 15m + rotating refresh; bcrypt cost-12 (argon2id new systems);
  OAuth2 code flow, TOTP (`otplib`) + backup codes.
- Data: Prisma (DX) / Drizzle (SQL-first); pools sized (`max` × instances ≤
  server limit); mongoose lean reads, populate sparingly.
- Jobs: BullMQ (retries/backoff/concurrency, Redis-backed) out of requests;
  node-cron in-process only, system cron for critical.

## Outbound/Realtime/Deploy

- Outbound: timeout+retry+breaker on every call (axios-retry); pools (undici)
  for hot paths; verify Stripe webhooks on RAW body.
- Realtime: ws + ping/pong heartbeats; Socket.IO rooms + Redis adapter +
  sticky sessions for scale.
- Deploy: PM2 ecosystem (cluster/logrotate/deploy) or systemd unit (cgroup/
  journal); SIGTERM drain + health-flip + 10s force cap; Dockerfile multi-stage
  (build vs alpine runtime, non-root, `--omit=dev`, HEALTHCHECK).
- Publish: `files:` whitelist, `prepublishOnly` build, `--provenance` supply chain.
