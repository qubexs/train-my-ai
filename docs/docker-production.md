# Production Docker Reference (docs.docker.com + GFG curriculum)

## Concepts

Client CLI → dockerd daemon (REST) → containerd (lifecycle) → runc (OCI) →
shim per container. Containers share the host kernel, isolated via
namespaces + cgroups. Images are read-only templates; containers are
runnable instances; volumes persist data outside container filesystems.

## Dockerfile (all instructions)

`FROM` (must be first, except global `ARG`) → `WORKDIR` (never `RUN cd`) →
`COPY` (local, predictable; `--from=` stages, `--link` fast rebuilds) →
`RUN` (one concern per layer, `&&`-chained) → `ENV`/`ARG` (ARG vanishes at
runtime, never secrets) → `EXPOSE` (documents only) → `USER` (drop root,
numeric UID) → `HEALTHCHECK` (`CMD curl -f ... || exit 1`) →
`ENTRYPOINT` exec + `CMD` args (overridable defaults). Also: `LABEL`
(metadata, replaces MAINTAINER), `VOLUME`, `STOPSIGNAL`, `SHELL`,
`ONBUILD` (deferred triggers — document them), parser directives first line
(`# syntax=`, `# escape=`, `# check=`). `.dockerignore` everything not
needed (`node_modules .git *.log .env`).

Exec form `["app","--flag"]` (signals reach PID 1, no shell) vs shell form
(globs/vars, wrapped in `sh -c`). Exec + JSON needs double-quoted escapes.

Advanced RUN: `--mount=type=cache` (persist package caches, zero image
weight), `--mount=type=secret` (tokens never in layers — ARG/env persist
forever), `--mount=type=ssh` (private repos), `--network=none|host`.

Multi-stage: `FROM x AS build` (deps+compile) → `FROM x-slim` +
`COPY --from=build` (runtime only). Order stable-first for cache hits.

## Compose

Services + `ports` + `environment`/`env_file` + named `volumes` +
user-defined `networks` (DNS included; default bridge has none) +
`depends_on: {condition: service_healthy}` + `restart: unless-stopped` +
`healthcheck` (`test/interval/timeout/retries/start_period`) +
`secrets:` (files under `/run/secrets`, never env) + `profiles:` (optional
services) + `develop.watch` (live sync). `.env` never committed.

## Engine: storage, network, registry

- Storage: overlay2 default; named volumes (data) vs bind (dev code) vs
  tmpfs (RAM secrets); `docker system df` per-type reclaimable.
- Network: bridge (single-host), host (wire speed, port clashes), none,
  overlay (swarm VIP/DNSRR), macvlan (LAN IPs). EXPOSE documents, `-p`
  publishes (`127.0.0.1:` for loopback-only), `-P` randomizes.
- Registry: `login → tag → push`; private via `registry:2` (+TLS/auth past
  localhost); pin digests, never floating `latest` in prod.

## Swarm, security, debug

- Swarm: `init --advertise-addr` (odd managers 3/5 Raft quorum), services
  with `--replicas/--update-delay/--update-failure-action rollback`.
- Security: non-root USER, `--read-only` + tmpfs, no-new-privileges,
  `--cap-drop ALL`, pinned digests, scanned bases, distroless, hadolint-gated
  Dockerfiles, secrets as files.
- Debug: exit codes (125 daemon, 126/127 exec, 137 OOM, 139 segfault, 143
  TERM) → `logs --tail` → `inspect -f '{{.State}}'` → `events --since`.
- Cleanup: `system df` before/after; prune containers/images/networks freely,
  `volume prune -a` never blindly (named data dies).
