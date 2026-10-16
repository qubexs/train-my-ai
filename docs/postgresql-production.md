# Production PostgreSQL Reference

## Data modeling

Normalize first (FKs, no repeating groups); denormalize measured hot paths.
PKs: `uuid v7` distributed (index-friendly) or `bigint GENERATED ALWAYS AS
IDENTITY`. `citext` emails, `timestamptz` always (never bare timestamp),
`CHECK`/`DOMAIN`/`EXCLUDE` invariants at DB level (apps race, constraints don't).

## Querying

`EXPLAIN (ANALYZE, BUFFERS, TIMING OFF)` before guessing; `pg_stat_statements`
top-expensive first. CTEs for readability, `LATERAL` for per-row subqueries,
window frames (`ROWS` vs `RANGE` vs `GROUPS`) explicit, recursive CTEs for trees.
`RETURNING` + CTE chains avoid round-trips; `ON CONFLICT` upserts (no
read-then-write races); `COPY` bulk loads (disable indexes, load, re-enable).

## JSON/search/geo/AI

`jsonb` operators (`->`, `->>`, `@>`, `?`) + `GIN` (`jsonb_path_ops` for paths);
`tsvector` + `GIN` + `ts_rank` real search (stemming/ranking/highlight);
`pg_trgm` fuzzy (`%` similarity) + `unaccent`; PostGIS `geography` for meters
(`ST_DWithin` + GiST); `pgvector` HNSW cosine (`<=>`, tune `ef_search`).

## Indexes

B-tree default; `INCLUDE` covering (index-only scans); partial `WHERE`
(skewed 5/95); expression (match queries exactly!); BRIN for append-only time;
`REINDEX CONCURRENTLY` online (needs space). Verify every index with EXPLAIN —
unused indexes tax writes for nothing.

## Integrity/advanced SQL

RLS (`ENABLE` + `USING`/`WITH CHECK`, test per role); FKs with actions;
`SERIALIZABLE` only where needed (retry 40P01 with backoff); savepoints for
partial rollback; advisory locks for cross-host singletons (`pg_try_advisory_lock`);
`LISTEN/NOTIFY` transactional wakeups (payloads small); `SECURITY DEFINER`
locked `search_path`, audited.

## Operations

`VACUUM (ANALYZE)` routine + autovacuum tuned (scale factors per big table);
WAL archiving + `pg_basebackup` = PITR (test restores quarterly!); pgBackRest
(stanzas, S3, encrypted) or Barman (catalogs); streaming replicas
(`primary_conninfo`, lag monitored) + sync quorum for valued data; PgBouncer
transaction pooling (`pool_size` × instances ≤ `max_connections`); partitions
monthly + `DETACH` archival; `pg_repack` over `VACUUM FULL`.

## Tuning/auth/HA

`shared_buffers` 25% RAM, `work_mem` per-sort (watch spills), big
`maintenance_work_mem` for builds; `max_wal_size` + smooth checkpoints;
`log_min_duration_statement=500` + auto_explain plans; `scram-sha-256` HBA
(first-match order!), `verify-full` SSL; Patroni/etcd failover rehearsed;
sync standbys for durability; daily DBA queries (locks, bloat, lag).

## Beginner path (W3Schools flow)

Install → `psql` → `CREATE DATABASE` → tables (`CREATE/INSERT/SELECT`) →
evolve (`ADD/ALTER/DROP COLUMN`) → filter (`WHERE`, never `= NULL`) →
summarize (MIN/MAX/COUNT/SUM/AVG) → pattern (`LIKE/ILIKE`, leading `%` kills
btree) → sets (`IN`, `BETWEEN`, `AS` aliases) → joins (INNER/LEFT/RIGHT/FULL/
CROSS map) → stack (`UNION` vertical vs JOIN horizontal) → group (`GROUP BY`
+ `HAVING` post-filter) → membership (`EXISTS` fastest, `ANY`/`ALL`) →
branch (`CASE` searched vs simple).

## Deep reference (pg.center wiki map)

SQLSTATE codes (23505/23503/40001/40P01/57014 — branch on codes, never text);
catalogs (`pg_class`, `pg_index`, `pg_attribute`); index AMs (btree/hash/GiST/
SP-GiST/GIN/BRIN matched to shape); plan nodes (Seq/Index/Nested/Hash/Merge/
Gather/Memoize read fluently); wait events (Lock/IO/LWLock/Client/IPC name the
bottleneck); lock ladder (ACCESS SHARE → ACCESS EXCLUSIVE, DDL blocks all);
auth methods (trust never remote, scram modern, peer local, cert/LDAP central);
predefined roles (`pg_read_all_data`, `pg_monitor`, `pg_signal_backend`);
psql slashes (`\\l \\dn \\dt \\df \\du \\x \\timing \\copy \\watch`);
CLI tools (createdb, pg_isready, pg_lsclusters); conninfo keywords (sslmode,
timeouts, app names); PLs (plpgsql default, plpython, plv8); FTS pieces
(parsers/dicts/configs, `ts_debug`); collations (fixed at initdb!); decoding
plugins (wal2json/pgoutput, slots monitored); extensions vetted (trusted vs
superuser); upgrades (`pg_upgrade --check`, extensions rebuilt, ANALYZE after);
GUC reload-vs-restart (`pg_file_settings` shows pending).
