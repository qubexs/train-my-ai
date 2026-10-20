# Production MySQL Reference (InnoDB)

## Setup/safety

`utf8mb4` + `_unicode_ci` from day one (emoji-safe); `caching_sha2_password`
(modern), host-scoped least-privilege users (never `%`), secrets vaulted;
`STRICT_TRANS_TABLES` + `ONLY_FULL_GROUP_BY` (bad data fails at write time);
`sql_mode` reviewed (`SELECT @@sqlmode`), never lenient-by-accident.

## InnoDB internals

Buffer Pool 70-80% RAM (the #1 knob, instances split contention, online resize);
redo (crash recovery, sized to hourly writes) + undo (MVCC versions, watch long
txns); doublewrite (torn pages, disable only on atomic storage with proof);
change buffer (non-unique deferral); AHI (disable on contention, measured);
gap/next-key locks (REPEATABLE READ phantoms; indexed equality narrows);
FKs need matching indexes (parent writes lock children otherwise!).

## Indexes/querying

B-tree default; composite leftmost-prefix (equality first, range last);
covering (`Using index`); ICP (`Using index condition`); FULLTEXT/SPATIAL per
shape; invisible/descending/functional variants; `EXPLAIN FORMAT=JSON/TREE`
before guessing; histograms (`ANALYZE` fresh); hints (`MAX_EXECUTION_TIME`)
last resort. Slow log always on (`long_query_time=0.5`); digest ranks
(Performance Schema/sys schema); `sys` views humanize (top queries, unused/
redundant indexes, lock waits).

## Transactions/replication/HA

REPEATABLE READ default vs READ COMMITTED (fewer gaps); deadlocks retried with
backoff (lock ordering prevents); savepoints nest; GTID (`MASTER_AUTO_POSITION`,
errant txns investigated!); ROW binlog (CDC-safe); heartbeat lag (not
Seconds_Behind lies); parallel workers; Group Replication (odd members, quorum,
single-primary default); Router splits (app oblivious); ProxySQL rules
(`FOR UPDATE` → writer!); Orchestrator rehearsed failovers; semi-sync acks
for valued writes; delayed/filtered replicas per need.

## Backup/users/ops

`--single-transaction` dumps (InnoDB, no locks) / mydumper parallel /
XtraBackup hot-physical (TB scale); binlog = PITR between backups; restores
fire-drilled quarterly (Schrödinger's backups otherwise). Roles group
privileges (`SET DEFAULT ROLE`); expiry/lockout rotates; audit plugins
filtered; TDE at rest (key management is the work); `REQUIRE SSL/X509`;
offboarding same-day (`DROP USER`, creds rotated). Bulk: `LOAD DATA`
(`secure_file_priv` jail) / mysqlimport; online DDL (`INPLACE/NONE`,
gh-ost/pt-osc for hot giants, watch replica lag!); blue-green atomic renames;
ordered UUID PKs (`UUID_TO_BIN(...,1)`); CDC (Debezium/Maxwell/outbox
transactional); migrations versioned (Flyway/Liquibase); ORMs picked per
need (velocity vs explicit SQL, N+1 watched everywhere).

## Beginner path (tutorialspoint flow)

Install → connect (`mysql -u -p`) → `CREATE DATABASE` → tables
(`CREATE/SHOW/DESCRIBE/ALTER/RENAME/CLONE/TRUNCATE/DROP`) → users
(`CREATE/DROP/SHOW`, passwords, GRANT/REVOKE, lock/unlock) → queries
(INSERT/SELECT/UPDATE/DELETE, REPLACE/IGNORE/UPSERT, INSERT..SELECT) →
views (create/update/drop/rename) → indexes (unique/clustered/show/drop) →
operators (WHERE/LIMIT/DISTINCT/ORDER/GROUP/HAVING, AND/OR/LIKE/IN/ANY/EXISTS/
NOT/BETWEEN/UNION/MINUS/INTERSECT/INTERVAL) → joins (all 7 + self/delete/update,
union-vs-join) → keys (unique/primary/foreign/composite/alternate) → triggers
(full lifecycle before/after × insert/update/delete) → types (varchar/boolean/
enum/decimal/int/float/bit/blob/set) → regex (`RLIKE`, `regexp_*`) → fulltext
(natural/boolean/expansion/ngram) → functions (date/numeric/string/aggregate)
→ misc (NULL/transactions/sequences/duplicates/injection/subquery/check/engines/
CSV/UUID/CTE/cascade/upsert/partitions/cursor/procedures/SIGNAL/charset/ROLLUP/
literals/EXPLAIN/JSON/stddev/random/processlist/coalesce).

## Manual map (dev.mysql.com 8.4)

Install/upgrade paths → tutorial (connect, batch mode) → programs (`mysql`,
`mysqladmin`, `mysqlcheck`, `mysqldump`, `mysqlimport`, `mysqlshow`,
`mysqlslap`, `mysqlbinlog`, `mysqldumpslow`, `perror`) → security (privileges,
roles, MFA, keyring, audit, firewall, masking, SELinux) → backup (mysqldump
tips, PITR positions) → optimization (SELECT/subquery/indexes/InnoDB/locks/
EXPLAIN/trace/benchmarks) → charset/collations → data types → functions
(JSON/window/spatial/cast/encryption) → SQL statements → data dictionary →
InnoDB deep → replication/Group Replication → Shell/Cluster/ReplicaSet →
partitioning → stored objects → INFORMATION_SCHEMA/Performance/sys schemas →
connectors/Workbench → errors/glossary.
