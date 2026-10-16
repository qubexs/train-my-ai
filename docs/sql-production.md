# Production SQL Reference (cross-engine)

## Standard advanced

Set ops (`EXCEPT`/`INTERSECT`, `ALL` keeps dupes!); `VALUES` row constructors
(test data inline!); `TABLESAMPLE` (method documented!); `GROUPING SETS/CUBE/
ROLLUP` multi-level (NULL disambiguated via `GROUPING()`!); `QUALIFY` filters
windows (subquery portable!); frames explicit (`ROWS` physical vs `RANGE`
peers!); `LEAD/LAG IGNORE NULLS`, `NTH_VALUE/NTILE/CUME_DIST/PERCENT_RANK`;
`FILTER (WHERE)` conditional aggregates; `LATERAL` per-row (small N!);
`PIVOT` static vs app-dynamic; `MERGE` atomic sync (all branches tested!);
`RETURNING`/`OUTPUT` (no second round-trip!); pagination with stable ORDER
(keyset at scale!); JSON operators per engine + generated indexed columns;
temporal/versioned/history tables; graph (`MATCH`, shortest path); spatial
(SRID matched!); fulltext (ngram CJK!); columnstore (OLAP!) vs row (OLTP!);
in-memory (PINNED, RAM-bounded!); ledger (digest verified!); masking/RLS
(non-bypassable!); SSIS/SSRS/SSAS/PowerBI/DAX/M stack; lakehouse medallion
+ dbt tested + `VACUUM` retention; engines (Trino pushdown! Snowflake prune!
BQ partition-required! ClickHouse ORDER! DuckDB serverless!); single-table
Dynamo (access-designed!) / Cassandra keys (hotspots killed!); SQLite FTS5/
JSON/upsert/`VACUUM INTO`; CDC (Debezium/outbox transactional!).

## Analytics patterns (training seeds)

Funnel (`MAX(CASE)` pivot + `NULLIF` zero-guard); cohort retention (LEFT JOIN
keeps churned!); sessionization (gaps-and-islands: LAG + cumulative SUM!);
running totals + rank + deltas (framing explicit!); dedupe keep-one
(transaction + UNIQUE after!); SCD MERGE (`IS DISTINCT FROM` NULL-safe!);
generic audit trigger (JSONB flexible!); top-N per group (`PARTITION` +
outer filter!); date gaps-and-islands (date-minus-rownumber!); slow-query
triage checklist (rank → EXPLAIN → classify → fix-one → monitor!).
