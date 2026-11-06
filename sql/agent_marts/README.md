# Agent PostgreSQL marts

This folder contains first-version SQL templates for four domain marts used by
the agents:

- Sales: `{{target_schema}}.v_agent_sales_sku_daily`
- Supply: `{{target_schema}}.v_agent_supply_sku_coverage_daily`
- Marketing: `{{target_schema}}.v_agent_marketing_sku_daily`
- Finance: `{{target_schema}}.v_agent_finance_sku_daily`
- Advertising decision evidence:
  `{{target_schema}}.v_agent_ad_campaign_sku_daily`
- Stable current price contract:
  `analytics_ai.v_agent_price_context_latest`

The files intentionally use `{{target_schema}}` as a placeholder. Replace it
with the final schema name before applying the SQL.

For an atomic preflight/deploy, use the repository runner:

```bash
./.venv/bin/python scripts/apply_agent_marts.py --schema agent_marts --check
./.venv/bin/python scripts/apply_agent_marts.py --schema agent_marts --apply
```

When `DECISION_SIMULATION_MODE` is `shadow` or `enforced`, the agent worker
runs the same transactional deployment during startup under a PostgreSQL
advisory lock. Mode `off` keeps the legacy candidate query and does not require
these marts.

## Apply order

1. `01_add_ad_attribution_columns.sql` (once per analytics database)
2. `00_create_agent_mart_schema.sql`
3. `sales_mart.sql`
4. `supply_mart.sql`
5. `marketing_mart.sql`
6. `finance_mart.sql`
7. `ad_campaign_sku_daily_mart.sql`
8. `price_sku_daily_mart.sql`
9. `price_context_latest.sql`

Example for local smoke testing:

```bash
for file in \
  sql/agent_marts/00_create_agent_mart_schema.sql \
  sql/agent_marts/sales_mart.sql \
  sql/agent_marts/supply_mart.sql \
  sql/agent_marts/marketing_mart.sql \
  sql/agent_marts/finance_mart.sql \
  sql/agent_marts/ad_campaign_sku_daily_mart.sql; do
  sed 's/{{target_schema}}/agent_marts_tmp/g' "$file" | psql -v ON_ERROR_STOP=1
done
```

Before deploying the views, apply the analytics compatibility migration:

```bash
psql -v ON_ERROR_STOP=1 \
  -f sql/agent_marts/01_add_ad_attribution_columns.sql
```

The advertising ETL must write WB fullstats `sum_price` into both raw tables.
It must preserve the distinction between `NULL` (the source metric was not
supplied) and `0` (the source explicitly reported zero attributed revenue).
For an upsert, `sum_price` must be included in the conflict update alongside
`sum_spend`, `orders`, and `shks`.

After all campaign and campaign-SKU rows for a day are committed, the ETL must
upsert `marketing.wb_marketing_campaign_stats_daily_watermarks` and set the
corresponding completeness flags to `true`. A failed or partial extraction must
leave them `false`. This watermark is the only condition under which an absent
statistics row is interpreted as proven zero activity; it also makes post-pause
SKU/day measurement rows observable.

## Source tables

- Sales: `sales.wb_supplier_sales`, `sales.wb_supplier_orders`,
  `sales.wb_sales_funnel_products_daily`
- Supply: `supply.wb_supplier_stocks`, `sales.wb_supplier_sales`
- Marketing: `marketing.wb_marketing_campaign_stats_daily_nms`,
  `marketing.wb_marketing_campaign_nms`, `marketing.wb_marketing_campaigns`,
  `marketing.wb_search_query_details`,
  `marketing.wb_search_query_orders_text_daily`,
  `sales.wb_sales_funnel_products_daily`
- Finance: `finance.wb_report_detail_daily`, `finance.wb_product_coast`

## First-version constraints

- All marts are plain `create or replace view` statements.
- No existing tables or views are dropped.
- No materialized views are created.
- The SQL does not depend on helper views that may not exist in a fresh
  database, such as `supply.v_wb_supplier_stocks_latest_daily` or
  `marketing.v_agent_marketing_*`.
- Metrics that depend on missing product cost data remain `null` in the finance
  mart instead of being silently treated as zero.
- Attributed advertising revenue remains `null` until the ETL supplies WB
  fullstats `sum_price`; the decision mart exposes explicit completeness and
  campaign-to-SKU reconciliation flags.
- Direct WB attribution is not presented as causal lift. The advertising
  decision mart exposes `causal_valid = false`; causal validation belongs to
  the experiment and measurement layer.
- `sku_contribution_after_ads_rub` subtracts the complete sum of advertising
  spend across all campaigns for that SKU/day exactly once. Consumers must
  require `sku_contribution_after_ads_complete = true` before using it as a
  measurement outcome.
- The advertising decision date spine includes finance/stock dates and complete
  advertising watermark dates through 35 days after the last observed
  campaign/SKU membership. Without a complete ad watermark, post-pause rows are
  retained but `sku_contribution_after_ads_complete` remains `false`.
- COGS is strictly as-of: a cost upload after `report_date` is never used for a
  historical row. Missing historical COGS keeps finance/economics incomplete.
- `stock_snapshot_present` and `supplier_snapshot_present` report row presence
  only. Until their ETLs publish explicit full-snapshot watermarks,
  `stock_snapshot_complete` and `supplier_snapshot_complete` remain `false`
  and `decision_input_complete` cannot become true.
- Decision candidates must read stock and inbound zeroes only from dedicated
  completed full-snapshot runs. Until those ETLs publish run IDs, completion
  timestamps and hashes, the simulation source deliberately returns
  `stock_snapshot_complete = false` and `supplier_snapshot_complete = false`.
- Post-action measurement also remains fail-closed while
  `finance_snapshot_complete = false`; row-level finance presence is not proof
  that a whole reporting day finished loading.
- Advertising fullstats and the replacement warehouse-remains report should be
  collected with a Personal or Service WB token. A deferred/rate-limited run
  must retain its cursor and must not publish a completion watermark.
- Supply coverage, reorder, out-of-stock, overstock, and warehouse concentration
  use sellable stock from `supply.wb_supplier_stocks.quantity` (`stock_qty`).
  `quantity_full` is kept as `stock_qty_full` for reference only because it can
  include goods already in transit to clients.
