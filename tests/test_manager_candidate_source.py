from __future__ import annotations

from agi_ecom_app.manager_decision import candidate_source


def _compact(sql: str) -> str:
    return " ".join(sql.split()).lower()


def test_simulation_snapshot_aggregates_normalized_campaign_sku_mart():
    sql = _compact(candidate_source._CAMPAIGN_SNAPSHOT_SQL)

    assert "from agent_marts.v_agent_ad_campaign_sku_daily mart" in sql
    assert "mart.ad_spend_rub as spend_rub" in sql
    assert "mart.ad_orders as orders" in sql
    assert "mart.ad_units as units" in sql
    assert (
        "mart.attributed_order_revenue_rub as attributed_revenue_rub" in sql
    )
    assert "group by advert_id, nm_id" in sql
    assert "st.advert_id = c.advert_id and st.nm_id = n.nm_id" in sql
    assert sql.count("marketing.wb_marketing_campaign_stats_daily_nms") == 1
    assert "from agent_marts.v_agent_ad_campaign_sku_daily mart" in sql


def test_simulation_snapshot_ends_at_latest_fully_completed_watermark_day():
    sql = _compact(candidate_source._CAMPAIGN_SNAPSHOT_SQL)

    assert "latest_complete_stats_day as" in sql
    assert (
        "from marketing.wb_marketing_campaign_stats_daily_watermarks" in sql
    )
    assert "campaign_stats_complete is true" in sql
    assert "campaign_sku_stats_complete is true" in sql
    assert "stat_date <= (current_date - interval '1 day')::date" in sql
    assert (
        "mart.report_date between (select report_date - interval '6 day' "
        "from latest_complete_stats_day) and (select report_date from "
        "latest_complete_stats_day)"
    ) in sql
    # A newer raw partial day cannot move the window: the public simulation
    # query intentionally never reads max(stat_date) from either raw stats table.
    assert "latest_stats_day" not in sql


def test_simulation_snapshot_keeps_proven_zero_complete_day_absence():
    sql = _compact(candidate_source._CAMPAIGN_SNAPSHOT_SQL)

    assert "mart.stats_evidence_origin" in sql
    assert "stats_evidence_origin = 'complete_day_absence'" in sql
    assert "jsonb_agg(round(spend_rub, 2) order by stat_date)" in sql
    assert "st.complete_day_absence_count" in sql
    assert "where stats_row_present" not in sql


def test_simulation_snapshot_requires_daily_campaign_sku_reconciliation():
    sql = _compact(candidate_source._CAMPAIGN_SNAPSHOT_SQL)

    assert (
        "bool_and(campaign_sku_attribution_reconciled is true) as "
        "campaign_sku_attribution_reconciled"
    ) in sql
    assert "and campaign_sku_attribution_reconciled ) as stats_window_complete" in sql
    assert "st.campaign_sku_attribution_reconciled" in sql
    assert "st.stats_window_complete is true" in sql
    assert "round(spend_rub, 2) as observed_ad_spend_7d_rub" in sql
    assert "st.observed_ad_spend_7d_rub" in sql


def test_partial_raw_stats_are_visible_only_as_diagnostic_spend():
    sql = _compact(candidate_source._CAMPAIGN_SNAPSHOT_SQL)

    assert "diagnostic_stats_7d as" in sql
    assert "from marketing.wb_marketing_campaign_stats_daily_nms raw" in sql
    assert "raw rows are used only" in sql
    assert (
        "coalesce(st.observed_ad_spend_7d_rub, "
        "diagnostic.observed_ad_spend_7d_rub)" in sql
    )
    assert "left join diagnostic_stats_7d diagnostic" in sql


def test_simulation_snapshot_does_not_treat_raw_stock_or_supply_presence_as_complete():
    sql = _compact(candidate_source._CAMPAIGN_SNAPSHOT_SQL)

    assert "stock.stock_snapshot_date" in sql
    assert "stock.stock_on_hand" in sql
    assert "false::boolean as stock_snapshot_complete" in sql
    assert "supply_day.snapshot_date as supplier_audit_snapshot_date" in sql
    assert "coalesce(inbound.supplier_inbound_qty, 0::numeric)" in sql
    assert "false::boolean as supplier_snapshot_complete" in sql


def test_legacy_snapshot_retains_campaign_level_backward_compatibility():
    sql = _compact(candidate_source._LEGACY_CAMPAIGN_SNAPSHOT_SQL)

    assert "from marketing.wb_marketing_campaign_stats_daily" in sql
    assert "group by stat_date, advert_id" in sql
    assert "left join stats_7d st on st.advert_id = c.advert_id" in sql
    assert "agent_marts.v_agent_ad_campaign_sku_daily" not in sql
    assert "campaign_stats_daily_watermarks" not in sql
    assert "sum_price" not in sql
    assert "stats_window_complete" not in sql
    assert "stock_snapshot_complete" not in sql
    assert "supplier_snapshot_complete" not in sql
    assert "derived:zero_attributed_orders" in sql


def test_latest_complete_supply_snapshot_proves_zero_when_no_inbound_row_exists():
    for snapshot_sql in (
        candidate_source._CAMPAIGN_SNAPSHOT_SQL,
        candidate_source._LEGACY_CAMPAIGN_SNAPSHOT_SQL,
    ):
        sql = _compact(snapshot_sql)

        assert "left join latest_supply_day supply_day on true" in sql
        assert "coalesce(inbound.supplier_inbound_qty, 0::numeric)" in sql
        assert "supply_day.snapshot_date as supplier_audit_snapshot_date" in sql
        assert "end as supplier_inbound_audited_qty" in sql


class _Cursor:
    def __init__(self, executed: list[tuple[str, list[int]]]):
        self.executed = executed

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def execute(self, sql: str, params: list[int]):
        self.executed.append((sql, params))

    def fetchall(self):
        return []


class _Connection:
    def __init__(self, executed: list[tuple[str, list[int]]]):
        self.executed = executed

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def cursor(self):
        return _Cursor(self.executed)


def test_loader_selects_legacy_sql_when_feature_is_off(monkeypatch):
    executed: list[tuple[str, list[int]]] = []
    supply_calls: list[dict[str, int]] = []
    monkeypatch.setenv("DECISION_SIMULATION_MODE", "off")
    monkeypatch.setattr(
        candidate_source,
        "get_connection",
        lambda: _Connection(executed),
    )
    monkeypatch.setattr(
        candidate_source,
        "get_supply_overview",
        lambda **kwargs: (
            supply_calls.append(kwargs) or {"critical_reorder_rows": []}
        ),
    )

    snapshot = candidate_source.load_manager_candidate_snapshot(limit=12)

    assert executed == [(candidate_source._LEGACY_CAMPAIGN_SNAPSHOT_SQL, [12])]
    assert supply_calls == [{"limit": 100}]
    assert "marketing.wb_marketing_campaign_stats_daily" in snapshot["sourceTables"]
    assert "agent_marts.v_agent_ad_campaign_sku_daily" not in snapshot["sourceTables"]


def test_loader_selects_mart_sql_for_shadow_and_enforced(monkeypatch):
    for mode in ("shadow", "enforced"):
        executed: list[tuple[str, list[int]]] = []
        monkeypatch.setenv("DECISION_SIMULATION_MODE", mode)
        monkeypatch.setattr(
            candidate_source,
            "get_connection",
            lambda: _Connection(executed),
        )
        monkeypatch.setattr(
            candidate_source,
            "get_supply_overview",
            lambda **_kwargs: {"critical_reorder_rows": []},
        )

        snapshot = candidate_source.load_manager_candidate_snapshot(limit=12)

        assert executed == [(candidate_source._CAMPAIGN_SNAPSHOT_SQL, [12])]
        assert "agent_marts.v_agent_ad_campaign_sku_daily" in snapshot["sourceTables"]
        assert (
            "marketing.wb_marketing_campaign_stats_daily_watermarks"
            in snapshot["sourceTables"]
        )
