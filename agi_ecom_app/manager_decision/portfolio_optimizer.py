from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta
from math import ceil, isfinite
from typing import Any, Mapping, Sequence


PORTFOLIO_OPTIMIZER_SCHEMA_VERSION = "portfolio-optimizer-v1"
MAX_STOCKOUT_PROBABILITY = 0.05


def optimize_portfolio(
    *,
    query_plan: Mapping[str, Any] | None,
    price_simulations: Sequence[Mapping[str, Any]] | None,
    inventory_rows: Sequence[Mapping[str, Any]] | None,
    candidate_snapshot: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Rank numeric price, advertising and supply alternatives on one scale.

    The optimizer never authorizes execution. It produces a shadow option set
    with comparable inventory, contribution-profit and stockout metrics.
    """

    plan = dict(query_plan or {})
    constraints = dict(plan.get("constraints") or {})
    horizon_days = max(1, min(90, int(constraints.get("horizon_days") or 30)))
    target_cover_days = max(0.0, float(constraints.get("target_cover_days") or 30))
    avoid_stockout = constraints.get("avoid_stockout") is not False
    preserve_profit = constraints.get("preserve_contribution_profit") is not False
    inventory = [dict(row) for row in list(inventory_rows or []) if isinstance(row, Mapping)]
    inventory_by_vendor, vendor_by_nm = _inventory_indexes(inventory)
    action_families = allowed_action_families(plan)

    scenarios = [
        *(
            _price_scenarios(
                price_simulations or [],
                inventory_by_vendor=inventory_by_vendor,
                horizon_days=horizon_days,
                target_cover_days=target_cover_days,
                avoid_stockout=avoid_stockout,
                preserve_profit=preserve_profit,
            )
            if "price" in action_families
            else []
        ),
        *(
            _marketing_scenarios(
                candidate_snapshot or {},
                inventory_by_vendor=inventory_by_vendor,
                vendor_by_nm=vendor_by_nm,
                horizon_days=horizon_days,
                avoid_stockout=avoid_stockout,
                preserve_profit=preserve_profit,
            )
            if "marketing" in action_families
            else []
        ),
        *(
            _supply_scenarios(
                candidate_snapshot or {},
                inventory_by_vendor=inventory_by_vendor,
                horizon_days=horizon_days,
                target_cover_days=target_cover_days,
            )
            if "supply" in action_families
            else []
        ),
    ]
    has_coverage_progress = any(
        item.get("eligible") is True
        and (_number(item.get("inventoryDaysReduction")) or 0) > 0
        for item in scenarios
    )
    if not has_coverage_progress:
        for item in scenarios:
            if item.get("eligible") is not True:
                continue
            if (_number(item.get("inventoryDaysReduction")) or 0) > 0:
                continue
            item["eligible"] = False
            item.setdefault("constraintReasons", []).append(
                "действие не снижает складское покрытие и не продвигает цель портфеля"
            )
    _assign_scores(scenarios, target_cover_days=target_cover_days)
    eligible = [item for item in scenarios if item.get("eligible") is True]
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in sorted(
        eligible,
        key=lambda value: (
            -float(value.get("businessScore") or 0),
            -float(value.get("expectedContributionProfitRub") or 0),
            str(value.get("scenarioId") or ""),
        ),
    ):
        key = (str(item.get("family") or ""), str(item.get("target") or ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    top = deduped[:5]
    hypothesis_eligible = [
        item for item in scenarios if item.get("hypothesisEligible") is True
    ]
    hypothesis_deduped: list[dict[str, Any]] = []
    hypothesis_seen: set[tuple[str, str]] = set()
    for item in sorted(
        hypothesis_eligible,
        key=lambda value: (
            -_confidence_adjusted_hypothesis_score(value),
            -float(value.get("expectedContributionProfitRub") or 0),
            str(value.get("scenarioId") or ""),
        ),
    ):
        key = (str(item.get("family") or ""), str(item.get("target") or ""))
        if key in hypothesis_seen:
            continue
        hypothesis_seen.add(key)
        hypothesis_deduped.append(item)

    top_hypotheses = hypothesis_deduped[:5]
    priority_recommendations = _build_priority_recommendations(
        plan=plan,
        eligible_recommendations=top,
        top_hypotheses=top_hypotheses,
        inventory_rows=inventory,
    )
    goal_attainment = _goal_attainment(
        inventory,
        recommendations=top,
        target_cover_days=target_cover_days,
    )
    hypothesis_goal_attainment = _goal_attainment(
        inventory,
        recommendations=top_hypotheses,
        target_cover_days=target_cover_days,
    )
    return {
        "schemaVersion": PORTFOLIO_OPTIMIZER_SCHEMA_VERSION,
        "mode": "shadow",
        "objective": (
            "0.60*turnover_progress + 0.40*normalized_positive_contribution_profit "
            "- 0.50*stockout_probability"
        ),
        "horizonDays": horizon_days,
        "targetCoverDays": target_cover_days,
        "constraints": {
            "avoidStockout": avoid_stockout,
            "preserveContributionProfit": preserve_profit,
            "maximumStockoutProbability": MAX_STOCKOUT_PROBABILITY,
        },
        "scenarioCount": len(scenarios),
        "eligibleScenarioCount": len(eligible),
        "topRecommendations": top,
        "topHypotheses": top_hypotheses,
        "priorityRecommendations": priority_recommendations,
        "scenarios": scenarios,
        "goalAttainment": goal_attainment,
        "hypothesisGoalAttainment": hypothesis_goal_attainment,
        "complete": len(top) == 5,
        "violations": validate_portfolio_optimization(top),
        "priorityComplete": (
            len(priority_recommendations) == 5
            if plan.get("decision_kind") == "action_portfolio"
            else True
        ),
        "priorityViolations": validate_priority_recommendations(
            priority_recommendations,
            required_families=_requested_priority_families(plan),
            require_exact_five=plan.get("decision_kind") == "action_portfolio",
        ),
    }


def validate_portfolio_optimization(
    recommendations: Sequence[Mapping[str, Any]] | None,
) -> list[str]:
    rows = list(recommendations or [])
    violations: list[str] = []
    if len(rows) != 5:
        violations.append(f"expected 5 optimized recommendations, got {len(rows)}")
    required_numeric = (
        "expectedContributionProfitRub",
        "p10ContributionProfitRub",
        "stockoutProbability",
        "businessScore",
    )
    for index, row in enumerate(rows, start=1):
        if row.get("eligible") is not True:
            violations.append(f"recommendation {index} is not eligible")
        for field in required_numeric:
            value = _number(row.get(field))
            if value is None:
                violations.append(f"recommendation {index} misses {field}")
        if not str(row.get("action") or "").strip():
            violations.append(f"recommendation {index} misses action")
    return violations


def validate_priority_recommendations(
    recommendations: Sequence[Mapping[str, Any]] | None,
    *,
    required_families: Sequence[str] = (),
    require_exact_five: bool = False,
) -> list[str]:
    """Validate the manager-facing list, where blocked rows are intentional.

    Executable recommendations remain governed by ``validate_portfolio_optimization``.
    This second contract guarantees a complete decision/checklist without converting
    missing evidence into a claimed effect.
    """

    rows = list(recommendations or [])
    violations: list[str] = []
    if require_exact_five and len(rows) != 5:
        violations.append(f"expected 5 priority recommendations, got {len(rows)}")
    present_families = {str(row.get("family") or "") for row in rows}
    for family in required_families:
        if family not in present_families:
            violations.append(f"missing {family} priority recommendation")
    for index, row in enumerate(rows, start=1):
        if not str(row.get("action") or "").strip():
            violations.append(f"priority recommendation {index} misses action")
        if row.get("eligible") is not True and not list(
            row.get("constraintReasons") or row.get("limitations") or []
        ):
            violations.append(
                f"blocked priority recommendation {index} misses limitation"
            )
    return violations


def allowed_action_families(plan: Mapping[str, Any] | None) -> list[str]:
    """Return output action families without confusing them with evidence domains.

    New query plans carry an explicit semantic contract. Older persisted plans and
    unit fixtures keep the legacy inference so stored runs remain renderable.
    """

    source = dict(plan or {})
    aliases = {
        "price": "price",
        "pricing": "price",
        "marketing": "marketing",
        "advertising": "marketing",
        "supply": "supply",
    }
    if "allowed_action_families" in source:
        return list(
            dict.fromkeys(
                aliases[str(item).strip().lower()]
                for item in list(source.get("allowed_action_families") or [])
                if str(item).strip().lower() in aliases
            )
        )
    inferred = _legacy_requested_priority_families(source)
    return inferred or ["price", "marketing", "supply"]


def _legacy_requested_priority_families(plan: Mapping[str, Any]) -> list[str]:
    analyses = {str(item) for item in list(plan.get("requested_analyses") or [])}
    domains = {
        str(getattr(item, "value", item))
        for item in list(plan.get("required_domains") or [])
    }
    families: list[str] = []
    if "price" in domains or analyses.intersection({"pricing", "price_scenarios"}):
        families.append("price")
    if "marketing" in domains or analyses.intersection({"advertising", "conversions"}):
        families.append("marketing")
    if "supply" in domains or analyses.intersection(
        {"replenishment", "inventory_turnover", "current_stock", "inventory_matrix"}
    ):
        families.append("supply")
    return families


def _requested_priority_families(plan: Mapping[str, Any]) -> list[str]:
    return allowed_action_families(plan)


def _priority_sort_key(item: Mapping[str, Any]) -> tuple[int, float, float, str]:
    return (
        0 if item.get("eligible") is True else 1,
        -float(_number(item.get("businessScore")) or 0.0),
        -max(0.0, _number(item.get("inventoryDaysReduction")) or 0.0),
        str(item.get("scenarioId") or ""),
    )


def _build_priority_recommendations(
    *,
    plan: Mapping[str, Any],
    eligible_recommendations: Sequence[Mapping[str, Any]],
    top_hypotheses: Sequence[Mapping[str, Any]],
    inventory_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build five manager-facing rows without hiding numeric hypotheses.

    Strictly eligible actions always come first. Numeric hypotheses fill the next
    slots and remain non-executable. Verification placeholders are used only when
    fewer than five calculated rows are available.
    """

    require_five = plan.get("decision_kind") == "action_portfolio"
    calculated_count = len(list(eligible_recommendations or [])) + len(
        list(top_hypotheses or [])
    )
    limit = 5 if require_five else min(5, calculated_count)
    if limit <= 0:
        return []

    eligible_rows = [
        dict(item)
        for item in eligible_recommendations
        if isinstance(item, Mapping)
    ]
    hypothesis_rows = [
        dict(item)
        for item in top_hypotheses
        if isinstance(item, Mapping)
    ]
    required_families = _requested_priority_families(plan)
    verification_rows = (
        _blocked_verification_rows(
            plan=plan,
            inventory_rows=inventory_rows,
            required_families=required_families,
        )
        if require_five
        else []
    )
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    seen_targets: set[tuple[str, str]] = set()

    def add(item: Mapping[str, Any]) -> None:
        if len(selected) >= limit:
            return
        row = dict(item)
        scenario_id = str(row.get("scenarioId") or "").strip()
        if not scenario_id:
            scenario_id = (
                f"priority:{row.get('family') or 'unknown'}:"
                f"{row.get('target') or 'portfolio'}:{len(selected) + 1}"
            )
            row["scenarioId"] = scenario_id
        if scenario_id in seen:
            return
        target_key = (
            str(row.get("family") or ""),
            str(row.get("target") or ""),
        )
        if target_key in seen_targets:
            return
        seen.add(scenario_id)
        seen_targets.add(target_key)
        selected.append(row)

    calculated_rows = [*eligible_rows, *hypothesis_rows]
    if require_five:
        # Required decision families must remain visible even when the calculated
        # top set is dominated by one family (for example, five price hypotheses).
        # Prefer a strict eligible calculation, then a ranked hypothesis, and use
        # a verification row only when the family has no calculated alternative.
        for family in required_families:
            family_calculated = [
                item
                for item in calculated_rows
                if str(item.get("family") or "") == family
            ]
            if family_calculated:
                add(family_calculated[0])
                continue
            family_verification = next(
                (
                    item
                    for item in verification_rows
                    if str(item.get("family") or "") == family
                ),
                None,
            )
            if family_verification is not None:
                add(family_verification)

    for item in eligible_rows:
        add(item)
    for item in hypothesis_rows:
        add(item)

    if require_five and len(selected) < limit:
        # Preserve requested domain coverage when placeholders are needed, then
        # fill any remaining slots with target-specific verification work.
        ordered_verification: list[dict[str, Any]] = []
        for family in required_families:
            family_row = next(
                (
                    item
                    for item in verification_rows
                    if str(item.get("family") or "") == family
                ),
                None,
            )
            if family_row is not None:
                ordered_verification.append(family_row)
        ordered_verification.extend(verification_rows)
        for item in ordered_verification:
            add(item)
            if len(selected) == limit:
                break

    return selected[:limit]


def _blocked_verification_rows(
    *,
    plan: Mapping[str, Any],
    inventory_rows: Sequence[Mapping[str, Any]],
    required_families: Sequence[str],
) -> list[dict[str, Any]]:
    focus_targets = [
        str(item.get("value") or "").strip()
        for item in list(plan.get("targets") or [])
        if isinstance(item, Mapping) and str(item.get("value") or "").strip()
    ]
    ranked_inventory = sorted(
        [dict(item) for item in inventory_rows if isinstance(item, Mapping)],
        key=lambda item: (
            -float(
                _coverage_days(
                    _number(item.get("current_stock_qty")),
                    _number(item.get("sales_7d_units")),
                    7,
                )
                or 0.0
            ),
            str(item.get("vendor_code") or item.get("nm_id") or ""),
        ),
    )
    targets = list(
        dict.fromkeys(
            [
                *focus_targets,
                *[
                    str(item.get("vendor_code") or item.get("nm_id") or "").strip()
                    for item in ranked_inventory
                ],
                "портфель",
            ]
        )
    )
    targets = [item for item in targets if item]
    inventory_by_target = {
        str(item.get("vendor_code") or item.get("nm_id") or "").strip(): item
        for item in ranked_inventory
    }
    families = list(required_families) or ["price", "marketing", "supply"]
    actions = {
        "price": (
            "сохранить текущую цену и собрать свежую историю цена–спрос, "
            "маржу и сопоставимые цены конкурентов"
        ),
        "marketing": (
            "сохранить текущий рекламный режим и сверить расходы, трафик, "
            "органическую позицию и конверсии в одном окне"
        ),
        "supply": (
            "не создавать поставку до подтверждения ETA, lead time, MOQ и "
            "актуального доступного остатка"
        ),
    }
    reasons = {
        "price": "ценовой эффект и безопасный конкурентный коридор не подтверждены",
        "marketing": "причинный эффект рекламы и органики не подтверждён",
        "supply": "срок и параметры будущей поставки не подтверждены",
    }
    rows: list[dict[str, Any]] = []
    for target in targets:
        inventory = inventory_by_target.get(target) or {}
        current_cover = _coverage_days(
            _number(inventory.get("current_stock_qty")),
            _number(inventory.get("sales_7d_units")),
            7,
        )
        for family in families:
            reason = reasons.get(family, "эффект действия не подтверждён")
            prefix = "По портфелю" if target == "портфель" else f"SKU {target}"
            rows.append(
                {
                    "scenarioId": f"verification:{family}:{target}",
                    "family": family,
                    "target": target,
                    "action": f"{prefix}: {actions.get(family, 'собрать недостающие данные')}",
                    "calculationStatus": "needs_data",
                    "executionStatus": "blocked",
                    "horizonDays": int(
                        (plan.get("constraints") or {}).get("horizon_days") or 30
                    ),
                    "inventoryMetric": "current_cover_days_7d",
                    "inventoryDaysBefore": _rounded(current_cover),
                    "inventoryDaysAfter": None,
                    "inventoryDaysReduction": None,
                    "expectedContributionProfitRub": None,
                    "p10ContributionProfitRub": None,
                    "stockoutProbability": None,
                    "businessScore": 0.0,
                    "eligible": False,
                    "constraintReasons": [reason],
                    "evidenceMode": "verification checklist; effect not estimated",
                    "limitations": [reason],
                }
            )
    return rows


def _inventory_indexes(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    by_vendor: dict[str, dict[str, Any]] = {}
    vendor_by_nm: dict[str, str] = {}
    for raw in rows:
        row = dict(raw)
        vendor = str(row.get("vendor_code") or "").strip()
        nm_id = str(row.get("nm_id") or "").strip()
        if vendor:
            by_vendor[vendor] = row
        if vendor and nm_id:
            vendor_by_nm[nm_id] = vendor
    return by_vendor, vendor_by_nm


def _price_scenarios(
    simulations: Sequence[Mapping[str, Any]],
    *,
    inventory_by_vendor: Mapping[str, Mapping[str, Any]],
    horizon_days: int,
    target_cover_days: float,
    avoid_stockout: bool,
    preserve_profit: bool,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for raw in simulations:
        simulation = dict(raw)
        target = str(simulation.get("vendorCode") or simulation.get("target") or "").strip()
        simulation_horizon = max(1, int(simulation.get("horizonDays") or horizon_days))
        completed_candidates = [
            dict(item)
            for item in list(simulation.get("candidates") or [])
            if isinstance(item, Mapping) and item.get("status") == "completed"
        ]
        raw_hypotheses = simulation.get("hypothesisCandidates")
        hypothesis_metadata = {
            "status": simulation.get("hypothesisStatus"),
            "executionEligible": simulation.get("hypothesisExecutionEligible"),
            "calculationMode": simulation.get("hypothesisCalculationMode"),
            "confidence": simulation.get("hypothesisConfidence"),
            "confidenceScore": simulation.get("hypothesisConfidenceScore"),
            "assumptions": simulation.get("hypothesisAssumptions"),
            "evidenceGaps": simulation.get("hypothesisEvidenceGaps"),
        }
        if isinstance(raw_hypotheses, Mapping):
            hypothesis_metadata.update(dict(raw_hypotheses))
        raw_hypothesis_candidates = (
            hypothesis_metadata.get("candidates")
            if isinstance(raw_hypotheses, Mapping)
            else raw_hypotheses
        )
        hypothesis_candidates = [
            dict(item)
            for item in list(raw_hypothesis_candidates or [])
            if isinstance(item, Mapping)
        ]
        merged_candidates: list[tuple[dict[str, Any], bool]] = []
        seen_candidates: set[tuple[str, str]] = set()
        for candidate, is_hypothesis in [
            *((item, False) for item in completed_candidates),
            *((item, True) for item in hypothesis_candidates),
        ]:
            price = _number(candidate.get("priceRub"))
            key = (
                str(candidate.get("operation") or "").strip(),
                f"{price:.9g}" if price is not None else str(candidate.get("candidateId") or ""),
            )
            if key in seen_candidates:
                continue
            seen_candidates.add(key)
            merged_candidates.append((candidate, is_hypothesis))
        hold_entry = next(
            (
                item
                for item in merged_candidates
                if item[0].get("operation") == "hold_price"
            ),
            None,
        )
        if hold_entry is None:
            continue
        hold, _hold_is_hypothesis = hold_entry
        hold_units = _first_number(hold, "expectedUnits")
        current_stock = _number(simulation.get("availableStockUnits"))
        included_inbound = sum(
            _number(item.get("units")) or 0.0
            for item in list(simulation.get("futureInboundEvents") or [])
            if isinstance(item, Mapping)
            and item.get("includeInCapacity") is True
            and item.get("status") == "scheduled"
        )
        if current_stock is not None:
            current_stock += included_inbound
        inventory_row = inventory_by_vendor.get(target) or {}
        baseline_days = _coverage_days(current_stock, hold_units, simulation_horizon)
        if baseline_days is None:
            baseline_days = _number(inventory_row.get("days_on_hand"))
        current_price = _number(simulation.get("currentPriceRub"))
        for candidate, hypothesis_mode in merged_candidates:
            if candidate.get("operation") == "hold_price":
                continue
            candidate_units = _first_number(candidate, "expectedUnits")
            candidate_days = _coverage_days(current_stock, candidate_units, simulation_horizon)
            mean_profit = _first_number(candidate, "meanEffectRub", "mean")
            p10_profit = _first_number(candidate, "p10EffectRub", "p10")
            stockout = _first_number(
                candidate,
                "stockoutProbability",
                "stockout",
            )
            probability_positive = _first_number(
                candidate,
                "probabilityPositive",
                "probPositive",
            )
            price = _first_number(candidate, "priceRub")
            if None in (mean_profit, p10_profit, stockout, price):
                continue
            days_reduction = _days_reduction(baseline_days, candidate_days)
            reasons: list[str] = []
            if preserve_profit and (mean_profit < 0 or p10_profit < 0):
                reasons.append("нижняя граница contribution profit отрицательна")
            if avoid_stockout and stockout > MAX_STOCKOUT_PROBABILITY:
                reasons.append("риск дефицита выше 5%")
            if baseline_days is not None and baseline_days <= target_cover_days and days_reduction > 0:
                reasons.append("снижение покрытия не требуется для достижения цели")
            move_pct = _number(candidate.get("priceMovePct"))
            if (
                move_pct is None
                and current_price is not None
                and current_price > 0
            ):
                move_pct = (price - current_price) * 100 / current_price
            competitor_guardrail = _competitor_guardrail(
                simulation,
                current_price=current_price,
                candidate_price=price,
            )
            competitor_status = str(competitor_guardrail.get("status") or "missing")
            competitor_is_direct = competitor_status == "applied"
            if hypothesis_mode:
                # Missing, stale or fallback competitor evidence changes how much
                # we trust a hypothesis, not whether it may be tested in shadow.
                # A violation against a fresh comparable corridor remains a hard
                # constraint.
                if competitor_is_direct and competitor_guardrail.get("violation"):
                    reasons.append(str(competitor_guardrail["violation"]))
            else:
                if not competitor_status.startswith("applied"):
                    reasons.append(
                        "нет качественного актуального конкурентного коридора для изменения цены"
                    )
                elif competitor_guardrail.get("violation"):
                    reasons.append(str(competitor_guardrail["violation"]))

            assumptions = _text_items(
                candidate.get("assumptions"),
                (
                    hypothesis_metadata.get("assumptions")
                    or simulation.get("hypothesisAssumptions")
                    if hypothesis_mode
                    else None
                ),
            )
            evidence_gaps = _text_items(
                candidate.get("evidenceGaps"),
                (
                    hypothesis_metadata.get("evidenceGaps")
                    or simulation.get("hypothesisEvidenceGaps")
                    if hypothesis_mode
                    else None
                ),
            )
            limitations = _text_items(
                candidate.get("limitations"),
                hypothesis_metadata.get("limitations") if hypothesis_mode else None,
            )
            confidence = _confidence_level(
                candidate.get("confidence")
                or hypothesis_metadata.get("confidence")
                or simulation.get("hypothesisConfidence")
                or ("medium" if hypothesis_mode else None)
            )
            if hypothesis_mode:
                assumptions = _unique_text(
                    [
                        *assumptions,
                        "числовой отклик спроса используется как проверяемая гипотеза, а не как доказанный причинный эффект",
                    ]
                )
                evidence_gaps = _unique_text(
                    [
                        *evidence_gaps,
                        "строгая причинная проверка ценового эффекта не пройдена",
                    ]
                )
                confidence = _lower_confidence(confidence)
                limitations = _unique_text(
                    [
                        *limitations,
                        "гипотеза предназначена только для теневого теста и не разрешает изменение цены",
                    ]
                )
                if not competitor_is_direct:
                    confidence = _lower_confidence(confidence)
                    evidence_gaps = _unique_text(
                        [
                            *evidence_gaps,
                            "нет свежего сопоставимого конкурентного коридора",
                        ]
                    )
                    limitations = _unique_text(
                        [
                            *limitations,
                            "конкурентные данные отсутствуют, устарели или имеют недостаточное покрытие; уверенность снижена",
                        ]
                    )
            limitations = _unique_text(
                [
                    *limitations,
                    (
                        "теневой расчёт; причинный эффект цены не подтверждён экспериментом"
                        if not hypothesis_mode
                        else ""
                    ),
                    *list(competitor_guardrail.get("limitations") or []),
                ]
            )
            result.append(
                {
                    "scenarioId": f"price:{target}:{price:g}",
                    "family": "price",
                    "target": target,
                    "action": (
                        f"SKU {target}: изменить цену с {_money(current_price)} до {_money(price)}"
                        + (f" ({move_pct:+.1f}%)" if move_pct is not None else "")
                    ),
                    "calculationStatus": (
                        "hypothesis" if hypothesis_mode else "completed"
                    ),
                    "executionStatus": (
                        "shadow_test" if hypothesis_mode else "shadow"
                    ),
                    "horizonDays": simulation_horizon,
                    "currentPriceRub": _rounded(current_price),
                    "candidatePriceRub": _rounded(price),
                    "priceMovePct": _rounded(move_pct),
                    "inventoryMetric": "simulated_forward_cover_days",
                    "inventoryDaysBefore": _rounded(baseline_days),
                    "inventoryDaysAfter": _rounded(candidate_days),
                    "inventoryDaysReduction": _rounded(days_reduction) or 0.0,
                    "expectedContributionProfitRub": round(mean_profit, 2),
                    "p10ContributionProfitRub": round(p10_profit, 2),
                    "stockoutProbability": round(stockout, 6),
                    "expectedUnits": _rounded(candidate_units),
                    "baselineExpectedUnits": _rounded(hold_units),
                    "expectedUnitsDelta": _rounded(
                        candidate_units - hold_units
                        if candidate_units is not None and hold_units is not None
                        else None
                    ),
                    "availableStockUnits": _rounded(current_stock),
                    "gapToTargetDays": _rounded(
                        max(candidate_days - target_cover_days, 0.0)
                        if candidate_days is not None
                        else None
                    ),
                    "probabilityPositive": (
                        round(probability_positive, 6)
                        if probability_positive is not None
                        else None
                    ),
                    "eligible": False if hypothesis_mode else not reasons,
                    "hypothesisEligible": hypothesis_mode and not reasons,
                    "constraintReasons": reasons,
                    "evidenceMode": (
                        "numeric price-demand hypothesis; strict causal gate not passed"
                        if hypothesis_mode
                        else "500-run price-demand bootstrap"
                    ),
                    "confidence": confidence,
                    "confidenceScore": (
                        _number(hypothesis_metadata.get("confidenceScore"))
                        if hypothesis_mode
                        else None
                    ),
                    "assumptions": assumptions,
                    "evidenceGaps": evidence_gaps,
                    "limitations": limitations,
                    "hypothesisCalculationMode": (
                        hypothesis_metadata.get("calculationMode")
                        if hypothesis_mode
                        else None
                    ),
                    "competitorGuardrail": competitor_guardrail,
                    "includedSupplierInboundUnits": round(included_inbound, 4),
                }
            )
    return result


def _text_items(*values: Any) -> list[str]:
    items: list[str] = []
    for value in values:
        if isinstance(value, str):
            candidates = [value]
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            candidates = list(value)
        else:
            candidates = []
        items.extend(
            text
            for item in candidates
            if (text := str(item).strip())
        )
    return _unique_text(items)


def _unique_text(items: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def _confidence_level(value: Any) -> str:
    normalized = str(value or "medium").strip().lower()
    return normalized if normalized in {"high", "medium", "low"} else "medium"


def _lower_confidence(value: Any) -> str:
    confidence = _confidence_level(value)
    return {"high": "medium", "medium": "low", "low": "low"}[confidence]


def _confidence_adjusted_hypothesis_score(item: Mapping[str, Any]) -> float:
    confidence_weight = {
        "high": 1.0,
        "medium": 0.7,
        "low": 0.4,
    }[_confidence_level(item.get("confidence"))]
    score = float(_number(item.get("businessScore")) or 0.0)
    # Subtract a fixed uncertainty penalty so low-confidence negative scores do
    # not accidentally outrank equally negative high-confidence hypotheses.
    return score - (1.0 - confidence_weight) * 100.0


def _competitor_guardrail(
    simulation: Mapping[str, Any],
    *,
    current_price: float | None,
    candidate_price: float | None,
) -> dict[str, Any]:
    context = simulation.get("currentMarketContext")
    if not isinstance(context, Mapping):
        competitor_fallback = _competitor_temporal_guardrail(
            simulation,
            current_price=current_price,
            candidate_price=candidate_price,
            fallback_reason="свежий сопоставимый конкурентный срез отсутствует",
        )
        if competitor_fallback is not None:
            return competitor_fallback
        return _temporal_price_guardrail(
            simulation,
            current_price=current_price,
            candidate_price=candidate_price,
            fallback_reason="свежий сопоставимый конкурентный срез отсутствует",
        )
    query_count = _number(context.get("query_count")) or 0.0
    found_share = _number(context.get("source_found_query_share")) or 0.0
    competitor_count = _number(context.get("organic_competitors_before_source")) or 0.0
    source_buyer_price = _number(context.get("source_price_effective"))
    competitor_median = _number(context.get("price_before_median"))
    if (
        None in (current_price, candidate_price, source_buyer_price, competitor_median)
        or current_price <= 0
        or candidate_price <= 0
        or source_buyer_price <= 0
        or competitor_median <= 0
        or not 0 <= found_share <= 1
    ):
        competitor_fallback = _competitor_temporal_guardrail(
            simulation,
            current_price=current_price,
            candidate_price=candidate_price,
            fallback_reason="конкурентный срез не содержит сопоставимой цены",
        )
        if competitor_fallback is not None:
            return competitor_fallback
        return _temporal_price_guardrail(
            simulation,
            current_price=current_price,
            candidate_price=candidate_price,
            fallback_reason="конкурентный срез не содержит сопоставимой цены",
        )
    basis_ratio = source_buyer_price / current_price if current_price else 0.0
    if (
        query_count < 3
        or found_share < 0.5
        or competitor_count < 5
        or not 0.75 <= basis_ratio <= 1.25
    ):
        fallback = _competitor_temporal_guardrail(
            simulation,
            current_price=current_price,
            candidate_price=candidate_price,
            fallback_reason="текущий конкурентный срез имеет низкое покрытие или несопоставимую ценовую базу",
        ) or _temporal_price_guardrail(
            simulation,
            current_price=current_price,
            candidate_price=candidate_price,
            fallback_reason="конкурентные цены исключены: недостаточная сопоставимость или покрытие",
        )
        fallback.update(
            {
                "queryCount": int(query_count),
                "sourceFoundQueryShare": round(found_share, 4),
                "competitorCount": int(competitor_count),
                "priceBasisRatio": round(basis_ratio, 4),
            }
        )
        return fallback
    projected_buyer_price = source_buyer_price * candidate_price / current_price
    gap_pct = (
        (projected_buyer_price - competitor_median) * 100 / competitor_median
        if competitor_median > 0
        else None
    )
    violation = (
        "прогнозная цена более чем на 20% выше медианы сопоставимых конкурентов"
        if gap_pct is not None and gap_pct > 20
        else None
    )
    return {
        "status": "applied",
        "competitorMedianBuyerPriceRub": round(competitor_median, 2),
        "projectedBuyerPriceRub": round(projected_buyer_price, 2),
        "projectedGapPct": round(gap_pct, 2) if gap_pct is not None else None,
        "violation": violation,
        "limitations": [],
    }


def _competitor_temporal_guardrail(
    simulation: Mapping[str, Any],
    *,
    current_price: float | None,
    candidate_price: float | None,
    fallback_reason: str,
) -> dict[str, Any] | None:
    if (
        current_price is None
        or current_price <= 0
        or candidate_price is None
        or candidate_price <= 0
    ):
        return None

    observations: list[tuple[str, float, float | None, float | None]] = []
    for raw in list(simulation.get("marketContextHistory") or []):
        if not isinstance(raw, Mapping):
            continue
        observed_date = str(raw.get("snapshot_date") or "").strip()
        median = _number(raw.get("market_price_median"))
        if observed_date and median is not None and median > 0:
            observations.append(
                (
                    observed_date,
                    median,
                    _number(raw.get("market_price_min")),
                    _number(raw.get("market_price_max")),
                )
            )
    observations.sort(key=lambda item: item[0], reverse=True)
    if not observations:
        return None

    today_date, today_median, today_min, today_max = observations[0]
    yesterday = observations[1] if len(observations) > 1 else None
    if yesterday is not None:
        yesterday_date, yesterday_median, yesterday_min, yesterday_max = yesterday
        day_change_pct = (today_median - yesterday_median) * 100 / yesterday_median
        allowed_move_pct = max(2.5, min(12.0, abs(day_change_pct) + 2.5))
        status = "applied_competitor_temporal_fallback"
        basis = "competitor_market_median_yesterday_today"
        confidence = "medium"
    else:
        yesterday_date = None
        yesterday_median = None
        yesterday_min = None
        yesterday_max = None
        day_change_pct = None
        allowed_move_pct = 2.5
        status = "applied_competitor_single_day_fallback"
        basis = "competitor_market_median_single_day"
        confidence = "low"

    corridor_low = current_price * (1 - allowed_move_pct / 100)
    corridor_high = current_price * (1 + allowed_move_pct / 100)
    rounding_tolerance = max(1.0, current_price * 0.001)
    violation = None
    if (
        candidate_price < corridor_low - rounding_tolerance
        or candidate_price > corridor_high + rounding_tolerance
    ):
        violation = (
            "кандидат выходит за конкурентный временной коридор "
            f"{corridor_low:.2f}–{corridor_high:.2f} ₽"
        )
    return {
        "status": status,
        "basis": basis,
        "confidence": confidence,
        "todaySnapshotDate": today_date,
        "yesterdaySnapshotDate": yesterday_date,
        "competitorTodayMedianRub": round(today_median, 2),
        "competitorYesterdayMedianRub": (
            round(yesterday_median, 2) if yesterday_median is not None else None
        ),
        "competitorYesterdayToTodayChangePct": (
            round(day_change_pct, 2) if day_change_pct is not None else None
        ),
        "competitorTodayMinRub": round(today_min, 2) if today_min is not None else None,
        "competitorTodayMaxRub": round(today_max, 2) if today_max is not None else None,
        "competitorYesterdayMinRub": (
            round(yesterday_min, 2) if yesterday_min is not None else None
        ),
        "competitorYesterdayMaxRub": (
            round(yesterday_max, 2) if yesterday_max is not None else None
        ),
        "allowedMovePct": round(allowed_move_pct, 2),
        "corridorLowerPriceRub": round(corridor_low, 2),
        "corridorUpperPriceRub": round(corridor_high, 2),
        "roundingToleranceRub": round(rounding_tolerance, 2),
        "projectedSellerPriceRub": round(candidate_price, 2),
        "violation": violation,
        "limitations": [
            fallback_reason,
            "использована динамика медианы конкурентов из БД; абсолютные min/max не задают коридор из-за шумных выбросов",
        ],
    }


def _temporal_price_guardrail(
    simulation: Mapping[str, Any],
    *,
    current_price: float | None,
    candidate_price: float | None,
    fallback_reason: str,
) -> dict[str, Any]:
    if current_price is None or current_price <= 0 or candidate_price is None or candidate_price <= 0:
        return {
            "status": "missing",
            "limitations": [fallback_reason, "текущая или кандидатная цена отсутствует"],
        }

    observations: list[tuple[str, float]] = []
    for raw in list(simulation.get("dailyPriceRows") or []):
        if not isinstance(raw, Mapping):
            continue
        observed_price = next(
            (
                value
                for value in (
                    _number(raw.get("sellerEffectivePriceRub")),
                    _number(raw.get("observedSellerPriceRub")),
                    _number(raw.get("basePriceRub")),
                )
                if value is not None and value > 0
            ),
            None,
        )
        observed_date = str(raw.get("date") or "").strip()
        if observed_price is not None and observed_date:
            observations.append((observed_date, observed_price))
    observations.sort(key=lambda item: item[0], reverse=True)

    today_observed = observations[0][1] if observations else current_price
    yesterday_observed = observations[1][1] if len(observations) > 1 else None
    if yesterday_observed is not None and yesterday_observed > 0:
        day_change_pct = (today_observed - yesterday_observed) * 100 / yesterday_observed
        allowed_move_pct = max(2.5, min(12.0, abs(day_change_pct) + 2.5))
        status = "applied_temporal_fallback"
        confidence = "medium"
        basis = "own_price_yesterday_today"
    else:
        day_change_pct = None
        allowed_move_pct = 2.5
        status = "applied_current_anchor_fallback"
        confidence = "low"
        basis = "current_price_only"

    corridor_low = current_price * (1 - allowed_move_pct / 100)
    corridor_high = current_price * (1 + allowed_move_pct / 100)
    rounding_tolerance = max(1.0, current_price * 0.001)
    violation = None
    if (
        candidate_price < corridor_low - rounding_tolerance
        or candidate_price > corridor_high + rounding_tolerance
    ):
        violation = (
            "кандидат выходит за временной ценовой коридор "
            f"{corridor_low:.2f}–{corridor_high:.2f} ₽"
        )
    return {
        "status": status,
        "basis": basis,
        "confidence": confidence,
        "todayPriceRub": round(today_observed, 2),
        "yesterdayPriceRub": round(yesterday_observed, 2) if yesterday_observed is not None else None,
        "yesterdayToTodayChangePct": round(day_change_pct, 2) if day_change_pct is not None else None,
        "allowedMovePct": round(allowed_move_pct, 2),
        "corridorLowerPriceRub": round(corridor_low, 2),
        "corridorUpperPriceRub": round(corridor_high, 2),
        "roundingToleranceRub": round(rounding_tolerance, 2),
        "projectedSellerPriceRub": round(candidate_price, 2),
        "violation": violation,
        "limitations": [
            fallback_reason,
            "использован временной коридор собственной цены; это не полноценный конкурентный срез",
        ],
    }


def _marketing_scenarios(
    snapshot: Mapping[str, Any],
    *,
    inventory_by_vendor: Mapping[str, Mapping[str, Any]],
    vendor_by_nm: Mapping[str, str],
    horizon_days: int,
    avoid_stockout: bool,
    preserve_profit: bool,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in list(snapshot.get("advertisingRows") or []):
        if isinstance(raw, Mapping):
            advert_id = str(raw.get("advert_id") or "").strip()
            if advert_id:
                grouped[advert_id].append(dict(raw))

    result: list[dict[str, Any]] = []
    for advert_id, rows in grouped.items():
        contributions = [_number(row.get("campaign_contribution_profit_7d_rub")) for row in rows]
        if not contributions or any(value is None for value in contributions):
            continue
        ad_orders = sum(_number(row.get("ad_orders_7d")) or 0 for row in rows)
        ad_units = sum(_number(row.get("ad_units_7d")) or 0 for row in rows)
        seven_day_contribution = sum(value or 0 for value in contributions)
        window_complete = all(row.get("stats_window_complete") is True for row in rows)
        if seven_day_contribution >= 0 or ad_orders > 0 or ad_units > 0:
            continue
        expected_profit = -seven_day_contribution * horizon_days / 7
        affected = sorted(
            {
                vendor_by_nm.get(str(row.get("nm_id") or ""), str(row.get("nm_id") or ""))
                for row in rows
                if row.get("nm_id") is not None
            }
        )
        target_vendor = affected[0] if len(affected) == 1 else ""
        baseline_days = _number((inventory_by_vendor.get(target_vendor) or {}).get("days_on_hand"))
        reasons: list[str] = []
        if not window_complete:
            reasons.append("неполное рекламное окно")
        if preserve_profit and expected_profit <= 0:
            reasons.append("положительный эффект contribution profit не подтверждён")
        stockout_probability = 0.0
        if avoid_stockout and stockout_probability > MAX_STOCKOUT_PROBABILITY:
            reasons.append("риск дефицита выше 5%")
        result.append(
            {
                "scenarioId": f"marketing:pause:{advert_id}",
                "family": "marketing",
                "target": advert_id,
                "affectedSkus": affected,
                "action": f"Кампания {advert_id}: приостановить расход без заказов",
                "calculationStatus": "completed" if window_complete else "needs_data",
                "executionStatus": "shadow",
                "horizonDays": horizon_days,
                "inventoryMetric": "turnover_days_30d_average_stock",
                "inventoryDaysBefore": _rounded(baseline_days),
                "inventoryDaysAfter": _rounded(baseline_days),
                "inventoryDaysReduction": 0.0,
                "expectedContributionProfitRub": round(expected_profit, 2),
                "p10ContributionProfitRub": round(expected_profit, 2),
                "stockoutProbability": stockout_probability,
                "eligible": not reasons,
                "constraintReasons": reasons,
                "evidenceMode": "observed zero-order campaign loss extrapolation",
                "limitations": [
                    "эффект экстраполирован из последнего полного 7-дневного окна"
                ],
            }
        )
    return result


def _supply_scenarios(
    snapshot: Mapping[str, Any],
    *,
    inventory_by_vendor: Mapping[str, Mapping[str, Any]],
    horizon_days: int,
    target_cover_days: float,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for raw in list(snapshot.get("supplyRows") or []):
        if not isinstance(raw, Mapping):
            continue
        row = dict(raw)
        target = str(row.get("vendor_code") or row.get("nm_id") or "").strip()
        stock = _number(row.get("stock_qty")) or 0.0
        velocity = _number(row.get("avg_daily_sales_7d"))
        raw_inbound = _number(row.get("supplier_inbound_audited_qty")) or 0.0
        supply_snapshot = _date_value(
            row.get("supplier_audit_snapshot_date")
            or row.get("supplier_snapshot_date")
        )
        supplier_eta = _date_value(row.get("supplier_eta_date"))
        snapshot_complete = row.get("supplier_snapshot_complete") is True
        inbound_status = "missing"
        inbound = 0.0
        if raw_inbound > 0:
            if not snapshot_complete:
                inbound_status = "unconfirmed"
            elif supply_snapshot is None or supplier_eta is None:
                inbound_status = "missing_eta"
            elif supplier_eta <= supply_snapshot:
                inbound_status = "overdue"
            elif supplier_eta > supply_snapshot + timedelta(days=horizon_days):
                inbound_status = "outside_horizon"
            else:
                inbound_status = "scheduled"
                inbound = raw_inbound
        if velocity is None or velocity <= 0:
            continue
        baseline_days = stock / velocity
        proposed_qty = max(0, ceil(velocity * target_cover_days - stock - inbound))
        moq = _number(row.get("minimum_order_qty"))
        if moq and proposed_qty > 0:
            proposed_qty = int(ceil(proposed_qty / moq) * moq)
        after_days = (stock + inbound + proposed_qty) / velocity
        missing = [
            label
            for field, label in (
                ("lead_time_days", "lead time"),
                ("target_cover_days", "утверждённое целевое покрытие"),
                ("minimum_order_qty", "MOQ"),
                ("replenishment_policy_version", "версия политики пополнения"),
            )
            if row.get(field) is None
        ]
        reasons = (
            [f"нет данных: {', '.join(missing)}"]
            if missing
            else ["нет прогноза contribution profit поставки"]
        )
        if raw_inbound > 0 and inbound_status != "scheduled":
            reasons.append(
                "будущая поставка не включена в запас: "
                f"статус {inbound_status}"
            )
        result.append(
            {
                "scenarioId": f"supply:replenish:{target}",
                "family": "supply",
                "target": target,
                "action": f"SKU {target}: рассмотреть дозаказ {proposed_qty} шт. до покрытия {target_cover_days:g} дней",
                "calculationStatus": "needs_data" if missing else "completed",
                "executionStatus": "blocked" if missing else "shadow",
                "horizonDays": horizon_days,
                "inventoryMetric": "current_cover_days_7d",
                "inventoryDaysBefore": round(baseline_days, 1),
                "inventoryDaysAfter": round(after_days, 1),
                "inventoryDaysReduction": round(baseline_days - after_days, 1),
                "expectedContributionProfitRub": None,
                "p10ContributionProfitRub": None,
                "stockoutProbability": 1.0 if baseline_days < horizon_days else 0.0,
                "eligible": False,
                "constraintReasons": reasons,
                "evidenceMode": "deterministic cover calculation",
                "limitations": [
                    "количество не исполняется без lead time, MOQ и утверждённой политики"
                ],
                "supplierInboundStatus": inbound_status,
                "observedSupplierInboundUnits": round(raw_inbound, 4),
                "includedSupplierInboundUnits": round(inbound, 4),
                "supplierEtaDate": (
                    supplier_eta.isoformat() if supplier_eta is not None else None
                ),
            }
        )
    return result


def _assign_scores(
    scenarios: list[dict[str, Any]],
    *,
    target_cover_days: float,
) -> None:
    eligible_profits = [
        max(0.0, float(item.get("expectedContributionProfitRub") or 0))
        for item in scenarios
        if item.get("eligible") is True
    ]
    all_profits = [
        max(0.0, float(item.get("expectedContributionProfitRub") or 0))
        for item in scenarios
    ]
    eligible_max_profit = max(eligible_profits, default=1.0) or 1.0
    blocked_max_profit = max(all_profits, default=1.0) or 1.0
    for item in scenarios:
        before = _number(item.get("inventoryDaysBefore"))
        reduction = max(0.0, _number(item.get("inventoryDaysReduction")) or 0.0)
        target_gap = max((before or target_cover_days) - target_cover_days, 0.0)
        turnover_progress = min(reduction / target_gap, 1.0) if target_gap > 0 else 0.0
        profit = max(0.0, _number(item.get("expectedContributionProfitRub")) or 0.0)
        max_profit = (
            eligible_max_profit
            if item.get("eligible") is True
            else blocked_max_profit
        )
        profit_component = profit / max_profit
        risk = min(max(_number(item.get("stockoutProbability")) or 0.0, 0.0), 1.0)
        score = 100 * (0.60 * turnover_progress + 0.40 * profit_component - 0.50 * risk)
        item["scoreBreakdown"] = {
            "turnoverProgress": round(turnover_progress, 6),
            "normalizedPositiveContributionProfit": round(profit_component, 6),
            "stockoutPenalty": round(risk, 6),
        }
        item["businessScore"] = round(score, 2)


def _goal_attainment(
    inventory_rows: Sequence[Mapping[str, Any]],
    *,
    recommendations: Sequence[Mapping[str, Any]],
    target_cover_days: float,
) -> dict[str, Any]:
    """Evaluate the portfolio target without mixing incompatible cover metrics."""

    above_target: list[dict[str, Any]] = []
    stockout_skus: list[dict[str, Any]] = []
    unmeasured: list[str] = []
    for raw in inventory_rows:
        row = dict(raw)
        target = str(row.get("vendor_code") or row.get("nm_id") or "").strip()
        if not target:
            continue
        current_stock = _number(row.get("current_stock_qty"))
        sales_units_7d = _number(row.get("sales_7d_units"))
        if (
            current_stock is not None
            and current_stock <= 0
            and sales_units_7d is not None
            and sales_units_7d > 0
        ):
            stockout_skus.append(
                {
                    "target": target,
                    "currentStockUnits": _rounded(current_stock),
                    "salesUnits7d": _rounded(sales_units_7d),
                    "latestSnapshotDate": row.get("latest_snapshot_date"),
                }
            )
            continue
        current_cover = _coverage_days(current_stock, sales_units_7d, 7)
        if current_cover is None:
            unmeasured.append(target)
            continue
        if current_cover > target_cover_days:
            above_target.append(
                {
                    "target": target,
                    "currentCoverDays7d": _rounded(current_cover),
                    "currentStockUnits": _rounded(current_stock),
                    "salesUnits7d": _rounded(sales_units_7d),
                    "latestSnapshotDate": row.get("latest_snapshot_date"),
                }
            )

    remaining: list[dict[str, Any]] = []
    resolved: list[dict[str, Any]] = []
    for row in above_target:
        target = str(row["target"])
        matching = [
            dict(item)
            for item in recommendations
            if target == str(item.get("target") or "")
            or target in {str(value) for value in list(item.get("affectedSkus") or [])}
        ]
        progress_scenarios = [
            item
            for item in matching
            if (_number(item.get("inventoryDaysReduction")) or 0) > 0
        ]
        qualifying = [
            item
            for item in progress_scenarios
            if (_number(item.get("inventoryDaysAfter")) is not None)
            and float(item["inventoryDaysAfter"]) <= target_cover_days
        ]
        best = min(
            progress_scenarios,
            key=lambda item: _number(item.get("inventoryDaysAfter")) or float("inf"),
            default=None,
        )
        enriched = {
            **row,
            "bestScenarioCoverDays": (
                _rounded(_number(best.get("inventoryDaysAfter"))) if best else None
            ),
            "bestScenarioMetric": best.get("inventoryMetric") if best else None,
            "bestScenarioId": best.get("scenarioId") if best else None,
        }
        (resolved if qualifying else remaining).append(enriched)

    above_target.sort(
        key=lambda row: (-float(row.get("currentCoverDays7d") or 0), str(row["target"]))
    )
    resolved.sort(key=lambda row: str(row["target"]))
    remaining.sort(
        key=lambda row: (-float(row.get("currentCoverDays7d") or 0), str(row["target"]))
    )
    unmeasured = sorted(set(unmeasured))
    stockout_skus.sort(key=lambda row: str(row["target"]))
    evaluated_count = len(inventory_rows) - len(unmeasured)
    goal_achieved = bool(
        evaluated_count > 0
        and not remaining
        and not stockout_skus
        and not unmeasured
    )
    status = (
        "not_achieved"
        if remaining or stockout_skus
        else ("not_proven" if unmeasured or evaluated_count == 0 else "achieved")
    )
    return {
        "status": status,
        "goalAchieved": goal_achieved,
        "targetCoverDays": target_cover_days,
        "currentMetric": "current_cover_days_7d",
        "currentMetricFormula": "current_stock_qty / (sales_7d_units / 7)",
        "scenarioMetricPolicy": (
            "Scenario coverage is reported with its own metric and is never substituted "
            "for current 7-day coverage."
        ),
        "evaluatedSkuCount": evaluated_count,
        "aboveTargetSkuCount": len(above_target),
        "stockoutSkuCount": len(stockout_skus),
        "resolvedByTopRecommendationsCount": len(resolved),
        "remainingAboveTargetSkuCount": len(remaining),
        "aboveTargetSkus": above_target,
        "stockoutSkus": stockout_skus,
        "resolvedByTopRecommendations": resolved,
        "remainingAboveTargetSkus": remaining,
        "unmeasuredSkus": unmeasured,
    }


def _coverage_days(stock: float | None, units: float | None, horizon_days: int) -> float | None:
    if stock is None or units is None or units <= 0:
        return None
    return stock / (units / horizon_days)


def _days_reduction(before: float | None, after: float | None) -> float:
    if before is None or after is None:
        return 0.0
    return before - after


def _first_number(row: Mapping[str, Any], *fields: str) -> float | None:
    for field in fields:
        value = _number(row.get(field))
        if value is not None:
            return value
    return None


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _date_value(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if value is None:
        return None
    try:
        return date.fromisoformat(str(value).strip()[:10])
    except ValueError:
        return None


def _rounded(value: float | None) -> float | None:
    return round(value, 1) if value is not None and isfinite(value) else None


def _money(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:,.0f} ₽".replace(",", " ")


__all__ = [
    "MAX_STOCKOUT_PROBABILITY",
    "PORTFOLIO_OPTIMIZER_SCHEMA_VERSION",
    "allowed_action_families",
    "optimize_portfolio",
    "validate_portfolio_optimization",
    "validate_priority_recommendations",
]
