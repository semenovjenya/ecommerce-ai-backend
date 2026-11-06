# Legacy M0 corpus v1

`manifest.json` pins the existing redacted regression fixtures by SHA-256. The
manifest is immutable after M0 acceptance; corrections create a new corpus version.

`human_manager_cases.jsonl` is intentionally absent until cases have been
labelled by a real business reviewer. Generated, synthetic, incomplete, or
LLM-produced labels do not satisfy the optional 40-case quality check. The
collector validates the complete existing Manager Evaluation contract,
including provenance, decisions, safety labels, priorities, executable actions,
audit, and measurement fields. The [M0 owner waiver](../../../../docs/ai-framework-migration/evidence/M0-owner-acceptance.md)
closed M0 without making 40 labels a prerequisite for M1/M2.

`deterministic_outputs.json` is also intentionally absent until canonical inputs
and output hashes are exported from the pinned business services. A passing pytest
suite does not substitute for business-output parity at M2/M8; the owner waiver
uses existing data and reproducible staging replay as the interim reference.

Capture a partial report inside the configured backend container:

```bash
docker compose run --rm --no-deps -v "$PWD:/app" -w /app backend \
  python scripts/capture_m0_baseline.py --allow-partial
```

Omit `--allow-partial` and pass `--environment staging` to run the original
strict completeness audit. It still exits non-zero until all original corpus
requirements are present; that result is not, by itself, the amended release
decision. The current M8 gate uses the rolling reference and acceptance rules in
the [migration ledger](../../../../docs/ai-framework-migration/MIGRATION_STATUS.md).
