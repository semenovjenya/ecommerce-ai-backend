# Portfolio preparation and verification

Prepared on 2026-11-06 from source commit `f19eee7`. The original Git history and uncommitted migration work were not copied.

## Scope

- Exported the committed backend, SQL, tests and experimental commerce-agent service.
- Excluded the original `.env`, personal resume files, historical execution reports, private migration evidence and build artifacts.
- Replaced client-specific database defaults, local development passwords and recorded financial example amounts.
- Renamed `фича1`–`фича5` to `sku_previews`, `document_import`, `action_board`, `unit_economics` and `commission_calculator`, including imports and tests.
- Added local-interface Docker port bindings and credential exclusions to Git/Docker ignore rules.
- Rewrote the root README, retained the detailed runtime guide and froze the verified Python environment in `requirements.lock.txt`.

## Verification

On a fresh Python 3.11 virtual environment:

```text
python -m pytest -q
1014 passed, 1 skipped, 68 subtests passed
```

The optional LangGraph/PostgreSQL integration test is skipped because that separate runtime is not installed. The experimental service's own suite is not included in this result.

Python source compilation and `docker compose --env-file .env.example config --quiet` passed. The initial clean-environment run exposed tests that depended on unmocked database enrichment calls. Test setup now supplies deterministic source responses; production calculations were not changed. One price test now explicitly verifies `needs_review` when the marketing mart is missing, while retaining its numerical recommendation assertions.

Fixture hashes were refreshed after replacing financial examples. These are portfolio hashes, not proof of equivalence to the private migration baseline.

A text scan against credential values in the original environment files found no matches. Common token-pattern matches in `services/commerce-agent/tests/test_safety.py` are intentional synthetic canaries used to test secret rejection. This bounded scan is not a comprehensive security audit.

## Not verified

- Real LLM calls, live WB reads/writes and full workflows against a seller database.
- Docker image execution and end-to-end deployment.
- The experimental LangGraph service and its private persistence integration.
- Economic effect, causal validity of pricing interventions, load capacity or production SLA.

## Known deployment gaps

Some legacy API routes have no shared authentication or authorization boundary. A public deployment needs consistent authentication, access control and an appropriate deployment configuration. Publishing source does not make this runtime production-ready. Default container passwords are for local development, and WB writes remain disabled by default.

GitHub Actions workflow installation was not available with the current GitHub credential scope; no CI status is claimed. The full local backend test command remains available above.
