# Langfuse Cloud: rollout и эксплуатация

Кодовая интеграция выключена по умолчанию и не меняет локальные
`agent_run_*` таблицы. Langfuse — внешний fail-open контур: ошибки SDK и
экспорта пишутся в application logs, но не меняют API/RQ/DB-статусы.

## Cloud project

1. Создать EU Cloud Pro project без public trace sharing.
2. В Project Settings установить Data Retention = 30 days.
3. Создать отдельные project keys для staging и production и хранить secret key
   только в secret manager/.env целевого окружения.
4. Добавить custom model definitions для точных имён активных моделей, которых
   нет в registry Langfuse. Цены брать из актуального provider contract на дату
   настройки; provider-reported usage/cost всегда имеет приоритет.
5. Задать `LANGFUSE_RELEASE` равным git SHA образа.

Для production-пилота:

```dotenv
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
LANGFUSE_TRACING_ENVIRONMENT=production
LANGFUSE_RELEASE=<git-sha>
LANGFUSE_SAMPLE_RATE=1.0
LANGFUSE_CAPTURE_MODE=full
LANGFUSE_MAX_FIELD_BYTES=262144
LANGFUSE_OPENAI_AGENTS_DUAL_EXPORT=true
```

Не задавать `OPENAI_AGENTS_DISABLE_TRACING`: он отключает поверхность Agent
SDK, которую использует OpenInference.

## Dashboards и saved views

Создать в Langfuse три dashboard:

### Cost & usage

- View: observations; type: generation.
- Measures: `sum(totalCost)`, `sum(inputTokens)`, `sum(outputTokens)`, count.
- Time grain: day.
- Dimensions: provided model name, `model_provider`, `model_profile`, `intent`,
  service.

### Latency & reliability

- Measures: p50/p95 latency, count, error count/rate.
- Dimensions: observation name, service, `target_agent`, model and release.
- Saved filters: errors; latency above the current seven-day p95;
  `agent-run-intake`; `agent-question`; `decision-simulation`;
  `memory-summary`; `feature2-document-answer`;
  `action-scenario-verification`.

### Agent operations

- Observation names: `agent-run-intake`, `agent-question`, `agent-run-result`.
- Dimensions/filters: run status, intent, query mode, workflow/prompt version,
  primary/target agent and manager decision status.
- Saved views: failed router, failed question, failed run, quality violations,
  runtime failures and simulation critic failures.

Не использовать `run_id`, `question_id`, trace ID или session ID как group-by:
они предназначены для точечного поиска и имеют высокую кардинальность.

## Семидневный пилот

1. Выполнить staging smoke: один router run, agent run с tools, прямой Responses
   API-вызов и simulation critic.
2. Включить production с dual export и sample rate 1.0.
3. Ежедневно сравнивать число локальных terminal runs/questions/simulations с
   Langfuse observations за тот же UTC-интервал.
4. Проверить, что не менее 99% локальных runs имеют Langfuse trace, orphan
   question spans не превышают 1%, а sampled generations имеют model, latency и
   usage. Cost должен быть provider-reported либо рассчитан по model definition.
5. Искать тестовые secret canaries в trace input/output/metadata; совпадений быть
   не должно.
6. На восьмой день поставить
   `LANGFUSE_OPENAI_AGENTS_DUAL_EXPORT=false` и перезапустить backend, worker и
   simulation-worker. В этом режиме OpenInference использует exclusive processor
   и заменяет стандартный OpenAI exporter.
7. Для rollback поставить `LANGFUSE_ENABLED=false` и перезапустить сервисы.

На 31-й день проверить, что первые production traces удалены retention policy.

## Correlation contract

`agent_run_traces.summary.observability` содержит Langfuse trace/root IDs,
capture mode, export mode и, после завершения, `result_observation_id`. Все
обновления summary обязаны merge-ить этот блок. Legacy runs без него получают
детерминированный trace context из локального `run_id`.

Raw business input/output экспортируется полностью до 256 KiB на одно поле.
Authorization, API keys, tokens, cookies, passwords, secrets и DSN passwords
маскируются непосредственно перед OTLP-экспортом. При усечении сохраняются
`original_bytes`, SHA-256 и безопасный prefix.
