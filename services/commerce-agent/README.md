# AI-ECOM Commerce Agent

Изолированный deployment package графа `commerce_agent`. На этапе M1 это
детерминированный skeleton без моделей, business tools и write-операций.

```text
intake -> router -> query_plan -> parallel fixed domain subgraphs
       -> evidence_gate -> gap_resolution/deep_research -> synthesis
       -> approval_interrupt -> action_execution -> effect_measurement_schedule
```

Sales, Marketing, Supply, Price и Finance уже представлены отдельными
subgraphs `prepare -> collect_evidence -> analyze -> validate_and_publish`.
Каждый subgraph получает только `query_plan` и публикует только собственные
status/result, gaps и errors. В M1 они возвращают `planned`/`skipped` и typed
`not_implemented` gaps; Deep Agent недостижим, action execution работает
fail-closed.

Публичный graph input содержит только `schema_version=1` и `messages`.
`schema_version` — обязательная константа protocol envelope; в M6 её будет
устанавливать BFF. Identity, plan, decisions, interrupts, actions и receipts
не принимаются через публичный input.

```bash
uv sync --group dev
uv run pytest
uv run langgraph dev --no-browser
```

## PostgreSQL checkpoints

Self-managed runtime использует `AsyncPostgresSaver` через bounded psycopg
pool. Schema migrations выполняются отдельной deployment-командой; обычный
startup не получает DDL-права и не вызывает `.setup()` автоматически.

Локальная отдельная agent database и restart smoke:

```bash
docker compose -f compose.persistence.yml up -d --wait agent-postgres
export COMMERCE_AGENT_ENVIRONMENT=development
export COMMERCE_AGENT_DATABASE_URI='postgresql://commerce_agent:commerce-agent-local-only@localhost:55432/commerce_agent?sslmode=disable'

uv run python scripts/checkpoint_db.py setup
uv run python scripts/checkpoint_db.py pause --thread-id m1-restart-smoke
uv run python scripts/checkpoint_db.py resume --thread-id m1-restart-smoke
uv run python scripts/checkpoint_db.py inspect --thread-id m1-restart-smoke
```

`inspect` выводит фактические `domain_statuses` и `analysis_statuses`, а также
число gaps/evidence/actions/receipts и ожидающие ноды. Он не выводит raw
messages, identity и текстовые поля checkpoint. В M1 skeleton статус домена
`planned` и результат `not_implemented` **не** означают завершённый бизнес-анализ.
Это локальная диагностическая сводка, не замена authenticated state/history API
и UI из `M1.5`/`M7`.

## M1.5 durable protocol records

Для self-managed Agent Streaming Protocol добавлены отдельные PostgreSQL
`commerce_agent_threads`, `commerce_agent_runs` и `commerce_agent_events`.
Их миграция запускается явно, после checkpoint schema setup:

```bash
uv run python scripts/protocol_db.py setup
```

`ProtocolStore` привязывает thread и каждое чтение event history к проверенным
`tenant_id`/`user_id`, дедуплицирует `run.start` по `command_id` и выдаёт
монотонный `seq` транзакцией PostgreSQL. Запрос с повторным command ID и другим
input или policy/context отклоняется. Синтетический credential-like input/event отклоняется до
записи; события можно прочитать по `since` из нового процесса/соединения.
`GuardedGraph.astream_protocol_events` открывает LangGraph v3 stream только
после ingress-проверки и не принимает caller-controlled stream options.
`normalize_graph_event` преобразует реальные v3 `messages`/`values` и другие
события в JSON, сохраняет protocol timestamp/interrupts и убирает локальный
sequence LangGraph; PostgreSQL назначает thread-wide sequence. Интеграционный
тест проверяет запись и replay structured `agent_statuses` из реального graph
stream через новый pool; отдельный тест проверяет реальный interrupt payload.

### Authenticated state/history read path

Внутренний read-only ASGI service `commerce_agent.protocol_read_api:app`
возвращает latest `CommerceState` и ограниченную историю checkpoints:
`GET /threads/{thread_id}/state` и `GET /threads/{thread_id}/history?limit=20`.
Он требует `COMMERCE_AGENT_SERVICE_TOKEN` (минимум 32 bytes) и проверенный BFF
контекст `tenant_id`/`user_id` в headers; доступ к thread проверяется в
PostgreSQL до чтения checkpoint. Credential-like значения отклоняются до
ответа; полный redaction/PII gate остаётся за M6/M8. Trace/checkpoint
metadata, кроме `checkpoint_id`, не публикуются. Только versioned поля
`CommerceState` могут выйти в ответ; несовместимая schema даёт безопасный 409.
State ограничен 1 MiB, history — 100 snapshots и 4 MiB суммарно. Таблицы создаются отдельными
командами `checkpoint_db.py setup` и `protocol_db.py setup`, не при старте API.

FastAPI предоставляет только соответствующие read routes через
`/v2/agent/threads/{thread_id}/state|history`. Он проверяет существующий Bearer
JWT, сам устанавливает single-tenant ID и user ID и не передаёт пользовательский
JWT во внутренний runtime. Для development нужны server-side
`COMMERCE_AGENT_INTERNAL_URL`, `COMMERCE_AGENT_SERVICE_TOKEN` и
`COMMERCE_AGENT_TENANT_ID`, `COMMERCE_AGENT_ENVIRONMENT`; реальные значения не
коммитятся. В staging/production upstream требует HTTPS, proxy-переменные
процесса для этого вызова игнорируются. LangSmith trace
остаётся маскированным: authorized state читается из PostgreSQL checkpoint.
После явного `checkpoint_db.py setup` и `protocol_db.py setup` development
API можно запустить командой
`uv run uvicorn commerce_agent.protocol_read_api:app --host 127.0.0.1 --port 8124`
при наличии приватного service token в environment. Это пока read-only
процесс; он не принимает `run.start` и не транслирует SSE.

Это пока persistence/stream-ingress основа `M1.5`, а не готовый Agent Server:
HTTP commands/SSE и строгая проверка `CommerceInput` до создания run,
фоновое выполнение и recovery runs, повторная доставка
producer events без дублей, проверка версии/формы protocol schema, остальные
BFF routes, pinned React SDK, interrupt/resume и retention ещё не подключены. Production routing и legacy
chat path не меняются. Для rollback до cutover достаточно вернуть package/image;
неиспользуемые protocol tables можно оставить без данных до отдельной
проверенной cleanup-операции.

`pause` и `resume` намеренно запускаются разными Python-процессами. Первый
процесс останавливается после `query_plan`; второй загружает тот же checkpoint,
выполняет пять domain branches и завершает thread. Для staging/production URI
без обязательного TLS отклоняется до подключения. DSN существует только в
secret environment и не включается в state, logs или graph metadata.

## Observability

Package поддерживает два независимых exporter: их сетевой отказ не меняет
бизнесовый результат, а небезопасная конфигурация останавливает инициализацию:

- LangSmith EU tracing показывает LangGraph/node trace;
- Langfuse показывает LLM/tool observations, usage, cost и dashboards.

Оба включаются только environment variables из `.env.example`. Для LangSmith
используется `COMMERCE_AGENT_LANGSMITH_TRACING=true` при обязательных
`LANGSMITH_TRACING=false`, `LANGSMITH_TRACING_V2=false`,
`LANGCHAIN_TRACING=false` и `LANGCHAIN_TRACING_V2=false`: глобальный auto-tracer
не проходит наш export boundary и потому отклоняется до запуска графа. Явный
`LangChainTracer` с EU `Client` перед отправкой применяет строгую проекцию run:
сохраняет структуру (ID, parent, порядок, имя разрешённой ноды, тип, время),
но удаляет inputs/outputs, metadata, tags, events, attachments, произвольные поля и
текст ошибки. `GuardedGraph` до запуска отклоняет переданные вызывающим кодом
callbacks, `run_name`, tags и metadata. При завершении диагностического run
или остановке service нужно вызвать `GuardedGraph.flush_traces()`: он ждёт
фоновые projection workers перед финальным drain. Отказ exporter не меняет
бизнесовый результат. Проекция закреплена unit-тестом pinned LangSmith SDK и offline
full-graph test; после upgrade SDK их необходимо повторить. Правильный EU
API endpoint — `https://eu.api.smith.langchain.com`; web URL
`https://eu.smith.langchain.com` и US API не принимаются как API endpoint.
При включённом LangSmith отсутствие `LANGSMITH_HIDE_INPUTS=true`,
`LANGSMITH_HIDE_OUTPUTS=true` или `LANGSMITH_HIDE_METADATA=true` останавливает
инициализацию графа. Langfuse `should_export_span` допускает только spans
Langfuse SDK с безопасными именами и разрешёнными resource attributes; spans
с events, links или произвольным status description отклоняются целиком,
поскольку эти поля нельзя исправить на export-stage. Такой фильтр может
разорвать trace tree или скрыть error span; полноту дерева проверять в staging.
Langfuse export mask затем удаляет произвольные span attributes,
сохраняя только проверенные числовые usage/cost details, тип observation и
имена моделей из `LANGFUSE_MODEL_ALLOWLIST`. Если модель не разрешена явно,
её имя не экспортируется и автоматическая атрибуция стоимости Langfuse для
этого run может быть неполной; при включении новой модели allowlist и canary
проверяются заново. Это независимая защита, поскольку
LangSmith masking не распространяется на Langfuse. Credentials не копируются в
сериализуемые graph metadata, state или checkpoints. Отказ Langfuse callback не
блокирует run. Неисправный LangSmith safety callback останавливает инициализацию,
чтобы не возник unprojected fallback; отказ уже подключённого exporter не
меняет результат run. LangSmith Deployments для этой схемы не требуются.

Production self-managed runtime должен открывать граф через
`open_postgres_graph`: его `GuardedGraph` проверяет input/config/context **до**
вызова LangGraph и tracing callbacks; дополнительные параметры вызова и
параметры чтения state/history проходят ту же проверку. Прямой экспорт `graph` в `langgraph.json`
оставлен только для локального skeleton development с синтетическими данными;
он не является разрешённым public ingress. В M1.5 тот же guard должен быть
подключён на authenticated HTTP boundary до логирования, tracing и постановки
run в очередь. Masking экспортёров — defense in depth, а не замена ingress.

Тесты package проверяют frozen state contract, parallel reducers, ownership
subgraphs, conditional routing, checkpoint replay, secret canaries и запрет
write path. Корневой backend suite намеренно не собирает эти тесты: package
использует собственные `pyproject.toml` и `uv.lock` на Python 3.11.

Checkpointer передаётся через `build_graph(checkpointer=...)`; lifecycle владеет
`open_postgres_graph`. Отдельные migrations, strict serializer, connection pool
и restart recovery реализованы и проверены на PostgreSQL 16. `langgraph.json`
сохраняет local development contract; production retention не должен полагаться
только на его TTL-настройку. 30-дневный cleanup job реализуется перед staging,
когда будет зафиксирована run/thread schema self-managed API. Изменение
LangGraph, psycopg или checkpointer требует повторного persistence smoke.

Read-only проверка доступа к LangSmith EU из process environment:

```bash
uv run python scripts/check_langsmith_eu.py
```

Она запрашивает один project через EU API и выводит только результат/HTTP
status, никогда значение ключа. Если локальный ключ хранится в ignored корневом
`.env`, для development можно добавить `--env-file ../../.env` к `uv run`;
скрипт всё равно обращается только к EU endpoint. HTTP 403 не означает, что
tracing EU готов: может понадобиться ключ EU workspace или `LANGSMITH_WORKSPACE_ID`.

После успешного read-only preflight отдельный canary выполняет синтетический
graph run через `GuardedGraph` и agent PostgreSQL, затем читает LangSmith EU
trace и проверяет, что input/output, application metadata и синтетические
identity-значения не сохранены. EU API добавляет только структурное числовое
metadata-поле `ls_run_depth`; любые другие metadata probe отклоняет:

```bash
COMMERCE_AGENT_ENVIRONMENT=development \
COMMERCE_AGENT_DATABASE_URI='postgresql://commerce_agent:commerce-agent-local-only@localhost:55432/commerce_agent?sslmode=disable' \
uv run --no-sync --env-file ../../.env python scripts/check_langsmith_export.py
```

Этот скрипт закрепляет EU endpoint и privacy flags в собственной process
environment до инициализации graph/client, отключает глобальный auto-tracer и
очищает SDK env-cache после смены настроек; inherited US endpoint не
используется. Если EU preflight вернул
403, скрипт останавливается **до** создания run и не считается evidence
успешного экспорта. Для ключа с доступом к нескольким workspace может
понадобиться `LANGSMITH_WORKSPACE_ID` в ignored `.env`.

При readback допускается только серверное структурное metadata-поле
`ls_run_depth` с неотрицательным целым значением; прикладные metadata
по-прежнему запрещены.

Synthetic Langfuse export/readback canary создаёт один тестовый trace:

```bash
uv run python scripts/check_langfuse_mask.py
```

Скрипт проверяет сохранённые input/output/metadata, модель, usage и cost через
Observations API. В том же trace он создаёт дочерний span с синтетическим
credential-like именем и безопасный контрольный span: контрольный должен
сохраниться, небезопасный — отсутствовать. Результат проверяется двумя
последовательными чтениями; выводятся только trace ID и итог, без credentials
или canary value. Это внешний negative canary для span name, но не для всех
возможных OTel events/resource attributes. Export mask не меняет эти поля:
запрет секретов на ingress и typed tools остаётся обязательным.

Отдельный Langfuse-only smoke выполняет синтетический `commerce_agent` через
`GuardedGraph` и локальную agent PostgreSQL, затем проверяет node observations
и отсутствие raw input/output в сохранённом trace:

```bash
COMMERCE_AGENT_ENVIRONMENT=development \
COMMERCE_AGENT_DATABASE_URI='postgresql://commerce_agent:commerce-agent-local-only@localhost:55432/commerce_agent?sslmode=disable' \
uv run --no-sync --env-file ../../.env python scripts/check_langfuse_graph.py
```

Скрипт принудительно отключает LangSmith tracing только для этого процесса;
проверка не доказывает доступность EU LangSmith или end-to-end BFF streaming.

Этот package не импортирует `agi_ecom_app` и не имеет доступа к SQL, WB или
legacy OpenAI Agents SDK. Такие зависимости появятся только через typed HTTP
adapters в M2.
