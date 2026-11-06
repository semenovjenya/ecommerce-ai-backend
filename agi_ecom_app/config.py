from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


AGENT_WORKFLOW_VERSIONS = frozenset({"v1", "v2_prompt", "v2_optimized"})
DEFAULT_AGENT_WORKFLOW_VERSION = "v1"
DECISION_SIMULATION_MODES = frozenset({"off", "shadow", "enforced"})
DEFAULT_DECISION_SIMULATION_MODE = "off"


@dataclass(frozen=True)
class PostgresConfig:
    host: str
    port: int
    name: str
    user: str
    password: str | None = field(repr=False)
    sslmode: str
    password_env_var: str
    missing_password_message: str

    @property
    def dsn(self) -> str:
        if not self.password:
            raise RuntimeError(self.missing_password_message)
        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.name}"
            f"?sslmode={self.sslmode}"
        )


@dataclass(frozen=True)
class SkuPreviewConfig:
    schema: str
    table: str
    sku_column: str
    wb_nm_id_column: str | None
    title_column: str | None
    image_url_column: str | None
    product_url_column: str | None


@dataclass(frozen=True)
class DalkosPostgresConfig:
    host: str
    port: int
    name: str
    user: str
    password: str | None = field(repr=False)
    sslmode: str


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None = field(repr=False)
    openai_model: str
    openai_memory_model: str
    openrouter_api_key: str | None = field(repr=False)
    openrouter_base_url: str
    openrouter_model: str
    openrouter_http_referer: str | None
    openrouter_title: str | None
    llm_model_profile: str
    db: PostgresConfig
    memory_db: PostgresConfig
    dalkos_db: DalkosPostgresConfig
    memory_context_message_limit: int
    backend_host: str
    backend_port: int
    backend_base_url: str
    airflow_base_url: str
    airflow_username: str | None
    airflow_password: str | None = field(repr=False)
    airflow_contract_dag_id: str
    redis_url: str
    agent_queue_name: str
    agent_job_timeout_seconds: int
    agent_workflow_version: str
    manager_decision_enabled: bool
    manager_daily_enabled: bool
    manager_daily_hour: int
    manager_daily_timezone: str
    manager_daily_chat_id: str
    manager_jwt_secret: str | None = field(repr=False)
    manager_jwt_ttl_seconds: int
    wb_ads_write_enabled: bool
    wb_price_write_enabled: bool
    wb_api_sandbox: bool
    wb_promotion_token: str | None = field(repr=False)
    wb_prices_token: str | None = field(repr=False)
    decision_simulation_mode: str
    decision_simulation_queue_name: str
    decision_simulation_maintenance_queue_name: str
    decision_simulation_sample_count: int
    decision_simulation_batch_size: int
    decision_simulation_job_timeout_seconds: int
    decision_simulation_stale_after_seconds: int
    decision_simulation_llm_critic_enabled: bool
    experiment_program_queue_name: str
    experiment_program_job_timeout_seconds: int
    experiment_program_outbox_lease_seconds: int
    experiment_program_outbox_recovery_seconds: int
    experiment_program_stale_after_seconds: int
    experiment_program_simulation_max_reconciles: int
    experiment_program_auto_attach: bool
    supply_latest_known_as_current: bool
    langfuse_enabled: bool
    langfuse_public_key: str | None = field(repr=False)
    langfuse_secret_key: str | None = field(repr=False)
    langfuse_base_url: str
    langfuse_tracing_environment: str
    langfuse_release: str | None
    langfuse_sample_rate: float
    langfuse_capture_mode: str
    langfuse_max_field_bytes: int
    langfuse_openai_agents_dual_export: bool
    postgres_pool_min_size: int
    postgres_pool_max_size: int
    postgres_pool_timeout_seconds: float
    memory_postgres_pool_min_size: int
    memory_postgres_pool_max_size: int
    memory_postgres_pool_timeout_seconds: float
    sku_preview: SkuPreviewConfig

    @property
    def db_host(self) -> str:
        return self.db.host

    @property
    def db_port(self) -> int:
        return self.db.port

    @property
    def db_name(self) -> str:
        return self.db.name

    @property
    def postgres_dsn(self) -> str:
        return self.db.dsn

    @property
    def memory_postgres_dsn(self) -> str:
        return self.memory_db.dsn


def _optional_env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name, default)
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _bounded_float_env(name: str, default: float, *, minimum: float, maximum: float) -> float:
    value = float(os.getenv(name, str(default)))
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}; got {value}.")
    return value


def _fixed_int_env(name: str, expected: int) -> int:
    """Read a v1 fixed-size knob and fail fast instead of queuing an invalid run."""

    value = int(os.getenv(name, str(expected)))
    if value != expected:
        raise ValueError(
            f"{name} is fixed at {expected} for advertising-bootstrap-v1; got {value}."
        )
    return value


def normalize_agent_workflow_version(value: str | None) -> str:
    normalized = str(value or DEFAULT_AGENT_WORKFLOW_VERSION).strip().lower()
    if normalized not in AGENT_WORKFLOW_VERSIONS:
        supported = ", ".join(sorted(AGENT_WORKFLOW_VERSIONS))
        raise ValueError(
            f"Unsupported AGENT_WORKFLOW_VERSION={value!r}. Supported values: {supported}."
        )
    return normalized


def normalize_decision_simulation_mode(value: str | None) -> str:
    normalized = str(value or DEFAULT_DECISION_SIMULATION_MODE).strip().lower()
    if normalized not in DECISION_SIMULATION_MODES:
        supported = ", ".join(sorted(DECISION_SIMULATION_MODES))
        raise ValueError(
            f"Unsupported DECISION_SIMULATION_MODE={value!r}. Supported values: {supported}."
        )
    return normalized


def get_settings() -> Settings:
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        openai_memory_model=os.getenv("OPENAI_MEMORY_MODEL") or os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
        openrouter_base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        openrouter_model=os.getenv("OPENROUTER_MODEL", "google/gemma-4-31b-it:free"),
        openrouter_http_referer=_optional_env("OPENROUTER_HTTP_REFERER"),
        openrouter_title=_optional_env("OPENROUTER_TITLE", "AI-ECOM Backend"),
        llm_model_profile=os.getenv("LLM_MODEL_PROFILE", "openai_default"),
        db=PostgresConfig(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "55434")),
            name=os.getenv("POSTGRES_DB", "wb_demo"),
            user=os.getenv("POSTGRES_USER", "wb_user"),
            password=os.getenv("POSTGRES_PASSWORD"),
            sslmode=os.getenv("POSTGRES_SSLMODE", "disable"),
            password_env_var="POSTGRES_PASSWORD",
            missing_password_message=(
                "Не найден POSTGRES_PASSWORD. Укажи пароль для базы "
                "localhost:55434/wb_demo и повтори запрос."
            ),
        ),
        memory_db=PostgresConfig(
            host=os.getenv("MEMORY_POSTGRES_HOST", "localhost"),
            port=int(os.getenv("MEMORY_POSTGRES_PORT", "55435")),
            name=os.getenv("MEMORY_POSTGRES_DB", "postgresmemory"),
            user=os.getenv("MEMORY_POSTGRES_USER", "postgresmemory_user"),
            password=os.getenv("MEMORY_POSTGRES_PASSWORD"),
            sslmode=os.getenv("MEMORY_POSTGRES_SSLMODE", "disable"),
            password_env_var="MEMORY_POSTGRES_PASSWORD",
            missing_password_message=(
                "Не найден MEMORY_POSTGRES_PASSWORD. Укажи пароль для базы "
                "памяти чата localhost:55435/postgresmemory и повтори запрос."
            ),
        ),
        dalkos_db=DalkosPostgresConfig(
            host=os.getenv("DALKOS_POSTGRES_HOST", "localhost"),
            port=int(os.getenv("DALKOS_POSTGRES_PORT", "4444")),
            name=os.getenv("DALKOS_POSTGRES_DB", "contracts_demo"),
            user=os.getenv("DALKOS_POSTGRES_USER", "demo_user"),
            password=_optional_env("DALKOS_POSTGRES_PASSWORD"),
            sslmode=os.getenv("DALKOS_POSTGRES_SSLMODE", "disable"),
        ),
        memory_context_message_limit=int(os.getenv("MEMORY_CONTEXT_MESSAGE_LIMIT", "12")),
        backend_host=os.getenv("BACKEND_HOST", "0.0.0.0"),
        backend_port=int(os.getenv("BACKEND_PORT", "8000")),
        backend_base_url=os.getenv("BACKEND_BASE_URL", "http://127.0.0.1:8000"),
        airflow_base_url=os.getenv("AIRFLOW_BASE_URL", "http://127.0.0.1:8082"),
        airflow_username=_optional_env("AIRFLOW_USERNAME", "admin"),
        airflow_password=_optional_env("AIRFLOW_PASSWORD", "admin"),
        airflow_contract_dag_id=os.getenv("AIRFLOW_CONTRACT_DAG_ID", "contract_document_ingest"),
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        agent_queue_name=os.getenv("AGENT_QUEUE_NAME", "agent-runs"),
        agent_job_timeout_seconds=int(os.getenv("AGENT_JOB_TIMEOUT_SECONDS", "900")),
        agent_workflow_version=normalize_agent_workflow_version(
            os.getenv("AGENT_WORKFLOW_VERSION", DEFAULT_AGENT_WORKFLOW_VERSION)
        ),
        manager_decision_enabled=_env_bool("MANAGER_DECISION_ENABLED"),
        manager_daily_enabled=_env_bool("MANAGER_DAILY_ENABLED"),
        manager_daily_hour=max(0, min(23, int(os.getenv("MANAGER_DAILY_HOUR", "9")))),
        manager_daily_timezone=os.getenv("MANAGER_DAILY_TIMEZONE", "Europe/Moscow"),
        manager_daily_chat_id=os.getenv("MANAGER_DAILY_CHAT_ID", "manager-daily"),
        manager_jwt_secret=_optional_env("MANAGER_JWT_SECRET"),
        manager_jwt_ttl_seconds=max(
            60,
            min(3600, int(os.getenv("MANAGER_JWT_TTL_SECONDS", "900"))),
        ),
        wb_ads_write_enabled=_env_bool("WB_ADS_WRITE_ENABLED"),
        wb_price_write_enabled=_env_bool("WB_PRICE_WRITE_ENABLED"),
        wb_api_sandbox=_env_bool("WB_API_SANDBOX", default=True),
        wb_promotion_token=_optional_env("WB_PROMOTION_TOKEN"),
        wb_prices_token=_optional_env("WB_PRICES_TOKEN"),
        decision_simulation_mode=normalize_decision_simulation_mode(
            os.getenv("DECISION_SIMULATION_MODE", DEFAULT_DECISION_SIMULATION_MODE)
        ),
        decision_simulation_queue_name=os.getenv(
            "DECISION_SIMULATION_QUEUE_NAME", "decision-simulations"
        ),
        decision_simulation_maintenance_queue_name=os.getenv(
            "DECISION_SIMULATION_MAINTENANCE_QUEUE_NAME",
            "decision-simulations-maintenance",
        ),
        decision_simulation_sample_count=_fixed_int_env(
            "DECISION_SIMULATION_SAMPLE_COUNT", 500
        ),
        decision_simulation_batch_size=_fixed_int_env(
            "DECISION_SIMULATION_BATCH_SIZE", 50
        ),
        decision_simulation_job_timeout_seconds=max(
            60, int(os.getenv("DECISION_SIMULATION_JOB_TIMEOUT_SECONDS", "900"))
        ),
        decision_simulation_stale_after_seconds=max(
            60, int(os.getenv("DECISION_SIMULATION_STALE_AFTER_SECONDS", "1200"))
        ),
        decision_simulation_llm_critic_enabled=_env_bool(
            "DECISION_SIMULATION_LLM_CRITIC_ENABLED", default=True
        ),
        experiment_program_queue_name=os.getenv(
            "EXPERIMENT_PROGRAM_QUEUE_NAME", "experiment-programs"
        ),
        experiment_program_job_timeout_seconds=max(
            60, int(os.getenv("EXPERIMENT_PROGRAM_JOB_TIMEOUT_SECONDS", "900"))
        ),
        experiment_program_outbox_lease_seconds=max(
            15, int(os.getenv("EXPERIMENT_PROGRAM_OUTBOX_LEASE_SECONDS", "60"))
        ),
        experiment_program_outbox_recovery_seconds=max(
            15, int(os.getenv("EXPERIMENT_PROGRAM_OUTBOX_RECOVERY_SECONDS", "30"))
        ),
        experiment_program_stale_after_seconds=max(
            60, int(os.getenv("EXPERIMENT_PROGRAM_STALE_AFTER_SECONDS", "300"))
        ),
        experiment_program_simulation_max_reconciles=max(
            1,
            int(os.getenv("EXPERIMENT_PROGRAM_SIMULATION_MAX_RECONCILES", "3")),
        ),
        experiment_program_auto_attach=_env_bool(
            "EXPERIMENT_PROGRAM_AUTO_ATTACH", default=False
        ),
        supply_latest_known_as_current=_env_bool(
            "SUPPLY_LATEST_KNOWN_AS_CURRENT", default=False
        ),
        langfuse_enabled=_env_bool("LANGFUSE_ENABLED", default=False),
        langfuse_public_key=_optional_env("LANGFUSE_PUBLIC_KEY"),
        langfuse_secret_key=_optional_env("LANGFUSE_SECRET_KEY"),
        langfuse_base_url=os.getenv(
            "LANGFUSE_BASE_URL", "https://cloud.langfuse.com"
        ).rstrip("/"),
        langfuse_tracing_environment=os.getenv(
            "LANGFUSE_TRACING_ENVIRONMENT", "development"
        ).strip().lower(),
        langfuse_release=_optional_env("LANGFUSE_RELEASE"),
        langfuse_sample_rate=_bounded_float_env(
            "LANGFUSE_SAMPLE_RATE", 1.0, minimum=0.0, maximum=1.0
        ),
        langfuse_capture_mode=os.getenv("LANGFUSE_CAPTURE_MODE", "full").strip().lower(),
        langfuse_max_field_bytes=max(
            1_024, int(os.getenv("LANGFUSE_MAX_FIELD_BYTES", "262144"))
        ),
        langfuse_openai_agents_dual_export=_env_bool(
            "LANGFUSE_OPENAI_AGENTS_DUAL_EXPORT", default=True
        ),
        postgres_pool_min_size=max(
            1, int(os.getenv("POSTGRES_POOL_MIN_SIZE", "1"))
        ),
        postgres_pool_max_size=max(
            1, int(os.getenv("POSTGRES_POOL_MAX_SIZE", "10"))
        ),
        postgres_pool_timeout_seconds=max(
            1.0, float(os.getenv("POSTGRES_POOL_TIMEOUT_SECONDS", "10"))
        ),
        memory_postgres_pool_min_size=max(
            1, int(os.getenv("MEMORY_POSTGRES_POOL_MIN_SIZE", "1"))
        ),
        memory_postgres_pool_max_size=max(
            1, int(os.getenv("MEMORY_POSTGRES_POOL_MAX_SIZE", "10"))
        ),
        memory_postgres_pool_timeout_seconds=max(
            1.0, float(os.getenv("MEMORY_POSTGRES_POOL_TIMEOUT_SECONDS", "10"))
        ),
        sku_preview=SkuPreviewConfig(
            schema=os.getenv("SKU_PREVIEW_SCHEMA", "analytics_ai"),
            table=os.getenv("SKU_PREVIEW_TABLE", "wb_sku_ai_analysis"),
            sku_column=os.getenv("SKU_PREVIEW_SKU_COLUMN", "vendor_code"),
            wb_nm_id_column=_optional_env("SKU_PREVIEW_WB_NM_ID_COLUMN", "nm_id"),
            title_column=_optional_env("SKU_PREVIEW_TITLE_COLUMN", "title"),
            image_url_column=_optional_env("SKU_PREVIEW_IMAGE_URL_COLUMN", "main_photo_url"),
            product_url_column=_optional_env("SKU_PREVIEW_PRODUCT_URL_COLUMN", "product_url"),
        ),
    )
