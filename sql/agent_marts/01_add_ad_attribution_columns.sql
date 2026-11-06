-- Analytics schema compatibility migration for WB advertising fullstats.
--
-- The source API field is intentionally stored under its raw name, sum_price.
-- NULL means that the ETL has not supplied the metric for that row; a proven
-- zero must be stored as 0. Consumers must not coalesce this column to zero
-- when checking evidence completeness.

alter table if exists marketing.wb_marketing_campaign_stats_daily
    add column if not exists sum_price numeric;

alter table if exists marketing.wb_marketing_campaign_stats_daily_nms
    add column if not exists sum_price numeric;

create table if not exists marketing.wb_marketing_campaign_stats_daily_watermarks (
    stat_date date primary key,
    campaign_stats_complete boolean not null default false,
    campaign_sku_stats_complete boolean not null default false,
    source_run_id text,
    source_completed_at timestamp with time zone,
    last_synced_at timestamp with time zone not null default now()
);

do $migration$
begin
    if not exists (
        select 1
        from pg_constraint constraint_row
        where constraint_row.conrelid =
                'marketing.wb_marketing_campaign_stats_daily_watermarks'::regclass
          and constraint_row.conname =
                'wb_marketing_campaign_stats_daily_watermarks_proven_complete'
    ) then
        alter table marketing.wb_marketing_campaign_stats_daily_watermarks
            add constraint wb_marketing_campaign_stats_daily_watermarks_proven_complete
            check (
                not (campaign_stats_complete or campaign_sku_stats_complete)
                or (
                    source_run_id is not null
                    and source_completed_at is not null
                )
            );
    end if;
end
$migration$;

comment on column marketing.wb_marketing_campaign_stats_daily.sum_price is
    'WB fullstats attributed order revenue for advert_id and stat_date, RUB; NULL means not supplied by ETL.';

comment on column marketing.wb_marketing_campaign_stats_daily_nms.sum_price is
    'WB fullstats attributed order revenue for advert_id, nm_id and stat_date, RUB; NULL means not supplied by ETL.';

comment on table marketing.wb_marketing_campaign_stats_daily_watermarks is
    'ETL completion watermark; only a complete day can prove that an absent campaign/SKU stats row means zero activity.';
