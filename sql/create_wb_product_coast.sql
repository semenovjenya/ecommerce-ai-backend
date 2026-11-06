create schema if not exists finance;

create table if not exists finance.wb_product_coast (
    upload_date date not null,
    product_key text not null,
    vendor_code text,
    nm_id bigint,
    product_name text,
    brand_name text,
    cost_price_rub numeric not null check (cost_price_rub >= 0),
    source_filename text not null,
    source_row_number integer not null check (source_row_number > 0),
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint wb_product_coast_product_identity_check check (
        nm_id is not null or nullif(vendor_code, '') is not null
    ),
    constraint wb_product_coast_upload_product_key unique (upload_date, product_key)
);

create index if not exists idx_wb_product_coast_nm_id
    on finance.wb_product_coast (nm_id);

create index if not exists idx_wb_product_coast_vendor_code
    on finance.wb_product_coast (vendor_code);

create index if not exists idx_wb_product_coast_upload_date
    on finance.wb_product_coast (upload_date);
