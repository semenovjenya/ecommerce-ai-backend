# Справочник производных признаков для агентов

Этот документ - пример справочника формул, который можно дать агентам как рабочий
протокол: агент не обязан иметь готовую витрину признаков, но может временно
считать признаки в read-only SQL через `WITH`, `CASE`, `lag`, rolling windows и
агрегации.

Правило использования:

1. Сначала агент формулирует гипотезу.
2. Затем выбирает производный признак из своего домена.
3. Считает признак в SQL или готовым tool.
4. Возвращает статус: `подтверждено`, `частично подтверждено`, `не подтверждено`
   или `нет данных`.
5. Не называет признак первопричиной без смежной проверки, если причина лежит в
   другом домене.

## Sales Agent

| Признак | Что показывает | Как считается | Grain | Как читать |
| --- | --- | --- | --- | --- |
| `orders_delta_pct` | Изменение заказов между периодами | `(orders_current - orders_previous) / nullif(orders_previous, 0)` | SKU + период | Если резко ниже 0, спрос или верх воронки просел, но причина еще не доказана. |
| `sales_delta_pct` | Изменение фактических продаж | `(sales_current - sales_previous) / nullif(sales_previous, 0)` | SKU + период | Показывает механику изменения реализации, не первопричину. |
| `revenue_delta_pct` | Изменение выручки продаж | `(revenue_current - revenue_previous) / nullif(revenue_previous, 0)` | SKU + период | Помогает ранжировать вклад SKU в рост/падение. |
| `avg_price_delta_pct` | Изменение средней цены продажи | `(avg_price_current - avg_price_previous) / nullif(avg_price_previous, 0)` | SKU + период | Если цена выросла и заказы упали, нужен price/marketing cross-check. |
| `open_to_cart_delta_pp` | Изменение перехода из открытия карточки в корзину | `open_to_cart_current - open_to_cart_previous` | SKU + период | Может указывать на слабость карточки/оффера, но это прокси. |
| `cart_to_order_delta_pp` | Изменение перехода из корзины в заказ | `cart_to_order_current - cart_to_order_previous` | SKU + период | Может указывать на цену, доставку, наличие или доверие к карточке. |
| `cancel_rate` | Доля отмен в заказах | `cancel_units / nullif(orders_units, 0)` | SKU + период | Рост отмен снижает будущую реализацию; причину нужно проверять отдельно. |
| `buyout_rate` | Доля выкупов в заказах/воронке | `buyout_units / nullif(order_count, 0)` | SKU + период | Низкий выкуп влияет на качество продаж, но не объясняет спрос сам по себе. |
| `lost_sales_candidate` | SKU потерял продажи в текущем периоде | `previous_sales > 0 and current_sales = 0` | SKU + период | Кандидат на проверку остатков, цены, рекламы и карточки. |
| `new_sales_candidate` | SKU появился в текущих продажах | `previous_sales = 0 and current_sales > 0` | SKU + период | Кандидат на масштабирование после проверки маржи, остатков и рекламы. |

## Marketing Agent

| Признак | Что показывает | Как считается | Grain | Как читать |
| --- | --- | --- | --- | --- |
| `ad_spend_delta_pct` | Изменение рекламных затрат | `(spend_current - spend_previous) / nullif(spend_previous, 0)` | Кампания/SKU + период | Рост затрат без роста заказов - риск неэффективности. |
| `ad_orders_delta_pct` | Изменение заказов из рекламного кабинета | `(ad_orders_current - ad_orders_previous) / nullif(ad_orders_previous, 0)` | Кампания/SKU + период | Это кабинетная атрибуция, не все продажи SKU. |
| `ctr_delta_pp` | Изменение кликабельности | `ctr_current - ctr_previous` | Кампания/SKU + период | Падение CTR может говорить о креативе, позиции или релевантности. |
| `cpc_delta_pct` | Изменение цены клика | `(cpc_current - cpc_previous) / nullif(cpc_previous, 0)` | Кампания/SKU + период | Рост CPC при тех же заказах ухудшает экономику рекламы. |
| `click_to_order_delta_pp` | Изменение конверсии клика в заказ | `cr_current - cr_previous` | Кампания/SKU + период | Падение CR часто требует price/sales/card check. |
| `wasted_spend_flag` | Расход есть, заказов нет | `ad_spend > 0 and ad_orders = 0` | Кампания/SKU + период | Кандидат на снижение ставки/остановку только после проверки периода и объема кликов. |
| `cost_per_ad_order` | Стоимость рекламного заказа | `ad_spend / nullif(ad_orders, 0)` | Кампания/SKU + период | Использовать как сигнал, не как CAC. |
| `drr_promoted_sku` | Доля рекламных расходов в выручке продвигаемых SKU | `ad_spend / nullif(sales_revenue_promoted_sku, 0)` | SKU/магазин + период | Требует sales-выручку, не считать только из рекламного кабинета. |
| `visibility_delta_pp` | Изменение видимости в поисковом отчете | `visibility_current - visibility_previous` | SKU/query + период | Падение видимости может объяснять меньше трафика, но не маржу. |
| `high_frequency_low_order_flag` | Частотный запрос не дает заказы | `frequency >= threshold and orders_count = 0` | Query/SKU + период | Кандидат на минусацию, карточку или нерелевантный запрос. |

## Price Agent

| Признак | Что показывает | Как считается | Grain | Как читать |
| --- | --- | --- | --- | --- |
| `price_gap_pct` | Насколько наша цена выше/ниже медианы конкурентов | `(our_price - competitor_median_price) / nullif(competitor_median_price, 0)` | SKU + run_date | Положительный gap - дороже рынка; влияние на продажи требует sales-check. |
| `competitor_cheaper_share` | Доля конкурентов дешевле нас | `competitor_cheaper_count / nullif(competitor_count, 0)` | SKU + run_date | Высокая доля - ценовое давление. |
| `competitor_pricier_share` | Доля конкурентов дороже нас | `competitor_pricier_count / nullif(competitor_count, 0)` | SKU + run_date | Высокая доля может давать пространство для повышения, но нужна маржа/спрос. |
| `price_segment_match_share` | Доля конкурентов в том же ценовом сегменте | `same_segment_count / nullif(competitor_count, 0)` | SKU + run_date | Низкая доля значит сравнение может быть шумным. |
| `thin_competitor_data_flag` | Мало конкурентов для вывода | `competitor_count < threshold` | SKU + run_date | Снижает confidence по ценовому выводу. |
| `relevance_confirmed_share` | Доля подтвержденных релевантных конкурентов | `relevant_competitor_count / nullif(competitor_count, 0)` | SKU + run_date | Низкая доля - конкуренты кандидаты, не жесткое доказательство. |
| `promo_discount_depth` | Глубина нашей скидки | `(regular_price - promo_price) / nullif(regular_price, 0)` | SKU + snapshot_date | Помогает понять, цена изменилась скидкой или регулярной ценой. |
| `club_price_gap_pct` | Разрыв клубной цены к рынку | `(club_price - competitor_median_price) / nullif(competitor_median_price, 0)` | SKU + snapshot_date | Важно, если WB-клуб влияет на фактическую видимую цену. |
| `price_change_since_previous_pct` | Изменение нашей цены между снимками | `(price_current - price_previous) / nullif(price_previous, 0)` | SKU + дата | Если совпало с падением заказов, это гипотеза, не доказанная причина. |
| `safe_discount_gap_to_margin` | Запас до минимально безопасной цены | `(promo_price - min_profitable_price) / nullif(promo_price, 0)` | SKU + период | Требует finance-данные; без них price-agent не должен подтверждать. |

## Supply Agent

| Признак | Что показывает | Как считается | Grain | Как читать |
| --- | --- | --- | --- | --- |
| `days_of_stock` | На сколько дней хватит sellable stock | `stock_qty / nullif(avg_daily_sales_7d, 0)` | SKU + snapshot_date | Низкое значение - риск доступности. |
| `stockout_flag` | Товар без доступного остатка | `stock_qty = 0` | SKU + snapshot_date | Может быть причиной падения только если в нужные даты были продажи/спрос. |
| `critical_coverage_flag` | Критически низкое покрытие | `sales_7d_units > 0 and days_of_stock < 7` | SKU + snapshot_date | Сильный сигнал риска дефицита. |
| `warning_coverage_flag` | Предупреждение по покрытию | `sales_7d_units > 0 and days_of_stock between 7 and 21` | SKU + snapshot_date | Нужен план пополнения, но не паника. |
| `reorder_qty_recommended` | Рекомендуемый дозаказ | `greatest(ceil(avg_daily_sales_7d * target_days - stock_qty), 0)` | SKU + snapshot_date | Target days задается политикой: например 21 или 56 дней. |
| `zero_sales_with_stock_flag` | Остаток есть, продаж нет | `stock_qty > 0 and sales_14d_units = 0` | SKU + snapshot_date | Проверять цену, карточку, рекламу или спрос. |
| `overstock_flag` | Потенциальный избыток | `stock_qty >= threshold and sales_7d_units = 0` | SKU + snapshot_date | Распродажа допустима только после price/finance-check. |
| `stock_concentration_share` | Доля остатка на крупнейшем складе | `top_warehouse_stock_qty / nullif(stock_qty, 0)` | SKU + snapshot_date | Высокая концентрация может ограничивать региональную доступность. |
| `in_way_pressure` | Сколько товара уже в пути к/от клиента | `(in_way_to_client + in_way_from_client) / nullif(stock_qty_full, 0)` | SKU + snapshot_date | Помогает не завышать доступность полным остатком. |
| `inbound_readiness_rate` | Готовность поставки к продаже | `ready_for_sale_quantity / nullif(quantity, 0)` | Поставка/SKU + snapshot_date | Низкая готовность - поставка еще не закрывает дефицит. |

## Finance Agent

| Признак | Что показывает | Как считается | Grain | Как читать |
| --- | --- | --- | --- | --- |
| `net_revenue` | Выручка после возвратов | `gross_revenue - returns` | SKU + период | База для маржи и долей расходов. |
| `marketplace_cost_share` | Доля расходов МП в нетто-выручке | `marketplace_costs_total / nullif(net_revenue, 0)` | SKU + период | Рост доли расходов ухудшает экономику. |
| `commission_share` | Доля комиссии и эквайринга | `(commission + acquiring) / nullif(net_revenue, 0)` | SKU + период | Показывает давление платформенных расходов. |
| `logistics_share` | Доля логистики | `logistics / nullif(net_revenue, 0)` | SKU + период | Полезно для тяжелых/дорогих в доставке товаров. |
| `storage_share` | Доля хранения | `storage / nullif(net_revenue, 0)` | SKU + период | Рост хранения может усиливать риск overstock. |
| `cogs_coverage_rate` | Покрытие строк себестоимостью | `costed_qty / nullif(total_sale_return_qty, 0)` | SKU + период | Если меньше 1, чистую маржу нельзя подтверждать полностью. |
| `gross_margin_pct` | Валовая маржа после COGS | `(net_revenue - cogs_total) / nullif(net_revenue, 0)` | SKU + период | Требует полного COGS; иначе `нет данных`. |
| `contribution_margin_pct` | Маржа после COGS и расходов МП | `(net_revenue - cogs_total - marketplace_costs_total) / nullif(net_revenue, 0)` | SKU + период | Главный финансовый признак, если COGS покрыт. |
| `margin_delta_abs` | Изменение маржи между периодами | `margin_current - margin_previous` | SKU + период | Показывает механический вклад SKU в прибыль. |
| `safe_price_floor` | Минимальная цена до нулевой contribution margin | `(cogs_per_unit + marketplace_cost_per_unit) / nullif(1 - variable_rate, 0)` | SKU + сценарий | Это сценарный расчет; требует явных допущений и проверки price. |

## Общий формат ответа агента по признаку

Агенту лучше возвращать не только число, а короткую карточку проверки:

| Поле | Пример |
| --- | --- |
| Гипотеза | `Цена выше рынка могла снизить заказы` |
| Признак | `price_gap_pct` |
| Формула | `(our_price - competitor_median_price) / competitor_median_price` |
| Источник | `analytics_ai.wb_price_ai_competitors_raw`, `pricing.wb_supplier_prices` |
| Grain | `SKU + run_date` |
| Период | `2026-04-15` |
| Результат | `+18.4%` |
| Статус | `частично подтверждено` |
| Ограничение | `Влияние на продажи требует sales-check` |

## Минимальная вставка в промт агента

```text
Для аналитических запросов работай циклом:
гипотеза -> производный признак -> расчет по данным -> статус -> рекомендация.

Ты можешь создавать временные производные признаки внутри read-only SQL через
CTE, CASE, lag/lead, rolling windows, ratios и классификации. Не сохраняй признаки
в БД.

Для каждого важного признака укажи, что он проверяет, формулу, источник данных,
grain, период, результат и статус: подтверждено, частично подтверждено,
не подтверждено или нет данных.

Не называй производный признак причиной сам по себе. Если причинность зависит от
другого домена, создай cross_agent_requests.
```
