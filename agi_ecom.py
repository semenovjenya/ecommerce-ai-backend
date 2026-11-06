from __future__ import annotations

import argparse
import sys
import time
import random
import os
import re
import shutil
import datetime
import threading
from pyfiglet import Figlet
from rich import box
from rich.columns import Columns
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from termcolor import colored

from agi_ecom_app.api_models import ActivePromosResponse, DashboardResponse, FlashSaleWinterResponse
from agi_ecom_app.backend_client import BackendClient, BackendClientError

console = Console()


def get_backend_client() -> BackendClient:
    return BackendClient()

# --- UTILS ---
def clear_screen():
    if sys.stdout.isatty():
        console.clear(home=True)
        sys.stdout.write("\033[3J")
        sys.stdout.flush()
        return
    os.system('cls' if os.name == 'nt' else 'clear')


def restart_app():
    """Restart the current CLI process with the same Python executable and args."""
    os.execv(sys.executable, [sys.executable, *sys.argv])


def render_markdown_answer(answer: str):
    """Render agent output as Markdown in the terminal, with plain-text fallback."""
    try:
        if answer.lstrip().startswith("ЧТО ДЕЛАТЬ СЕЙЧАС"):
            console.print(Text(answer, style="white"))
            return
        formatted = widen_nested_list_indent(answer)
        formatted = add_spacing_after_headings(formatted)
        console.print(Markdown(formatted))
    except Exception:
        print(colored(answer, "white"))


def widen_nested_list_indent(text: str) -> str:
    """Add a bit more space to nested markdown list items for terminal readability."""
    widened_lines = []
    bullet_pattern = re.compile(r"^(?P<indent>\s+)(?P<marker>[-*+])\s+")

    for line in text.splitlines():
        match = bullet_pattern.match(line)
        if match and len(match.group("indent")) >= 2:
            line = "  " + line
        widened_lines.append(line)

    return "\n".join(widened_lines)


def add_spacing_after_headings(text: str) -> str:
    """Insert an extra blank line after markdown headings before body text."""
    lines = text.splitlines()
    spaced_lines = []

    for index, line in enumerate(lines):
        spaced_lines.append(line)
        is_heading = bool(re.match(r"^\s{0,3}#{1,6}\s+\S", line))
        next_line = lines[index + 1] if index + 1 < len(lines) else None

        if is_heading and next_line is not None and next_line.strip():
            spaced_lines.append("")

    return "\n".join(spaced_lines)

def type_text(text, delay=0.01, color='white', attrs=None):
    """Prints text with a typing effect."""
    sys.stdout.write(colored(text, color, attrs=attrs))
    sys.stdout.flush()
    time.sleep(delay * len(text) * 0.5) 
    print("")


def start_scanning_status(text, base_color="#556b7c", accent_color="#9fd7ff", interval=0.08):
    """Animate a left-aligned status line with a moving highlight."""
    display_width = len(text)
    stop_event = threading.Event()

    def _animate():
        index = 0
        highlight_width = 6
        while not stop_event.is_set():
            styled = Text()
            for pos, char in enumerate(text):
                distance = (pos - index) % display_width
                style = accent_color if distance < highlight_width else base_color
                styled.append(char, style=style)
            console.print(styled, end="\r")
            index = (index + 1) % display_width
            time.sleep(interval)

        sys.stdout.write("\r" + " " * display_width + "\r")
        sys.stdout.flush()

    worker = threading.Thread(target=_animate, daemon=True)
    worker.start()
    return stop_event, worker


def stop_status_animation(stop_event, worker):
    stop_event.set()
    worker.join(timeout=1.0)
    print("")


def build_command_prompt(letter_phase=0, arrow_phase=0) -> Text:
    prompt = Text()
    letter_palette = ["#8a6d12", "#c79a1f", "#f2cf63", "#c79a1f"]
    arrow_palette = ["#8a6d12", "#b79025", "#d4af37", "#ffe08a", "#d4af37", "#b79025"]
    letter_style = f"bold {letter_palette[letter_phase % len(letter_palette)]}"

    for char in "AI-ECOM":
        if char.isalpha() or char == "-":
            prompt.append(char, style=letter_style)
        else:
            prompt.append(char, style=letter_style)

    prompt.append(" ", style="bold")
    prompt.append(">>>", style=f"bold {arrow_palette[arrow_phase % len(arrow_palette)]}")
    prompt.append(" ", style="bold")
    return prompt


def render_command_prompt(buffer_text: str, letter_phase: int, arrow_phase: int) -> None:
    sys.stdout.write("\r\033[2K")
    sys.stdout.flush()
    console.print(build_command_prompt(letter_phase, arrow_phase), end="")
    sys.stdout.write(buffer_text)
    sys.stdout.flush()


def read_command_prompt() -> str:
    console.print(build_command_prompt(letter_phase=1, arrow_phase=2), end="")
    return input()



def terminal_width():
    return shutil.get_terminal_size((100, 24)).columns


def loading_bar(label, duration=1.0, width=None, color='magenta', label_width=30):
    """Renders a progress bar that takes 'duration' seconds to complete."""
    term_width = terminal_width()
    bar_width = width or max(18, min(34, term_width - label_width - 8))

    sys.stdout.write(colored(f"{label:<{label_width}} ", 'white') + "[")
    sys.stdout.flush()
    
    step_time = duration / bar_width
    chars = "█"
    
    for _ in range(bar_width):
        time.sleep(random.uniform(step_time * 0.8, step_time * 1.2))
        sys.stdout.write(colored(chars, color))
        sys.stdout.flush()
    
    sys.stdout.write("] " + colored("OK", 'green') + "\n")

def print_separator(char="─", length=None, color='dark_grey'):
    if length is None:
        length = min(terminal_width() - 4, 70)
    print(colored(char * length, color))

def print_header(text):
    print("\n" + colored(f" {text} ", 'black', 'on_cyan', attrs=['bold']))

def colored_val(val, type='neutral'):
    if type == 'pos': return colored(val, 'green')
    if type == 'neg': return colored(val, 'red')
    if type == 'warn': return colored(val, 'yellow')
    return colored(val, 'white')


def format_dashboard_int(value):
    if value is None:
        return "Н/Д"
    return f"{int(value):,}".replace(",", " ")


def format_dashboard_currency(value):
    if value is None:
        return "Н/Д"
    return f"{round(float(value)):,}".replace(",", " ") + " ₽"


def format_dashboard_delta(value):
    if value is None:
        return "Н/Д"
    return f"{float(value):+.2f}%"


def format_full_date(value):
    if value is None:
        return "данные недоступны"
    return value.strftime("%d.%m.%Y")


def format_short_date(value):
    if value is None:
        return "Н/Д"
    return value.strftime("%d.%m")


def format_snapshot_hour(value):
    if value is None:
        return "данные недоступны"
    local_tz = datetime.datetime.now().astimezone().tzinfo
    if value.tzinfo is not None and local_tz is not None:
        value = value.astimezone(local_tz)
    return value.strftime("%d.%m.%Y %H:%M")


def dashboard_value_style(kind, value):
    if kind != "delta":
        return "bold white"
    if value is None:
        return "bold yellow"
    if value > 0:
        return "bold green"
    if value < 0:
        return "bold red"
    return "bold #8a8f98"


def dashboard_border_style(kind, value):
    if kind == "sales_units":
        return "#00d1d9"
    if kind == "sales_rub":
        return "#7c9cff"
    if kind == "orders":
        return "#d4af37"
    if kind == "transit":
        return "#3fbf8f"
    if value is None:
        return "#7a7f87"
    if value > 0:
        return "green"
    if value < 0:
        return "red"
    return "#7a7f87"


def build_kpi_panel(title, value, subtitle, kind, raw_value=None, width=None):
    border_style = dashboard_border_style(kind, raw_value)
    title_text = Text(title.upper(), style=f"bold {border_style}")
    value_text = Text(value, style=dashboard_value_style(kind, raw_value))
    subtitle_text = Text(subtitle, style="dim")
    accent_line = Text("━" * 12, style=border_style)

    body = Group(title_text, accent_line, value_text, subtitle_text)
    return Panel(
        body,
        border_style=border_style,
        box=box.ROUNDED,
        padding=(1, 1),
        width=width,
    )


def build_dashboard_panels(kpis: DashboardResponse, card_width):
    sales_compare_label = (
        f"{format_short_date(kpis.sales_date)} vs {format_short_date(kpis.prev_sales_date)}"
        if kpis.sales_date and kpis.prev_sales_date
        else "данные недоступны"
    )

    return [
        build_kpi_panel(
            "Продажи шт",
            format_dashboard_int(kpis.sales_units),
            f"за {format_full_date(kpis.sales_date)}",
            "sales_units",
            kpis.sales_units,
            width=card_width,
        ),
        build_kpi_panel(
            "Продажи Руб",
            format_dashboard_currency(kpis.sales_rub),
            f"for_pay за {format_full_date(kpis.sales_date)}",
            "sales_rub",
            kpis.sales_rub,
            width=card_width,
        ),
        build_kpi_panel(
            "Заказы сегодня",
            format_dashboard_int(kpis.orders_today),
            f"по срезу {format_full_date(kpis.orders_date)}",
            "orders",
            kpis.orders_today,
            width=card_width,
        ),
        build_kpi_panel(
            "Динамика Продаж вчера к позавчера шт",
            format_dashboard_delta(kpis.sales_units_delta_pct),
            sales_compare_label,
            "delta",
            kpis.sales_units_delta_pct,
            width=card_width,
        ),
        build_kpi_panel(
            "Динамика продаж вчера к позавчера руб",
            format_dashboard_delta(kpis.sales_rub_delta_pct),
            sales_compare_label,
            "delta",
            kpis.sales_rub_delta_pct,
            width=card_width,
        ),
        build_kpi_panel(
            "Товара в пути к клиенту",
            format_dashboard_int(kpis.in_way_to_client_units),
            f"срез {format_snapshot_hour(kpis.stocks_snapshot_hour)}",
            "transit",
            kpis.in_way_to_client_units,
            width=card_width,
        ),
    ]


def render_dashboard_grid(kpis: DashboardResponse):
    width = terminal_width()
    columns = 3 if width >= 96 else 2 if width >= 64 else 1
    card_width = max(28, (width - ((columns - 1) * 2)) // columns)
    panels = build_dashboard_panels(kpis, card_width)

    for start in range(0, len(panels), columns):
        console.print(
            Columns(
                panels[start:start + columns],
                equal=True,
                expand=True,
                padding=(0, 1),
            )
        )
        print("")

# --- COMPONENTS ---

def startup_sequence():
    clear_screen()
    
    f = Figlet(font='ansi_shadow')
    console.print(f.renderText('AI-ECOM').rstrip("\n"), style="bold #c38bf2")
    console.print("AUTONOMOUS E-COMMERCE INTELLIGENCE", style="#4e6576")
    
    type_text(">>> УСТАНОВЛЕНИЕ ЗАЩИЩЕННОГО СОЕДИНЕНИЯ...", delay=0.01, color='dark_grey')
    time.sleep(0.2)
    
    loading_steps = [
        ("ПОДКЛЮЧЕНИЕ К БАЗЕ ДАННЫХ", 0.9),
        ("СИНХРОНИЗАЦИЯ ПРОДАЖ", 0.7),
        ("АНАЛИЗ КОНКУРЕНТОВ", 1.0),
        ("АНАЛИЗ ЦЕН", 0.7),
        ("СИСТЕМА РЕКОМЕНДАЦИЙ", 0.9),
    ]

    for label, duration in loading_steps:
        loading_bar(label, duration=duration, color='cyan', label_width=30)
    
    print("")
    type_text("[СИСТЕМА ГОТОВА]", delay=0.015, color='green')
    time.sleep(0.2)
    clear_screen()

def render_daily_report():
    today = datetime.date.today().strftime("%d.%m.%Y")
    
    print_separator("═")
    print(colored(f"  ЕЖЕДНЕВНЫЙ ОТЧЕТ  |  ДАТА: {today}  |  СТАТУС: НОМИНАЛЬНЫЙ", 'white', attrs=['bold']))
    print_separator("═")

    try:
        kpis = get_backend_client().fetch_daily_dashboard()
    except BackendClientError:
        kpis = DashboardResponse(
            sales_units=None,
            sales_rub=None,
            orders_today=None,
            sales_units_delta_pct=None,
            sales_rub_delta_pct=None,
            in_way_to_client_units=None,
            sales_date=None,
            prev_sales_date=None,
            orders_date=None,
            stocks_snapshot_hour=None,
            is_partial=True,
            warning="Данные KPI временно недоступны",
        )

    if kpis.warning:
        console.print(Text(f" [!] {kpis.warning}", style="bold yellow"))
        print("")

    render_dashboard_grid(kpis)


def resolve_chat_id(explicit_chat_id: str | None) -> str:
    if explicit_chat_id:
        return explicit_chat_id
    return get_backend_client().create_chat_session().chat_id


def check_backend_health():
    return get_backend_client().check_backend_health()


def render_chat_session_status(chat_id: str):
    console.print(Text(f" CHAT ID: {chat_id}", style="bold cyan"))
    print("")


def render_backend_health_warning():
    try:
        health = check_backend_health()
    except BackendClientError as exc:
        raise BackendClientError(f"Не удалось подключиться к backend API: {exc}") from exc

    if health.status == "healthy":
        return

    dependency_lines = []
    for label, dependency in (
        ("memory_db", health.memory_db),
        ("analytics_db", health.analytics_db),
        ("openai_config", health.openai_config),
    ):
        if dependency.status != "healthy":
            dependency_lines.append(f"{label}: {dependency.detail or dependency.status}")

    if dependency_lines:
        warning = "Backend в degraded-режиме: " + "; ".join(dependency_lines)
        console.print(Text(f" [!] {warning}", style="bold yellow"))
        print("")

# --- PROMO COMPONENT ---

def render_active_promos(promos_response: ActivePromosResponse):
    print_header("АКТИВНЫЕ ПРОМО-КАМПАНИИ (5)")

    print(colored(f"{'НАЗВАНИЕ':<25} | {'ЭФФЕКТИВНОСТЬ':<15} | {'SKU':<8} | {'АЛЕРТЫ'}", 'dark_grey'))
    print_separator("-")

    for promo in promos_response.items:
        eff_colored = colored_val(promo.eff, promo.status)
        alert_colored = colored_val(
            promo.alerts,
            promo.status if promo.alerts != "Норма" else 'neutral',
        )
        print(f" {promo.name:<24} | {eff_colored:<24} | {promo.sku:<7} | {alert_colored}")

    print_separator("-")
    print(f"\n {colored('ИТОГО:', 'magenta', attrs=['bold'])} {len(promos_response.items)} активных кампаний")
    print("\n")

def render_promo_flash_sale_winter(report: FlashSaleWinterResponse):
    print_header("ДЕТАЛЬНАЯ АНАЛИТИКА: РАСПРОДАЖА -15%")

    type_text(f">>> {report.analyzed_items_text}", delay=0.02, color='dark_grey')
    time.sleep(0.5)

    # --- ОБЩИЙ РЕЗУЛЬТАТ ---
    print(colored("\n 📊 ОБЩИЙ РЕЗУЛЬТАТ АКЦИИ (7 дней):", 'white', attrs=['bold']))
    print_separator("-")

    print(f" {'Влияние на план:':<22} {colored_val(report.impact_on_plan, report.impact_on_plan_status)}")
    print(f" {'Выручка по акции:':<22} {colored_val(report.revenue_text, report.revenue_status)}")
    print(f" {'Чистая прибыль:':<22} {colored_val(report.profit_text, report.profit_status)}")
    print(f" {'Статус плана:':<22} {colored_val(report.plan_status_text, report.plan_status_kind)}")

    print(f"\n {'Маржинальность:':<22} {colored_val(report.margin_text, report.margin_status)}")
    print(f" {'ДРР:':<22} {colored_val(report.drr_text, report.drr_status)}")
    print(f" {'Оборачиваемость:':<22} {colored_val(report.turnover_text, report.turnover_status)}")
    print(f" {'Экономия хранения:':<22} {colored_val(report.storage_savings_text, report.storage_savings_status)}")

    # --- ДЕТАЛИ ПО SKU ---
    print(colored("\n ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", 'dark_grey'))
    print(colored(" ДЕТАЛИ ПО КЛЮЧЕВЫМ SKU:", 'cyan', attrs=['bold']))
    print(colored(" ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", 'dark_grey'))

    for sku in report.sku_details:
        sku_label = colored(sku.sku_label, 'white')
        print(
            f"\n [{colored(sku.badge, sku.badge_color)}] {sku_label} → "
            f"{colored(sku.action_text, sku.action_color)}"
        )
        print(f"    {'Продажи:':<12} {colored_val(sku.sales, sku.sales_status)}")
        print(f"    {'Позиция:':<12} {colored_val(sku.position, sku.position_status)}")
        print(f"    {'Маржа:':<12} {colored_val(sku.margin, sku.margin_status)}")
        print(f"    {'Остатки:':<12} {colored_val(sku.stock, sku.stock_status)}")
        if sku.note:
            print(colored(f"    → {sku.note}", sku.note_color or 'magenta', attrs=['bold']))

    print(colored("\n ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", 'dark_grey'))
    print(f" {colored('РЕКОМЕНДАЦИЯ:', 'magenta', attrs=['bold'])} {report.recommendation_text}\n")


def render_agent_response(user_input: str, chat_id: str):
    header = "CEO ОРКЕСТРАТОР"
    status_text = ">>> Backend оркестрирует ответ через данные, агентов и CEO..."
    print("")
    print_header(header)
    animation_stop, animation_worker = start_scanning_status(
        status_text,
    )

    try:
        client = get_backend_client()
        response = client.send_chat_message(chat_id=chat_id, message=user_input)
        while response.status in {"queued", "running"} and response.run_id is not None:
            time.sleep(2)
            status_response = client.fetch_agent_run_status(
                run_id=response.run_id,
                chat_id=chat_id,
            )
            response.answer = status_response.answer
            response.server_message_id = status_response.server_message_id
            response.status = status_response.status
            if status_response.error:
                response.answer = status_response.error
        answer = response.answer or ""
        stop_status_animation(animation_stop, animation_worker)
        if response.status == "failed":
            print(colored(f" [!] {answer or 'Agent run failed'}", "red"))
        else:
            render_markdown_answer(answer)
    except BackendClientError as exc:
        stop_status_animation(animation_stop, animation_worker)
        print(colored(f" [!] {exc}", "red"))
    except Exception as exc:  # pragma: no cover - defensive UX path
        stop_status_animation(animation_stop, animation_worker)
        print(colored(f" [!] Ошибка агента: {exc}", "red"))

# --- MAIN ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-intro", action="store_true", help="Skip the startup animation")
    parser.add_argument("--chat-id", help="Continue an existing chat session by chat_id")
    args = parser.parse_args()

    # 1. Startup
    if not args.skip_intro:
        startup_sequence()

    try:
        render_backend_health_warning()
        chat_id = resolve_chat_id(args.chat_id)
    except BackendClientError as exc:
        print(colored(f" [!] {exc}", "red"))
        return

    render_daily_report()
    render_chat_session_status(chat_id)
    
    # 2. Command Loop
    while True:
        try:
            raw_cmd = read_command_prompt().strip()
            cmd = raw_cmd.lower()
            
            if not cmd:
                continue
                
            if cmd == "покажи активные промо":
                render_active_promos(get_backend_client().fetch_active_promos())
            elif "распродажа -15%" in cmd or "flash sale winter" in cmd:
                render_promo_flash_sale_winter(get_backend_client().fetch_flash_sale_winter())
            elif cmd in ["/clear", "clear", "cls"]:
                chat_id = resolve_chat_id(None)
                clear_screen()
                render_daily_report()
                render_chat_session_status(chat_id)
            elif cmd in ["/restart", "restart", "reload"]:
                type_text(">>> ПЕРЕЗАПУСК СЕАНСА...", color='dark_grey')
                restart_app()
            elif cmd in ["/report", "report", "отчет"]:
                render_daily_report()
            elif cmd in ["/exit", "exit", "quit", "выход", "stop"]:
                type_text(">>> ЗАВЕРШЕНИЕ СЕАНСА...", color='dark_grey')
                break
            elif cmd in ["/help", "help", "помощь", "?"]:
                print(colored("\n ДОСТУПНЫЕ КОМАНДЫ:", 'cyan'))
                print(" • 'Покажи активные промо'     - список текущих акций")
                print(" • 'распродажа -15%'           - детализация конкретной акции")
                print(" • '/help'                     - показать список команд")
                print(" • '/clear'                    - очистить экран и создать новый chat_id")
                print(" • '/restart'                  - перезапустить приложение")
                print(" • '/report'                   - показать ежедневный отчет")
                print(" • любой другой запрос         - отправить вопрос CEO orchestrator")
                print(" • '/exit'                     - закрыть терминал\n")
            else:
                render_agent_response(raw_cmd, chat_id=chat_id)
        except KeyboardInterrupt:
            print("\n")
            type_text(">>> АВАРИЙНОЕ ЗАВЕРШЕНИЕ...", color='red')
            break
        except EOFError:
            print("\n")
            type_text(">>> СЕАНС ЗАВЕРШЕН (EOF)...", color='dark_grey')
            break
        except BackendClientError as exc:
            print(colored(f" [!] {exc}", "red"))

if __name__ == "__main__":
    main()
