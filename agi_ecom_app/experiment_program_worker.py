from __future__ import annotations

import os
import threading


def main() -> None:
    from rq import SimpleWorker

    from agi_ecom_app.experiment_program.jobs import (
        build_experiment_program_queue,
        run_program_outbox_dispatcher,
    )
    from agi_ecom_app.services.memory import (
        close_memory_connection_pool,
        ensure_memory_schema,
    )

    ensure_memory_schema()
    os.environ["MEMORY_SCHEMA_PREINITIALIZED"] = "1"
    close_memory_connection_pool()

    queue = build_experiment_program_queue()
    stop_event = threading.Event()
    dispatcher = threading.Thread(
        target=run_program_outbox_dispatcher,
        kwargs={"stop_event": stop_event, "queue": queue},
        name="experiment-program-outbox",
        daemon=True,
    )
    dispatcher.start()
    try:
        # SimpleWorker avoids forking a process that owns the LISTEN thread.
        SimpleWorker([queue], connection=queue.connection).work(with_scheduler=True)
    finally:
        stop_event.set()
        dispatcher.join(timeout=5)
        close_memory_connection_pool()


if __name__ == "__main__":
    main()
