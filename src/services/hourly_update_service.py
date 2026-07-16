"""Background collection and ML refresh aligned to Hanoi clock hours."""

from __future__ import annotations

import json
import logging
import os
import threading
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

from dotenv import load_dotenv


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATUS_PATH = PROJECT_ROOT / "artifacts" / "hourly_update_status.json"


def _environment_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def next_hourly_run(
    now: datetime,
    *,
    timezone_name: str = "Asia/Ho_Chi_Minh",
    delay_seconds: int = 45,
) -> datetime:
    """Return the next top-of-hour target plus a small provider-ready delay."""

    zone = ZoneInfo(timezone_name)
    local = now.astimezone(zone)
    hour_start = local.replace(minute=0, second=0, microsecond=0)
    target = hour_start + timedelta(seconds=max(0, delay_seconds))
    if local >= target:
        target += timedelta(hours=1)
    return target


class HourlyUpdateService:
    """Run collectors once at startup and continuously at each Hanoi hour."""

    def __init__(
        self,
        collection_runner: Callable[[], dict[str, int]],
        *,
        forecast_refresher: Callable[[], dict[str, Any]] | None = None,
        enabled: bool = True,
        run_on_startup: bool = True,
        timezone_name: str = "Asia/Ho_Chi_Minh",
        delay_seconds: int = 45,
        retry_minutes: int = 5,
        status_path: str | Path = DEFAULT_STATUS_PATH,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.collection_runner = collection_runner
        self.forecast_refresher = forecast_refresher
        self.enabled = bool(enabled)
        self.run_on_startup = bool(run_on_startup)
        self.timezone_name = timezone_name
        self.delay_seconds = max(0, min(int(delay_seconds), 3_599))
        self.retry_minutes = max(1, int(retry_minutes))
        self.status_path = Path(status_path)
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._state_lock = threading.RLock()
        self._run_lock = threading.Lock()
        self._wake_event = threading.Event()
        self._stop_event = threading.Event()
        self._manual_requested = False
        self._thread: threading.Thread | None = None
        self._state: dict[str, Any] = {
            "enabled": self.enabled,
            "running": False,
            "schedule": "hourly",
            "timezone": self.timezone_name,
            "delay_seconds": self.delay_seconds,
            "run_on_startup": self.run_on_startup,
            "last_started_at": None,
            "last_completed_at": None,
            "last_success_at": None,
            "next_run_at": None,
            "last_trigger": None,
            "last_result": None,
            "last_error": None,
            "consecutive_failures": 0,
        }
        self._restore_previous_status()

    @classmethod
    def from_environment(
        cls,
        *,
        forecast_refresher: Callable[[], dict[str, Any]] | None = None,
    ) -> "HourlyUpdateService":
        load_dotenv(override=False)
        config_path = Path(os.getenv("HOURLY_UPDATE_CONFIG_PATH", "config.yaml"))

        def run_collection_from_config() -> dict[str, int]:
            from scripts.run_collection import load_collection_config, run_collection

            config = load_collection_config(config_path)
            return run_collection(config)

        under_pytest = "PYTEST_CURRENT_TEST" in os.environ
        return cls(
            run_collection_from_config,
            forecast_refresher=forecast_refresher,
            enabled=_environment_flag("HOURLY_UPDATES_ENABLED", True) and not under_pytest,
            run_on_startup=_environment_flag("HOURLY_UPDATE_RUN_ON_STARTUP", True),
            timezone_name=os.getenv("HOURLY_UPDATE_TIMEZONE", "Asia/Ho_Chi_Minh"),
            delay_seconds=int(os.getenv("HOURLY_UPDATE_DELAY_SECONDS", "45")),
            retry_minutes=int(os.getenv("HOURLY_UPDATE_RETRY_MINUTES", "5")),
            status_path=os.getenv("HOURLY_UPDATE_STATUS_PATH", str(DEFAULT_STATUS_PATH)),
        )

    def _restore_previous_status(self) -> None:
        try:
            previous = json.loads(self.status_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return
        if not isinstance(previous, dict):
            return
        for key in (
            "last_started_at",
            "last_completed_at",
            "last_success_at",
            "last_trigger",
            "last_result",
            "last_error",
            "consecutive_failures",
        ):
            if key in previous:
                self._state[key] = previous[key]

    def _persist_status(self) -> None:
        try:
            self.status_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.status_path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(self._state, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(self.status_path)
        except OSError as error:
            LOGGER.warning("Could not persist hourly update status: %s", error)

    def _update_state(self, **values: Any) -> None:
        with self._state_lock:
            self._state.update(values)
            self._persist_status()

    def status(self) -> dict[str, Any]:
        with self._state_lock:
            result = deepcopy(self._state)
        result["worker_alive"] = bool(self._thread and self._thread.is_alive())
        return result

    def run_once(self, trigger: str = "manual") -> bool:
        """Run one complete collection/forecast cycle without overlapping a prior run."""

        if not self.enabled or not self._run_lock.acquire(blocking=False):
            return False
        started = self._now().astimezone(ZoneInfo(self.timezone_name))
        self._update_state(
            running=True,
            last_started_at=started.isoformat(),
            last_trigger=trigger,
            last_error=None,
        )
        try:
            collection = self.collection_runner()
            forecast = self.forecast_refresher() if self.forecast_refresher else None
            completed = self._now().astimezone(ZoneInfo(self.timezone_name))
            result: dict[str, Any] = {"collection": collection}
            if forecast is not None:
                result["forecast_refresh"] = forecast
            self._update_state(
                running=False,
                last_completed_at=completed.isoformat(),
                last_success_at=completed.isoformat(),
                last_result=result,
                last_error=None,
                consecutive_failures=0,
            )
            LOGGER.info("Hourly environment update completed: %s", result)
            return True
        except Exception as error:  # Provider failures must not terminate the worker.
            completed = self._now().astimezone(ZoneInfo(self.timezone_name))
            failures = int(self.status().get("consecutive_failures") or 0) + 1
            message = f"{type(error).__name__}: {error}"
            self._update_state(
                running=False,
                last_completed_at=completed.isoformat(),
                last_error=message[:800],
                consecutive_failures=failures,
            )
            LOGGER.exception("Hourly environment update failed")
            return False
        finally:
            self._run_lock.release()

    def trigger_now(self) -> bool:
        if not self.enabled or self.status()["running"]:
            return False
        with self._state_lock:
            self._manual_requested = True
        self._wake_event.set()
        return True

    def start(self) -> bool:
        if not self.enabled or (self._thread and self._thread.is_alive()):
            return False
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._worker_loop,
            name="environment-hourly-update",
            daemon=True,
        )
        self._thread.start()
        return True

    def stop(self, timeout_seconds: float = 10.0) -> None:
        self._stop_event.set()
        self._wake_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=max(0.0, timeout_seconds))

    def _worker_loop(self) -> None:
        if self.run_on_startup and not self._stop_event.is_set():
            self._run_with_retry("startup")

        while not self._stop_event.is_set():
            target = next_hourly_run(
                self._now(),
                timezone_name=self.timezone_name,
                delay_seconds=self.delay_seconds,
            )
            self._update_state(next_run_at=target.isoformat())
            while not self._stop_event.is_set():
                wait_seconds = max(0.0, (target - self._now().astimezone(target.tzinfo)).total_seconds())
                signalled = self._wake_event.wait(timeout=min(wait_seconds, 60.0))
                if self._stop_event.is_set():
                    return
                if signalled:
                    self._wake_event.clear()
                    with self._state_lock:
                        manual = self._manual_requested
                        self._manual_requested = False
                    if manual:
                        self._run_with_retry("manual")
                        break
                if wait_seconds <= 0:
                    self._run_with_retry("hourly")
                    break

    def _run_with_retry(self, trigger: str) -> bool:
        """Retry failed provider/model cycles without terminating the scheduler."""

        current_trigger = trigger
        while not self._stop_event.is_set():
            if self.run_once(current_trigger):
                return True
            retry_at = (
                self._now().astimezone(ZoneInfo(self.timezone_name))
                + timedelta(minutes=self.retry_minutes)
            )
            self._update_state(next_run_at=retry_at.isoformat())
            if self._stop_event.wait(self.retry_minutes * 60):
                return False
            current_trigger = "retry"
        return False
