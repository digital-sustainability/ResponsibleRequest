"""Per-request cost and the budget guard."""

from __future__ import annotations

import openai
from loguru import logger

from .config import CostConfig, LogConfig
from .records import RequestRecord
from .sources import open_source


class BudgetExceeded(openai.OpenAIError):
    """Raised instead of sending a request once ``CostConfig.budget_usd`` has been spent."""

    def __init__(self, spent_usd: float, budget_usd: float) -> None:
        super().__init__(f"budget of ${budget_usd:.4f} exhausted (${spent_usd:.4f} spent)")
        self.spent_usd = spent_usd
        self.budget_usd = budget_usd


class CostTracker:
    """Adds up the cost of the requests of one client. Available as ``client.cost``."""

    def __init__(self, config: CostConfig, log: LogConfig | None = None) -> None:
        self.config = config
        self.logged_usd = 0.0  # spent before this client was created, according to the log
        if config.include_logged:
            source = open_source(log)
            if source is not None:
                try:
                    self.logged_usd = source.total_cost()
                finally:
                    source.close()
        self.session_usd = 0.0
        self.priced = 0
        self.unpriced = 0
        self._warned: set[str | None] = set()
        self._exhausted_logged = False

    @property
    def spent_usd(self) -> float:
        return self.logged_usd + self.session_usd

    @property
    def remaining_usd(self) -> float | None:
        if self.config.budget_usd is None:
            return None
        return max(0.0, self.config.budget_usd - self.spent_usd)

    def check(self) -> None:
        """Raise :class:`BudgetExceeded` if the budget is exhausted."""
        budget = self.config.budget_usd
        if budget is None or self.spent_usd < budget:
            return
        if not self._exhausted_logged:
            self._exhausted_logged = True
            logger.warning(
                "budget of ${:.4f} exhausted (${:.4f} spent): refusing new requests",
                budget,
                self.spent_usd,
            )
        raise BudgetExceeded(self.spent_usd, budget)

    def add(self, record: RequestRecord) -> None:
        """Fill in ``record.cost_usd`` if the provider did not report it, and count it."""
        if record.cost_usd is None:
            price = self.config.prices.get(record.model or "")
            if price is None and record.response_model:
                price = self.config.prices.get(record.response_model)
            if price is not None and record.prompt_tokens is not None:
                record.cost_usd = price.cost(
                    record.prompt_tokens, record.completion_tokens, record.cached_tokens
                )
        if record.cost_usd is not None:
            self.priced += 1
            self.session_usd += record.cost_usd
        elif record.status_code == 200:
            self.unpriced += 1
            if self.config.budget_usd is not None and record.model not in self._warned:
                self._warned.add(record.model)
                logger.warning(
                    "{}: no cost reported and no price configured; the budget does not see "
                    "these requests (set CostConfig(prices=...))",
                    record.model or record.path,
                )

    def stats(self) -> dict[str, float | int | None]:
        return {
            "spent_usd": round(self.spent_usd, 6),
            "session_usd": round(self.session_usd, 6),
            "budget_usd": self.config.budget_usd,
            "remaining_usd": self.remaining_usd,
            "priced": self.priced,
            "unpriced": self.unpriced,
        }
