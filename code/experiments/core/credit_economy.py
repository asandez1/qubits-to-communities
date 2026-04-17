"""
Credit economy service implementing bonding curves, treasury, and demurrage.

Bonding curve (Section 3.6.2):  P_C = P0 * (D_C / S_C)^k
Demurrage (Section 4.5):        d = min(d_max, d_max * (1 - V / V_target))
Treasury: 2.5% protocol fee, min 15% reserve ratio.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class PricePoint:
    cycle: int
    price: float
    supply: float
    demand: float
    ewma_supply: float


@dataclass
class TransactionResult:
    success: bool
    amount: float
    fee: float
    price: float
    category: str
    message: str = ""


@dataclass
class BridgeResult:
    success: bool
    credits_spent: float
    fiat_received: float
    discount_applied: float
    message: str = ""


@dataclass
class EconomyMetrics:
    cycle: int
    gini: float
    velocity: float
    total_supply: float
    total_transactions: int
    prices: Dict[str, float]
    match_rate: float
    treasury_fiat: float
    treasury_credits: float
    reserve_ratio: float
    per_category_supply: Dict[str, float]
    per_category_demand: Dict[str, float]


class BondingCurve:
    """Per-category automated market maker: P_C = P0 * (D_C / S_C)^k"""

    def __init__(
        self,
        category: str,
        base_price: float,
        steepness: float,
        ewma_alpha: float = 0.1,
    ):
        self.category = category
        self.base_price = base_price
        self.steepness = steepness
        self.ewma_alpha = ewma_alpha

        self._raw_supply: float = 1.0
        self._raw_demand: float = 1.0
        self._ewma_supply: float = 0.0  # will be seeded on first update
        self._initialized: bool = False
        self._history: List[PricePoint] = []

    def get_price(self) -> float:
        """Current spot price: P0 * (D / S_tilde)^k"""
        if not self._initialized:
            return self.base_price
        ratio = max(0.01, self._raw_demand) / max(0.01, self._ewma_supply)
        return self.base_price * (ratio ** self.steepness)

    def update(self, supply_count: int, demand_count: int, cycle: int):
        """Update supply/demand counts and EWMA for this cycle."""
        self._raw_supply = max(0.01, float(supply_count))
        self._raw_demand = max(0.01, float(demand_count))
        # Seed EWMA from first observation to avoid extreme initial prices
        if not self._initialized:
            self._ewma_supply = self._raw_supply
            self._initialized = True
        else:
            # EWMA smoothing: S_tilde(t) = alpha * S(t) + (1-alpha) * S_tilde(t-1)
            self._ewma_supply = (
                self.ewma_alpha * self._raw_supply
                + (1 - self.ewma_alpha) * self._ewma_supply
            )
        self._history.append(PricePoint(
            cycle=cycle,
            price=self.get_price(),
            supply=self._raw_supply,
            demand=self._raw_demand,
            ewma_supply=self._ewma_supply,
        ))

    def get_history(self) -> List[PricePoint]:
        return list(self._history)

    @property
    def supply(self) -> float:
        return self._raw_supply

    @property
    def demand(self) -> float:
        return self._raw_demand


class CreditEconomyService:
    """Manages all credit flows, balances, bonding curves, and treasury."""

    def __init__(
        self,
        categories: Dict[str, dict],
        protocol_fee: float = 0.025,
        ewma_alpha: float = 0.1,
        demurrage_max: float = 0.02,
        demurrage_velocity_target: float = 1.0,
        demurrage_grace_cycles: int = 180,
        treasury_fiat: float = 10000.0,
        min_reserve_ratio: float = 0.15,
        suspended_reserve_ratio: float = 0.05,
        bridge_config: Optional[dict] = None,
        demurrage_mode: str = "flat",
        fee_mode: str = "flat",
    ):
        self.protocol_fee = protocol_fee
        self.demurrage_max = demurrage_max
        self.demurrage_velocity_target = demurrage_velocity_target
        self.demurrage_grace_cycles = demurrage_grace_cycles
        self.min_reserve_ratio = min_reserve_ratio
        self.suspended_reserve_ratio = suspended_reserve_ratio
        self.demurrage_mode = demurrage_mode  # "flat" or "progressive"
        self.fee_mode = fee_mode  # "flat" or "progressive"

        # Redistribution pool: accumulates surplus from progressive fees
        self._redistribution_pool: float = 0.0

        # Compute mean base price for progressive fee thresholds
        self._mean_base_price = float(np.mean(
            [c["base_price"] for c in categories.values()]
        )) if categories else 5.0

        # Bonding curves per category
        self.curves: Dict[str, BondingCurve] = {}
        for cat_name, cat_config in categories.items():
            self.curves[cat_name] = BondingCurve(
                category=cat_name,
                base_price=cat_config["base_price"],
                steepness=cat_config["steepness"],
                ewma_alpha=ewma_alpha,
            )

        # Balances
        self._balances: Dict[str, float] = {}

        # Treasury
        self.treasury_fiat = treasury_fiat
        self.treasury_credits = 0.0

        # Bridge config
        self.bridge_config = bridge_config or {}
        self.bridge_vesting_cycles = self.bridge_config.get("bridge_vesting_cycles", 30)
        self.bridge_monthly_cap = self.bridge_config.get("bridge_monthly_cap", 100.0)
        self.bridge_discounts = self.bridge_config.get("bridge_progressive_discount", [
            {"threshold": 50, "discount": 0.0},
            {"threshold": 100, "discount": 0.10},
            {"threshold": 200, "discount": 0.30},
            {"threshold": 500, "discount": 0.60},
        ])

        # Tracking
        self._transaction_count: int = 0
        self._cycle_transactions: int = 0
        self._total_credits_issued: float = 0.0
        self._cycle_matched: int = 0
        self._cycle_total_requests: int = 0
        self._bridge_redemptions: Dict[str, float] = {}  # member_id -> total redeemed

        # Per-cycle velocity tracking
        self._active_balances_sum: float = 0.0
        self._cycle_transfer_volume: float = 0.0

    def set_balance(self, member_id: str, amount: float):
        """Set initial balance for a member."""
        self._balances[member_id] = amount
        self._total_credits_issued += amount

    def get_balance(self, member_id: str) -> float:
        return self._balances.get(member_id, 0.0)

    def get_price(self, category: str) -> float:
        if category in self.curves:
            return self.curves[category].get_price()
        return 5.0  # fallback

    def get_all_prices(self) -> Dict[str, float]:
        return {cat: curve.get_price() for cat, curve in self.curves.items()}

    def update_curves(self, supply_counts: Dict[str, int], demand_counts: Dict[str, int], cycle: int):
        """Update all bonding curves with current supply/demand counts."""
        for cat, curve in self.curves.items():
            s = supply_counts.get(cat, 0)
            d = demand_counts.get(cat, 0)
            curve.update(s, d, cycle)

    def _compute_fee_rate(self, category: str) -> float:
        """Compute protocol fee rate — flat or progressive based on category price.

        Low-price categories (<= mean): 1% (saves money for common-service consumers)
        Medium (mean to 1.5x mean): linear 1% to 4%
        High (> 1.5x mean): 5% (excess above flat funds redistribution pool)
        """
        if self.fee_mode != "progressive":
            return self.protocol_fee

        price = self.get_price(category)
        mean_price = self._mean_base_price

        if price <= mean_price:
            return 0.01
        elif price <= 1.5 * mean_price:
            t = (price - mean_price) / (0.5 * mean_price)
            return 0.01 + t * 0.03  # 1% to 4%
        else:
            return 0.05  # 5% for high-price categories

    def transfer(
        self,
        from_id: str,
        to_id: str,
        amount: float,
        category: str,
    ) -> TransactionResult:
        """Execute a credit transfer with protocol fee."""
        if amount <= 0:
            return TransactionResult(False, 0, 0, 0, category, "Invalid amount")

        fee_rate = self._compute_fee_rate(category)
        fee = amount * fee_rate
        total_debit = amount + fee

        if self._balances.get(from_id, 0) < total_debit:
            return TransactionResult(
                False, 0, 0, self.get_price(category), category,
                "Insufficient balance"
            )

        self._balances[from_id] -= total_debit
        self._balances[to_id] = self._balances.get(to_id, 0) + amount

        if self.fee_mode == "progressive":
            # Treasury gets the lesser of actual fee and flat-equivalent
            flat_fee = amount * self.protocol_fee
            self.treasury_credits += min(fee, flat_fee)
            # Surplus above flat goes to redistribution pool for below-median members
            surplus = fee - flat_fee
            if surplus > 0:
                self._redistribution_pool += surplus
        else:
            self.treasury_credits += fee

        self._transaction_count += 1
        self._cycle_transactions += 1
        self._cycle_transfer_volume += amount

        return TransactionResult(True, amount, fee, self.get_price(category), category)

    def bridge_redeem(self, member_id: str, amount: float, current_cycle: int) -> BridgeResult:
        """Attempt to redeem credits for fiat via the asymmetric bridge."""
        balance = self._balances.get(member_id, 0)
        if amount > balance:
            return BridgeResult(False, 0, 0, 0, "Insufficient credits")

        # Progressive discount based on total redeemed
        total_redeemed = self._bridge_redemptions.get(member_id, 0)
        discount = 0.0
        for tier in sorted(self.bridge_discounts, key=lambda t: t["threshold"], reverse=True):
            if total_redeemed + amount >= tier["threshold"]:
                discount = tier["discount"]
                break

        # Check reserve ratio
        total_redeemable = max(1, self._total_credits_issued)
        reserve_ratio = self.treasury_fiat / total_redeemable
        if reserve_ratio < self.suspended_reserve_ratio:
            return BridgeResult(False, 0, 0, 0, "Bridge suspended: reserve too low")
        if reserve_ratio < self.min_reserve_ratio:
            # Restricted mode: cap at 50% of requested
            amount = amount * 0.5

        fiat_value = amount * (1.0 - discount)
        if fiat_value > self.treasury_fiat:
            return BridgeResult(False, 0, 0, 0, "Insufficient treasury fiat")

        self._balances[member_id] -= amount
        self.treasury_fiat -= fiat_value
        self._bridge_redemptions[member_id] = total_redeemed + amount

        return BridgeResult(True, amount, fiat_value, discount)

    def apply_demurrage(self, current_cycle: int):
        """
        Apply floating demurrage: d = min(d_max, d_max * (1 - V / V_target))
        Only after grace period.

        In progressive mode, demurrage rate scales with balance size relative
        to the median: 0% at/below median, d_max at 2x median, 2.5*d_max at 5x median.
        """
        if current_cycle < self.demurrage_grace_cycles:
            return

        velocity = self._compute_velocity()
        if velocity >= self.demurrage_velocity_target:
            return  # healthy circulation, no demurrage

        d_base = min(
            self.demurrage_max,
            self.demurrage_max * (1.0 - velocity / self.demurrage_velocity_target)
        )

        demurrage_collected = 0.0

        if self.demurrage_mode == "progressive":
            positive_balances = [b for b in self._balances.values() if b > 0]
            if not positive_balances:
                return
            median_balance = float(np.median(positive_balances))
            if median_balance < 1.0:
                median_balance = 1.0

            for mid in self._balances:
                bal = self._balances[mid]
                if bal <= 0:
                    continue
                ratio = bal / median_balance
                if ratio <= 1.0:
                    d = 0.0  # no demurrage at/below median
                elif ratio <= 2.0:
                    # Linear from 0% to d_base between 1x and 2x median
                    d = d_base * (ratio - 1.0)
                elif ratio <= 5.0:
                    # Linear from d_base to 2.5*d_base between 2x and 5x median
                    d = d_base * (1.0 + 0.5 * (ratio - 2.0))
                else:
                    d = d_base * 2.5  # cap at 2.5x base rate
                reduction = bal * d
                self._balances[mid] -= reduction
                demurrage_collected += reduction
        else:
            for mid in self._balances:
                if self._balances[mid] > 0:
                    reduction = self._balances[mid] * d_base
                    self._balances[mid] -= reduction
                    demurrage_collected += reduction

        self.treasury_credits += demurrage_collected

    def redistribute(self):
        """
        Distribute the redistribution pool equally to members below median balance.
        Called once per cycle after demurrage. Only active in progressive fee mode.
        """
        if self._redistribution_pool <= 0:
            return

        positive_balances = [b for b in self._balances.values() if b > 0]
        if not positive_balances:
            return
        median_balance = float(np.median(positive_balances))

        eligible = [mid for mid, bal in self._balances.items() if bal < median_balance]
        if not eligible:
            # Nobody below median — carry pool forward
            return

        per_member = self._redistribution_pool / len(eligible)
        for mid in eligible:
            self._balances[mid] += per_member
        self._redistribution_pool = 0.0

    def _compute_velocity(self) -> float:
        """Credit velocity = transfer volume / total credits in circulation."""
        total_credits = sum(self._balances.values())
        if total_credits <= 0:
            return 0.0
        return self._cycle_transfer_volume / max(1.0, total_credits)

    def begin_cycle(self):
        """Reset per-cycle counters."""
        self._cycle_transactions = 0
        self._cycle_transfer_volume = 0.0
        self._cycle_matched = 0
        self._cycle_total_requests = 0

    def record_match_stats(self, matched: int, total_requests: int):
        """Record matching statistics for this cycle."""
        self._cycle_matched = matched
        self._cycle_total_requests = total_requests

    def get_metrics(self, cycle: int) -> EconomyMetrics:
        """Compute all economy metrics for the current state."""
        balances = list(self._balances.values())
        total_supply = sum(balances) if balances else 0

        # Gini coefficient
        gini = self._compute_gini(balances)

        # Velocity
        velocity = self._compute_velocity()

        # Match rate
        match_rate = (
            self._cycle_matched / max(1, self._cycle_total_requests)
        )

        # Reserve ratio
        total_redeemable = max(1, self._total_credits_issued)
        reserve_ratio = self.treasury_fiat / total_redeemable

        # Per-category supply/demand
        per_cat_supply = {}
        per_cat_demand = {}
        for cat, curve in self.curves.items():
            per_cat_supply[cat] = curve.supply
            per_cat_demand[cat] = curve.demand

        return EconomyMetrics(
            cycle=cycle,
            gini=gini,
            velocity=velocity,
            total_supply=total_supply,
            total_transactions=self._cycle_transactions,
            prices=self.get_all_prices(),
            match_rate=match_rate,
            treasury_fiat=self.treasury_fiat,
            treasury_credits=self.treasury_credits,
            reserve_ratio=reserve_ratio,
            per_category_supply=per_cat_supply,
            per_category_demand=per_cat_demand,
        )

    @staticmethod
    def _compute_gini(values: List[float]) -> float:
        """Compute Gini coefficient of a distribution."""
        if not values or len(values) < 2:
            return 0.0
        arr = np.array(values, dtype=np.float64)
        arr = arr[arr >= 0]  # only non-negative
        if len(arr) < 2 or arr.sum() == 0:
            return 0.0
        arr = np.sort(arr)
        n = len(arr)
        index = np.arange(1, n + 1)
        return (2.0 * np.sum(index * arr) / (n * np.sum(arr))) - (n + 1) / n
