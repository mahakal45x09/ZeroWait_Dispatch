"""
Business Rules Engine for ZeroWait Dispatch.

Extracts the hardcoded business logic from app.py into a configurable,
testable rules engine. Each rule is defined declaratively and applied
sequentially to adjust the base KPT prediction.
"""


class BusinessRule:
    """A single business rule that adjusts the KPT prediction."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def evaluate(self, context: dict, current_kpt: float) -> tuple[float, str | None]:
        """
        Evaluate this rule against the given context.
        
        Returns:
            tuple of (adjusted_kpt, applied_rule_message or None)
        """
        raise NotImplementedError


class ReliabilityBufferRule(BusinessRule):
    """Adds buffer time for restaurants with low reliability scores."""

    def __init__(self, threshold: float = 0.75, buffer_minutes: float = 4.0):
        super().__init__("Reliability Buffer", "Adds buffer for unreliable restaurants")
        self.threshold = threshold
        self.buffer_minutes = buffer_minutes

    def evaluate(self, context: dict, current_kpt: float) -> tuple[float, str | None]:
        if context.get("reliability_score", 1.0) < self.threshold:
            return (
                current_kpt + self.buffer_minutes,
                f"Low Reliability Buffer (+{self.buffer_minutes}m)"
            )
        return current_kpt, None


class KitchenCapacitySurgeRule(BusinessRule):
    """Applies surge penalty when kitchen is overwhelmed."""

    def __init__(self, surge_percent: float = 0.20):
        super().__init__("Capacity Surge", "Penalty when kitchen exceeds capacity")
        self.surge_percent = surge_percent

    def evaluate(self, context: dict, current_kpt: float) -> tuple[float, str | None]:
        active = context.get("current_active_orders", 0)
        capacity = context.get("kitchen_capacity", 999)
        if active >= capacity:
            surge = current_kpt * self.surge_percent
            return (
                current_kpt + surge,
                f"Capacity Surge Penalty (+{surge:.1f}m)"
            )
        return current_kpt, None


class POSKitchenLoadRule(BusinessRule):
    """Adds penalty when POS dine-in load is high."""

    def __init__(self, threshold: int = 15, penalty_per_order: float = 0.5):
        super().__init__("POS Load Surge", "Penalty for high dine-in traffic")
        self.threshold = threshold
        self.penalty_per_order = penalty_per_order

    def evaluate(self, context: dict, current_kpt: float) -> tuple[float, str | None]:
        pos_load = context.get("total_pos_kitchen_load", 0)
        if pos_load > self.threshold:
            penalty = (pos_load - self.threshold) * self.penalty_per_order
            return (
                current_kpt + penalty,
                f"POS Dine-in Load Surge (+{penalty:.1f}m)"
            )
        return current_kpt, None


class MerchantBiasRule(BusinessRule):
    """Adjusts KPT based on merchant's historical geo-FOR bias."""

    def __init__(self, high_buffer: float = 5.0, trustworthy_reduction: float = 2.0):
        super().__init__("Merchant Bias", "Adjustment based on marking behavior")
        self.high_buffer = high_buffer
        self.trustworthy_reduction = trustworthy_reduction

    def evaluate(self, context: dict, current_kpt: float) -> tuple[float, str | None]:
        bias = context.get("merchant_bias_score", "Medium (Standard)")
        if bias == "High (Marks Early)":
            return (
                current_kpt + self.high_buffer,
                f"High Merchant Bias Buffer (+{self.high_buffer}m)"
            )
        elif bias == "Low (Trustworthy)":
            return (
                current_kpt - self.trustworthy_reduction,
                f"Trustworthy Merchant Reduction (-{self.trustworthy_reduction}m)"
            )
        return current_kpt, None


class IoTButtonRule(BusinessRule):
    """Reduces KPT when merchant uses ZeroTap IoT button."""

    def __init__(self, reduction: float = 3.0):
        super().__init__("ZeroTap IoT", "Reduction when IoT button confirms readiness")
        self.reduction = reduction

    def evaluate(self, context: dict, current_kpt: float) -> tuple[float, str | None]:
        if context.get("used_iot_button", False):
            return (
                current_kpt - self.reduction,
                f"ZeroTap IoT Button Used (-{self.reduction}m)"
            )
        return current_kpt, None


class WeatherBufferRule(BusinessRule):
    """Adds buffer for adverse weather conditions."""

    def __init__(self):
        super().__init__("Weather Buffer", "Adjusts for live weather impact")
        self.weather_penalties = {
            "Light Rain": 3.0,
            "Heavy Rain / Waterlogging": 8.0,
        }

    def evaluate(self, context: dict, current_kpt: float) -> tuple[float, str | None]:
        weather = context.get("live_weather_condition", "Clear")
        penalty = self.weather_penalties.get(weather, 0.0)
        if penalty > 0:
            label = "Light Rain Buffer" if "Light" in weather else "Heavy Rain Emergency Buffer"
            return (
                current_kpt + penalty,
                f"{label} (+{penalty}m)"
            )
        return current_kpt, None


# ── Default rule chain (order matters) ──────────────────────────────
DEFAULT_RULES = [
    ReliabilityBufferRule(),
    KitchenCapacitySurgeRule(),
    POSKitchenLoadRule(),
    MerchantBiasRule(),
    IoTButtonRule(),
    WeatherBufferRule(),
]

MIN_KPT_MINUTES = 5.0


def apply_business_rules(
    base_kpt: float,
    context: dict,
    rules: list[BusinessRule] | None = None,
    min_kpt: float = MIN_KPT_MINUTES,
) -> tuple[float, list[str]]:
    """
    Apply all business rules sequentially to adjust the base KPT.

    Args:
        base_kpt: Raw ML model prediction in minutes.
        context: Dictionary with all order/rider/merchant context.
        rules: List of BusinessRule instances (defaults to DEFAULT_RULES).
        min_kpt: Floor value for the adjusted KPT.

    Returns:
        tuple of (final_adjusted_kpt, list_of_applied_rule_messages)
    """
    if rules is None:
        rules = DEFAULT_RULES

    adjusted_kpt = base_kpt
    applied_rules = []

    for rule in rules:
        adjusted_kpt, message = rule.evaluate(context, adjusted_kpt)
        if message:
            applied_rules.append(message)

    # Ensure KPT never drops below the minimum
    adjusted_kpt = max(min_kpt, adjusted_kpt)

    return adjusted_kpt, applied_rules
