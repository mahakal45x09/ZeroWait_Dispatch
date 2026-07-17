"""
Unit & Integration tests for ZeroWait Dispatch.

Run with:  pytest tests/ -v
"""

import pytest
from business_rules import (
    ReliabilityBufferRule,
    KitchenCapacitySurgeRule,
    POSKitchenLoadRule,
    MerchantBiasRule,
    IoTButtonRule,
    WeatherBufferRule,
    apply_business_rules,
)


# ════════════════════════════════════════════════════════════════════
# Unit Tests — Individual Business Rules
# ════════════════════════════════════════════════════════════════════

class TestReliabilityBufferRule:
    """Tests for Rule A: Reliability Buffer."""

    def test_low_reliability_adds_buffer(self):
        rule = ReliabilityBufferRule(threshold=0.75, buffer_minutes=4.0)
        kpt, msg = rule.evaluate({"reliability_score": 0.60}, 20.0)
        assert kpt == 24.0
        assert msg is not None
        assert "+4.0m" in msg

    def test_high_reliability_no_change(self):
        rule = ReliabilityBufferRule()
        kpt, msg = rule.evaluate({"reliability_score": 0.90}, 20.0)
        assert kpt == 20.0
        assert msg is None

    def test_exact_threshold_no_change(self):
        rule = ReliabilityBufferRule(threshold=0.75)
        kpt, msg = rule.evaluate({"reliability_score": 0.75}, 20.0)
        assert kpt == 20.0
        assert msg is None


class TestKitchenCapacitySurgeRule:
    """Tests for Rule B: Kitchen Capacity Surge."""

    def test_overwhelmed_kitchen_adds_surge(self):
        rule = KitchenCapacitySurgeRule(surge_percent=0.20)
        context = {"current_active_orders": 12, "kitchen_capacity": 10}
        kpt, msg = rule.evaluate(context, 20.0)
        assert kpt == 24.0  # 20 + (20 * 0.20)
        assert msg is not None

    def test_under_capacity_no_change(self):
        rule = KitchenCapacitySurgeRule()
        context = {"current_active_orders": 5, "kitchen_capacity": 10}
        kpt, msg = rule.evaluate(context, 20.0)
        assert kpt == 20.0
        assert msg is None

    def test_exact_capacity_triggers_surge(self):
        rule = KitchenCapacitySurgeRule(surge_percent=0.20)
        context = {"current_active_orders": 10, "kitchen_capacity": 10}
        kpt, msg = rule.evaluate(context, 20.0)
        assert kpt == 24.0
        assert msg is not None


class TestPOSKitchenLoadRule:
    """Tests for Rule C: POS Kitchen Load."""

    def test_high_pos_load_adds_penalty(self):
        rule = POSKitchenLoadRule(threshold=15, penalty_per_order=0.5)
        kpt, msg = rule.evaluate({"total_pos_kitchen_load": 20}, 20.0)
        assert kpt == 22.5  # 20 + (5 * 0.5)
        assert msg is not None

    def test_low_pos_load_no_change(self):
        rule = POSKitchenLoadRule()
        kpt, msg = rule.evaluate({"total_pos_kitchen_load": 10}, 20.0)
        assert kpt == 20.0
        assert msg is None


class TestMerchantBiasRule:
    """Tests for Rule D: Merchant Geo-FOR Bias."""

    def test_high_bias_adds_buffer(self):
        rule = MerchantBiasRule(high_buffer=5.0)
        kpt, msg = rule.evaluate({"merchant_bias_score": "High (Marks Early)"}, 20.0)
        assert kpt == 25.0
        assert msg is not None

    def test_trustworthy_reduces_kpt(self):
        rule = MerchantBiasRule(trustworthy_reduction=2.0)
        kpt, msg = rule.evaluate({"merchant_bias_score": "Low (Trustworthy)"}, 20.0)
        assert kpt == 18.0
        assert msg is not None

    def test_medium_bias_no_change(self):
        rule = MerchantBiasRule()
        kpt, msg = rule.evaluate({"merchant_bias_score": "Medium (Standard)"}, 20.0)
        assert kpt == 20.0
        assert msg is None


class TestIoTButtonRule:
    """Tests for Rule E: ZeroTap IoT Button."""

    def test_iot_button_reduces_kpt(self):
        rule = IoTButtonRule(reduction=3.0)
        kpt, msg = rule.evaluate({"used_iot_button": True}, 20.0)
        assert kpt == 17.0
        assert msg is not None

    def test_no_iot_button_no_change(self):
        rule = IoTButtonRule()
        kpt, msg = rule.evaluate({"used_iot_button": False}, 20.0)
        assert kpt == 20.0
        assert msg is None


class TestWeatherBufferRule:
    """Tests for Rule F: Weather Buffer."""

    def test_light_rain_adds_buffer(self):
        rule = WeatherBufferRule()
        kpt, msg = rule.evaluate({"live_weather_condition": "Light Rain"}, 20.0)
        assert kpt == 23.0
        assert msg is not None

    def test_heavy_rain_adds_large_buffer(self):
        rule = WeatherBufferRule()
        kpt, msg = rule.evaluate({"live_weather_condition": "Heavy Rain / Waterlogging"}, 20.0)
        assert kpt == 28.0
        assert msg is not None

    def test_clear_weather_no_change(self):
        rule = WeatherBufferRule()
        kpt, msg = rule.evaluate({"live_weather_condition": "Clear"}, 20.0)
        assert kpt == 20.0
        assert msg is None


# ════════════════════════════════════════════════════════════════════
# Integration Tests — Full Rules Pipeline
# ════════════════════════════════════════════════════════════════════

class TestApplyBusinessRules:
    """Tests for the full business rules pipeline."""

    def test_no_rules_triggered(self):
        context = {
            "reliability_score": 0.90,
            "current_active_orders": 5,
            "kitchen_capacity": 10,
            "total_pos_kitchen_load": 5,
            "merchant_bias_score": "Medium (Standard)",
            "used_iot_button": False,
            "live_weather_condition": "Clear",
        }
        adjusted, rules = apply_business_rules(20.0, context)
        assert adjusted == 20.0
        assert len(rules) == 0

    def test_multiple_rules_triggered(self):
        context = {
            "reliability_score": 0.60,      # triggers Rule A (+4)
            "current_active_orders": 12,
            "kitchen_capacity": 10,          # triggers Rule B (+20%)
            "total_pos_kitchen_load": 20,    # triggers Rule C
            "merchant_bias_score": "High (Marks Early)",  # triggers Rule D (+5)
            "used_iot_button": False,
            "live_weather_condition": "Heavy Rain / Waterlogging",  # triggers Rule F (+8)
        }
        adjusted, rules = apply_business_rules(20.0, context)
        assert adjusted > 20.0
        assert len(rules) >= 4  # At least 4 rules should fire

    def test_minimum_kpt_floor(self):
        """KPT should never drop below 5 minutes even with reductions."""
        context = {
            "reliability_score": 0.95,
            "current_active_orders": 2,
            "kitchen_capacity": 10,
            "total_pos_kitchen_load": 0,
            "merchant_bias_score": "Low (Trustworthy)",  # -2
            "used_iot_button": True,                      # -3
            "live_weather_condition": "Clear",
        }
        # base_kpt=6.0, minus 2 (bias) minus 3 (IoT) = 1.0 → floored to 5.0
        adjusted, rules = apply_business_rules(6.0, context)
        assert adjusted == 5.0
        assert len(rules) == 2

    def test_empty_context_uses_defaults(self):
        """Rules should handle missing keys gracefully via defaults."""
        adjusted, rules = apply_business_rules(20.0, {})
        assert adjusted == 20.0
        assert len(rules) == 0
