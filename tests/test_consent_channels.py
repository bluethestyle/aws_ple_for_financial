"""
Tests for M3 channel-level marketing consent under the new ConsentConfig.

Covers:
- Legacy default (4 channels) still works.
- pipeline.yaml 5-channel config (SMS/EMAIL/APP_PUSH/PHONE/MAIL) is respected.
- Night-hours window is config-driven, wraps midnight correctly.
- DNC registry can be disabled via config.
- Unknown channels are rejected.

Run: pytest tests/test_consent_channels.py -v
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from core.compliance.consent_manager import (
    ConsentConfig,
    ConsentManager,
)

_KST = timezone(timedelta(hours=9))


# ---------------------------------------------------------------------------
# ConsentConfig dataclass
# ---------------------------------------------------------------------------

class TestConsentConfig:
    def test_legacy_defaults(self):
        cfg = ConsentConfig()
        assert cfg.channels == ("sms", "email", "app_push", "third_party")
        assert cfg.night_hours_kst == (21, 8)
        assert cfg.default_retention_days == 365
        assert cfg.dnc_registry_enabled is True

    def test_from_dict_uppercase_channels(self):
        cfg = ConsentConfig.from_dict({
            "channels": ["SMS", "EMAIL", "APP_PUSH", "PHONE", "MAIL"],
            "night_hours_kst": [21, 8],
            "default_retention_days": 180,
            "dnc_registry_enabled": True,
        })
        assert "SMS" in cfg.channels
        assert "PHONE" in cfg.channels
        assert "MAIL" in cfg.channels
        assert cfg.default_retention_days == 180

    def test_rejects_empty_channels(self):
        with pytest.raises(ValueError):
            ConsentConfig(channels=())

    def test_rejects_bad_hours(self):
        with pytest.raises(ValueError):
            ConsentConfig(night_hours_kst=(25, 8))

    def test_rejects_nonpositive_retention(self):
        with pytest.raises(ValueError):
            ConsentConfig(default_retention_days=0)


# ---------------------------------------------------------------------------
# Channel validation
# ---------------------------------------------------------------------------

class TestChannelValidation:
    def test_legacy_channel_accepted(self):
        mgr = ConsentManager(use_dynamo=False)
        mgr.grant_consent("C001", "sms", "customer_portal")
        assert mgr.check_consent("C001", "sms") is True

    def test_uppercase_config_accepts_uppercase(self):
        cfg = ConsentConfig(
            channels=("SMS", "EMAIL", "APP_PUSH", "PHONE", "MAIL"),
        )
        mgr = ConsentManager(use_dynamo=False, config=cfg)
        mgr.grant_consent("C001", "PHONE", "customer_portal")
        assert mgr.check_consent("C001", "PHONE") is True

    def test_uppercase_config_rejects_lowercase(self):
        cfg = ConsentConfig(
            channels=("SMS", "EMAIL", "APP_PUSH", "PHONE", "MAIL"),
        )
        mgr = ConsentManager(use_dynamo=False, config=cfg)
        with pytest.raises(ValueError, match="Invalid channel"):
            mgr.grant_consent("C001", "sms", "customer_portal")

    def test_unknown_channel_rejected(self):
        mgr = ConsentManager(use_dynamo=False)
        with pytest.raises(ValueError):
            mgr.grant_consent("C001", "telegram", "customer_portal")


# ---------------------------------------------------------------------------
# Night-time window
# ---------------------------------------------------------------------------

class TestNightTimeWindow:
    def test_default_21_to_08_wraps_midnight(self):
        mgr = ConsentManager(use_dynamo=False)
        at_22 = datetime(2026, 4, 21, 22, 0, 0, tzinfo=_KST)
        at_03 = datetime(2026, 4, 21, 3, 0, 0, tzinfo=_KST)
        at_10 = datetime(2026, 4, 21, 10, 0, 0, tzinfo=_KST)
        assert mgr.check_nighttime("sms", now=at_22) is True
        assert mgr.check_nighttime("sms", now=at_03) is True
        assert mgr.check_nighttime("sms", now=at_10) is False

    def test_night_exempt_channel_never_blocked(self):
        mgr = ConsentManager(use_dynamo=False)
        at_22 = datetime(2026, 4, 21, 22, 0, 0, tzinfo=_KST)
        # "third_party" is in default night_exempt_channels
        assert mgr.check_nighttime("third_party", now=at_22) is False

    def test_custom_window_non_wrapping(self):
        cfg = ConsentConfig(
            channels=("SMS",),
            night_hours_kst=(1, 5),
            night_exempt_channels=(),
        )
        mgr = ConsentManager(use_dynamo=False, config=cfg)
        at_03 = datetime(2026, 4, 21, 3, 0, 0, tzinfo=_KST)
        at_06 = datetime(2026, 4, 21, 6, 0, 0, tzinfo=_KST)
        assert mgr.check_nighttime("SMS", now=at_03) is True
        assert mgr.check_nighttime("SMS", now=at_06) is False


# ---------------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------------

class TestRetention:
    def test_grant_uses_config_default_retention(self):
        cfg = ConsentConfig(
            channels=("sms",), default_retention_days=30,
        )
        mgr = ConsentManager(use_dynamo=False, config=cfg)
        mgr.grant_consent("C001", "sms", "customer_portal")
        status = mgr.get_consent_status("C001")
        record = status["sms"]
        expires = datetime.fromisoformat(record.expires_at)
        delta = expires - datetime.fromisoformat(record.updated_at)
        # ~30 days (allow floating point / microsecond drift)
        assert 29 <= delta.days <= 31

    def test_grant_explicit_expires_days_overrides_config(self):
        cfg = ConsentConfig(
            channels=("sms",), default_retention_days=30,
        )
        mgr = ConsentManager(use_dynamo=False, config=cfg)
        mgr.grant_consent("C001", "sms", "customer_portal", expires_days=7)
        status = mgr.get_consent_status("C001")
        record = status["sms"]
        expires = datetime.fromisoformat(record.expires_at)
        delta = expires - datetime.fromisoformat(record.updated_at)
        assert 6 <= delta.days <= 8


# ---------------------------------------------------------------------------
# DNC toggle
# ---------------------------------------------------------------------------

class TestDNCRegistry:
    def test_default_dnc_enabled(self):
        mgr = ConsentManager(use_dynamo=False)
        mgr.add_to_dnc("C001", reason="customer request")
        assert mgr.check_dnc("C001") is True

    def test_dnc_disabled_by_config(self):
        cfg = ConsentConfig(
            channels=("sms",), dnc_registry_enabled=False,
        )
        mgr = ConsentManager(use_dynamo=False, config=cfg)
        mgr.add_to_dnc("C001", reason="customer request")
        # Even though we added to DNC, check_dnc returns False because
        # the registry is disabled config-wide.
        assert mgr.check_dnc("C001") is False


# ---------------------------------------------------------------------------
# Full contactability flow
# ---------------------------------------------------------------------------

class TestContactability:
    def test_contactable_when_consented_and_daytime(self, monkeypatch):
        mgr = ConsentManager(use_dynamo=False)
        mgr.grant_consent("C001", "sms", "customer_portal")

        daytime = datetime(2026, 4, 21, 10, 0, 0, tzinfo=_KST)

        class FakeDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                if tz is None:
                    return daytime.astimezone(timezone.utc).replace(tzinfo=None)
                return daytime.astimezone(tz)

        monkeypatch.setattr(
            "core.compliance.consent_manager.datetime", FakeDatetime
        )
        contactable, reason = mgr.is_contactable("C001", "sms")
        assert contactable is True
        assert reason == ""


# ---------------------------------------------------------------------------
# Config flows from pipeline.yaml
# ---------------------------------------------------------------------------

class TestPipelineYAMLBlock:
    def test_consent_block_drives_config(self):
        import yaml
        from pathlib import Path

        cfg = yaml.safe_load(Path("configs/pipeline.yaml").read_text(
            encoding="utf-8"
        ))
        consent_cfg = ConsentConfig.from_dict(cfg["compliance"]["consent"])
        expected = {"SMS", "EMAIL", "APP_PUSH", "PHONE", "MAIL"}
        assert set(consent_cfg.channels) == expected
        assert consent_cfg.night_hours_kst == (21, 8)

    def test_manager_constructed_from_config(self):
        import yaml
        from pathlib import Path

        cfg = yaml.safe_load(Path("configs/pipeline.yaml").read_text(
            encoding="utf-8"
        ))
        consent_cfg = ConsentConfig.from_dict(cfg["compliance"]["consent"])
        mgr = ConsentManager(use_dynamo=False, config=consent_cfg)
        mgr.grant_consent("C001", "PHONE", "customer_portal")
        assert mgr.check_consent("C001", "PHONE") is True
