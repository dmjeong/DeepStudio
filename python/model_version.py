"""Deployment timestamps, independent of training and the JSON schema version."""

from datetime import datetime


def stamp_model_version(config):
    """Stamp a newly exported/converted artifact using the export PC's local time.

    Keep the timezone in a separate ISO-8601 value. Fractional seconds distinguish
    rapid exports, but this is a display/release timestamp, not a content hash or
    a globally unique, monotonic identifier. Encryption alone preserves it.
    """
    now = datetime.now().astimezone()
    config["model_version"] = now.strftime("%Y%m%d_%H%M%S_%f")
    config["model_exported_at"] = now.isoformat(timespec="microseconds")
