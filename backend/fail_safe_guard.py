"""Validate telemetry packets before they enter the analytics pipeline."""

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import ValidationError

from backend.schemas import TelemetryPacket


@dataclass(frozen=True)
class TelemetryAssessment:
    quality: Literal["good", "degraded"]
    telemetry: dict[str, Any] | None
    issues: tuple[str, ...] = ()


def assess_telemetry(packet: dict[str, Any]) -> TelemetryAssessment:
    """Reject incomplete or physically implausible packets without imputation."""
    try:
        validated = TelemetryPacket.model_validate(packet)
    except ValidationError as exc:
        issues = tuple(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        )
        return TelemetryAssessment(
            quality="degraded",
            telemetry=None,
            issues=issues,
        )

    return TelemetryAssessment(
        quality="good",
        telemetry=validated.model_dump(mode="json", exclude_none=True),
    )
