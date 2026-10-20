"""
mission_profiles.py
--------------------
Defines UAV mission phases and profiles used to drive the engine simulator
and the mission-simulation module. Mission = altitude(t) + throttle(t) +
ambient_temperature + duration + operating phase.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple


class MissionPhase(str, Enum):
    START = "start"
    TAKEOFF = "takeoff"
    CLIMB = "climb"
    CRUISE = "cruise"
    LOITER = "loiter"
    THROTTLE_TRANSITION = "throttle_transition"
    DESCENT = "descent"
    SHUTDOWN = "shutdown"


@dataclass
class PhaseSegment:
    phase: MissionPhase
    start_s: float
    end_s: float
    throttle: float          # 0.0 - 1.0
    altitude_ft: float


@dataclass
class MissionProfile:
    mission_id: str
    ambient_temperature: float          # deg C, constant assumption for MVP
    segments: List[PhaseSegment] = field(default_factory=list)

    def duration(self) -> float:
        return max(s.end_s for s in self.segments) if self.segments else 0.0

    def state_at(self, t: float) -> Tuple[MissionPhase, float]:
        for seg in self.segments:
            if seg.start_s <= t < seg.end_s:
                return seg.phase, seg.throttle
        # past mission end -> hold last segment
        last = self.segments[-1]
        return last.phase, last.throttle

    def altitude_at(self, t: float) -> float:
        for seg in self.segments:
            if seg.start_s <= t < seg.end_s:
                return seg.altitude_ft
        return self.segments[-1].altitude_ft if self.segments else 0.0


def make_normal_cruise_mission(mission_id: str = "MISSION-NORMAL") -> MissionProfile:
    return MissionProfile(
        mission_id=mission_id,
        ambient_temperature=25.0,
        segments=[
            PhaseSegment(MissionPhase.START, 0, 30, 0.0, 0),
            PhaseSegment(MissionPhase.TAKEOFF, 30, 90, 0.95, 500),
            PhaseSegment(MissionPhase.CLIMB, 90, 300, 0.85, 5000),
            PhaseSegment(MissionPhase.CRUISE, 300, 3600, 0.65, 6000),
            PhaseSegment(MissionPhase.DESCENT, 3600, 3800, 0.4, 1000),
            PhaseSegment(MissionPhase.SHUTDOWN, 3800, 3860, 0.0, 0),
        ],
    )


def make_endurance_mission(mission_id: str = "MISSION-ENDURANCE") -> MissionProfile:
    """8-hour hot-weather high-altitude endurance/surveillance mission."""
    return MissionProfile(
        mission_id=mission_id,
        ambient_temperature=36.0,
        segments=[
            PhaseSegment(MissionPhase.START, 0, 30, 0.0, 0),
            PhaseSegment(MissionPhase.TAKEOFF, 30, 90, 0.95, 500),
            PhaseSegment(MissionPhase.CLIMB, 90, 400, 0.9, 8000),
            PhaseSegment(MissionPhase.CRUISE, 400, 27000, 0.72, 8000),
            PhaseSegment(MissionPhase.LOITER, 27000, 28200, 0.55, 7500),
            PhaseSegment(MissionPhase.THROTTLE_TRANSITION, 28200, 28260, 0.85, 7500),
            PhaseSegment(MissionPhase.DESCENT, 28260, 28560, 0.4, 1000),
            PhaseSegment(MissionPhase.SHUTDOWN, 28560, 28620, 0.0, 0),
        ],
    )


def make_hot_high_altitude_mission(mission_id: str = "MISSION-HOTALT") -> MissionProfile:
    return MissionProfile(
        mission_id=mission_id,
        ambient_temperature=42.0,
        segments=[
            PhaseSegment(MissionPhase.START, 0, 30, 0.0, 0),
            PhaseSegment(MissionPhase.TAKEOFF, 30, 90, 0.97, 500),
            PhaseSegment(MissionPhase.CLIMB, 90, 500, 0.9, 12000),
            PhaseSegment(MissionPhase.CRUISE, 500, 10000, 0.75, 12000),
            PhaseSegment(MissionPhase.DESCENT, 10000, 10300, 0.4, 1000),
            PhaseSegment(MissionPhase.SHUTDOWN, 10300, 10360, 0.0, 0),
        ],
    )
