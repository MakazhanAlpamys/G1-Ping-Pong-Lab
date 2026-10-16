"""Conservative single-return evaluation from physical ball contacts.

This is a return-skill benchmark, not a complete ITTF referee. The caller must
observe every physics substep and supply *active MuJoCo ball contacts*, mapped
to ``table``, ``racket``, ``net``, ``floor`` and ``body``. Geometry proximity and
table-height plane crossings are deliberately insufficient to score a bounce.
The negative-X table half belongs to the robot relative to ``net_x``.
"""

from dataclasses import dataclass
import math
from typing import Iterable


@dataclass(frozen=True)
class TableGeometry:
    net_x: float = 0.0
    table_x_min: float = -1.37
    table_x_max: float = 1.37
    table_y_min: float = -0.7625
    table_y_max: float = 0.7625
    net_top_z: float = 0.9125
    ball_radius: float = 0.02

    def __post_init__(self):
        if not all(math.isfinite(value) for value in vars(self).values()):
            raise ValueError("Table geometry must be finite")
        if not (self.table_x_min < self.net_x < self.table_x_max
                and self.table_y_min < self.table_y_max and self.ball_radius > 0):
            raise ValueError("Invalid table bounds, net position or ball radius")

    def contains(self, position: tuple[float, float, float]) -> bool:
        x, y, _ = position
        return (self.table_x_min <= x <= self.table_x_max
                and self.table_y_min <= y <= self.table_y_max)


class ReturnAttempt:
    """One explicit serve; create a fresh instance for the next serve.

    A valid return needs one own-half physical table bounce, one racket contact,
    an outgoing net crossing above the net, and the first post-hit physical
    table contact on the far half. Sustained contacts count as one contact
    episode. A valid landing settles the ball outcome, while subsequent robot
    falls or an aborted attempt still invalidate success. Call ``finish`` only
    at the end of the robot-stability observation window.
    """

    CONTACT_TYPES = frozenset({"table", "racket", "net", "floor", "body"})

    def __init__(self, geometry: TableGeometry | None = None):
        self.geometry = geometry or TableGeometry()
        self.own_bounce = False
        self.contacted = False
        self.net_clearance = False
        self.fallen = False
        self.first_landing: tuple[float, float, float] | None = None
        self.trace: list[dict] = []
        self._active_contacts: set[str] = set()
        self._previous_position: tuple[float, float, float] | None = None
        self._time: float | None = None
        self._failure: str | None = None
        self._landed = False
        self._finished = False
        self._aborted = False

    def _event(self, event: str, position: tuple[float, float, float]):
        self.trace.append({"time": self._time, "event": event, "position": list(position)})

    def _fail(self, reason: str, position: tuple[float, float, float]):
        if self._failure is None:
            self._failure = reason
            self._event(reason, position)

    def observe(self, *, time: float, ball_position: Iterable[float],
                contacts: Iterable[str] = (), fallen: bool = False) -> None:
        """Process a substep. Contact names represent actual ball-geom contact.

        ``ball_position`` is the world-frame ball centre. On table contact its
        horizontal position must be inside the configured table rectangle.
        Times must be finite and strictly increase, preventing ambiguous event
        ordering and accidental replay of an earlier serve on this instance.
        """
        if self._finished:
            raise RuntimeError("Attempt is finished; create a new ReturnAttempt for each serve")
        position = tuple(float(value) for value in ball_position)
        time = float(time)
        active = set(contacts)
        if (len(position) != 3 or not all(math.isfinite(value) for value in position)
                or not math.isfinite(time) or (self._time is not None and time <= self._time)):
            raise ValueError("Ball position/time must be finite and times must strictly increase")
        if active - self.CONTACT_TYPES:
            raise ValueError("Unknown physical contact type: " + str(sorted(active - self.CONTACT_TYPES)))
        onset = active - self._active_contacts
        had_racket_contact = self.contacted
        self._time = time
        if fallen and not self.fallen:
            self.fallen = True
            self._event("robot_fall", position)

        # Once a legal first far-side landing has occurred, subsequent ball exit
        # is normal; keep monitoring the robot without revoking the ball result.
        if not self._landed:
            for invalid_contact in ("net", "floor", "body"):
                if invalid_contact in onset:
                    self._fail(invalid_contact + "_contact", position)

            if "racket" in onset:
                if self.contacted:
                    self._fail("double_hit", position)
                else:
                    self.contacted = True
                    self._event("racket_contact", position)
                    if not self.own_bounce:
                        self._fail("volley", position)
                    if "table" in onset:
                        self._fail("ambiguous_simultaneous_contact", position)

            # Interpolate actual successive ball positions only for net clearance.
            # Require the hit to have happened on a preceding substep.
            previous = self._previous_position
            if (had_racket_contact and not self.net_clearance and previous is not None
                    and previous[0] <= self.geometry.net_x < position[0]):
                fraction = (self.geometry.net_x - previous[0]) / (position[0] - previous[0])
                crossing_z = previous[2] + fraction * (position[2] - previous[2])
                crossing_y = previous[1] + fraction * (position[1] - previous[1])
                if crossing_z > self.geometry.net_top_z + self.geometry.ball_radius:
                    self.net_clearance = True
                    self._event("net_clearance", (self.geometry.net_x, crossing_y, crossing_z))
                else:
                    self._fail("low_net_crossing", (self.geometry.net_x, crossing_y, crossing_z))

            if "table" in onset:
                inside = self.geometry.contains(position)
                if self.contacted:
                    if self.first_landing is None:
                        self.first_landing = position
                        if not inside or position[0] <= self.geometry.net_x:
                            self._fail("wrong_first_landing", position)
                        elif not self.net_clearance:
                            self._fail("missing_net_clearance", position)
                        else:
                            self._event("far_bounce", position)
                            self._landed = self._failure is None
                elif inside and position[0] < self.geometry.net_x:
                    if self.own_bounce:
                        self._fail("double_bounce", position)
                    else:
                        self.own_bounce = True
                        self._event("own_bounce", position)

        self._active_contacts = active
        self._previous_position = position

    @property
    def success(self) -> bool:
        return self._landed and not self.fallen and not self._aborted and self._failure is None

    def result(self) -> dict:
        if self._aborted:
            reason = "aborted"
        elif self.fallen:
            reason = "robot_fall"
        elif self._failure:
            reason = self._failure
        elif self.success:
            reason = "valid_return"
        elif not self.own_bounce:
            reason = "no_own_bounce"
        elif not self.contacted:
            reason = "no_racket_contact"
        elif not self.net_clearance:
            reason = "no_net_clearance"
        else:
            reason = "no_far_bounce"
        return {
            "contacted": self.contacted,
            "success": self.success,
            "fallen": self.fallen,
            "reason": reason,
            "own_bounce": self.own_bounce,
            "net_clearance": self.net_clearance,
            "first_landing": list(self.first_landing) if self.first_landing is not None else None,
            "finished": self._finished,
            "trace": [{**event, "position": list(event["position"])} for event in self.trace],
        }

    def finish(self, *, aborted: bool = False) -> dict:
        """Close this serve; any reset/respawn before completion must be aborted."""
        self._aborted |= bool(aborted)
        if not self._finished:
            self._finished = True
            if self._previous_position is not None:
                self._event("aborted" if self._aborted else "finished", self._previous_position)
        return self.result()
