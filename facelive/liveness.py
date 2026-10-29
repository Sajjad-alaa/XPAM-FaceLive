"""Liveness checks: repeated-frame detection and a head-movement challenge.

What this stops: a still image (a photo, or a picture sent through a virtual camera) and a short video
that loops the same frames. A person who turns or nods on request passes.
What it does not stop: a video of the real user doing the same movement (replayed live, or edited), a
tilted printed photo that gives a similar landmark shift, a 3D mask, or a malicious camera driver.
Treat it as a second factor with basic presentation-attack resistance, not as certified anti-spoofing.
"""
import hashlib
from collections import deque

import cv2
import numpy as np

SHAKE = "shake_head"
NOD = "nod_head"

TEXTS = {
    SHAKE: "Turn your head left and right, then face forward",
    NOD: "Nod once, then face forward",
}

# How much the head must move (see analyzer.Pose): nose offset in eye distances for a turn, change of the
# nose position between eyes and mouth for a nod. Turning 16 degrees is about 0.15; nodding 10 degrees about 0.09.
DEFAULT_YAW_TURN = 0.15
DEFAULT_PITCH_NOD = 0.09
RETURN_FRACTION = 0.5    # back to "forward" when within this part of the needed movement


class ReplayGuard:
    """Flags frames that are exact copies of a recent frame, or a camera that sends no new picture."""

    def __init__(self, history=90, static_difference=0.05):
        self._seen = deque(maxlen=history)
        self._seen_set = set()
        self._previous = None
        self._static_difference = static_difference

    def check(self, raw, frame):
        """None for a fresh frame, otherwise a short reason ("duplicate" or "static")."""
        digest = hashlib.sha256(raw).digest()[:16]
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (48, 48), interpolation=cv2.INTER_AREA)
        small = small.astype(np.float32)
        reason = None
        if digest in self._seen_set:
            reason = "duplicate"
        elif self._previous is not None and float(np.mean(np.abs(small - self._previous))) < self._static_difference:
            reason = "static"
        if len(self._seen) == self._seen.maxlen:
            self._seen_set.discard(self._seen[0])
        self._seen.append(digest)
        self._seen_set.add(digest)
        self._previous = small
        return reason


class Challenge:
    """One active challenge, judged from the pose change against the calibrated forward position."""

    def __init__(self, kind, now, timeout, base_u, base_v, yaw_turn=DEFAULT_YAW_TURN, pitch_nod=DEFAULT_PITCH_NOD):
        self.kind = kind
        self.text = TEXTS[kind]
        self.started = now
        self.deadline = now + timeout
        self.base_u = base_u
        self.base_v = base_v
        self.yaw_turn = yaw_turn
        self.pitch_nod = pitch_nod
        self.min_du = 0.0
        self.max_du = 0.0
        self.min_dv = 0.0
        self.max_dv = 0.0
        self.moved = False   # the needed movement was seen
        self.passed = False
        self.failed = False
        self.reason = "waiting"

    def remaining(self, now):
        return max(0.0, self.deadline - now)

    def reset_progress(self):
        self.min_du = self.max_du = self.min_dv = self.max_dv = 0.0
        self.moved = False
        self.reason = "waiting"

    def score(self):
        if self.passed:
            return 1.0
        if self.kind == SHAKE:
            left = min(1.0, max(0.0, -self.min_du) / self.yaw_turn)
            right = min(1.0, max(0.0, self.max_du) / self.yaw_turn)
            value = (left + right) / 2.0
        else:
            value = min(1.0, max(abs(self.min_dv), abs(self.max_dv)) / self.pitch_nod)
        return round(min(value, 0.95), 3)

    def update(self, pose, now):
        """Feed one frame's pose. Returns True when the challenge is just passed."""
        if self.passed or self.failed:
            return False
        if now > self.deadline:
            self.failed = True
            self.reason = "timeout"
            return False
        du = pose.yaw_u - self.base_u
        dv = pose.pitch_v - self.base_v
        self.min_du, self.max_du = min(self.min_du, du), max(self.max_du, du)
        self.min_dv, self.max_dv = min(self.min_dv, dv), max(self.max_dv, dv)
        if self.kind == SHAKE:
            if -self.min_du >= self.yaw_turn and self.max_du >= self.yaw_turn:
                self.moved = True
            back = abs(du) <= self.yaw_turn * RETURN_FRACTION
        else:
            if max(abs(self.min_dv), abs(self.max_dv)) >= self.pitch_nod:
                self.moved = True
            back = abs(dv) <= self.pitch_nod * RETURN_FRACTION
        if self.moved and back:
            self.passed = True
            self.reason = "passed"
            return True
        self.reason = "return_forward" if self.moved else "waiting"
        return False
