"""The capture flow of one WebSocket connection: enrol a face or check a face.

Steps: look straight (calibrate) -> head-movement challenge -> samples (enrol) or comparison with core (auth).
The flow keeps only small numbers in memory (pose values and 128-number features). It never stores images.
"""
import asyncio
import random
import time
from datetime import datetime, timezone
from dataclasses import dataclass

import numpy as np

from . import codes
from .liveness import Challenge, ReplayGuard, DEFAULT_PITCH_NOD, DEFAULT_YAW_TURN, SHAKE, NOD

ENROLL = "enroll"
AUTH = "auth"

CALIBRATION_FRAMES = 3          # frames looking straight before the challenge starts
FRONTAL_YAW_DEG = 12.0          # "looking straight" limits
FRONTAL_ROLL_DEG = 20.0
FACE_MAX_JUMP = 0.9             # the face may not move more than this many face widths between two frames
MISSING_RESTART_SECONDS = 3.0   # no face for this long during the challenge: look straight again
CONTINUITY_COSINE = 0.5         # the face after the challenge must be the face before it
SAMPLE_GAP_SECONDS = 0.5        # between two enrolment samples
VERIFY_GAP_SECONDS = 0.2        # between two comparisons with core
SAMPLE_MIN_SCORE = 0.9          # detector confidence for a sample
AUTH_SAMPLES = 2                # matching frames needed after the challenge
AUTH_MAX_MISMATCHES = 6         # non-matching frames that end the attempt
MAX_CHALLENGE_TRIES = 2
MAX_SWAPS = 2                   # a different face after the challenge: end the attempt
MAX_BAD_RUN = 4                 # frames in a row whose face is not the face that started the capture
MAX_REPLAY_STRIKES = 25
SECOND_FACE_RATIO = 0.10        # another face this wide (of the frame) counts as a second person
COMPARE_ERRORS_FATAL = 3

# Messages the person may read (core shows the failure text on the sign-in page)
FAIL_TEXT = {
    "timeout": "Face capture timed out",
    "liveness": "Liveness check failed",
    "replay": "The camera image looks repeated or static",
    "mismatch": "Facial comparison failed",
}


@dataclass
class Outcome:
    success: bool
    face_code: str = ""
    error_message: str = ""


class CompareUnavailable(Exception):
    """Core could not answer a comparison."""


def _now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class FlowSession:
    def __init__(self, settings, mode, analyzer, compare, target_name="", threshold=0.4,
                 executor=None, clock=time.monotonic, rng=None,
                 yaw_turn=DEFAULT_YAW_TURN, pitch_nod=DEFAULT_PITCH_NOD):
        """compare(face_code) -> (matched, similarity) is core's answer; used in auth mode only."""
        self.settings = settings
        self.mode = mode
        self.analyzer = analyzer
        self.compare = compare
        self.target_name = target_name
        self.threshold = float(threshold)
        self.executor = executor
        self.clock = clock
        self.rng = rng or random.SystemRandom()
        self.yaw_turn = yaw_turn
        self.pitch_nod = pitch_nod

        self.required = settings.enrollment_samples if mode == ENROLL else AUTH_SAMPLES
        self.use_challenge = settings.liveness_mode == "challenge"
        self.replay = ReplayGuard()
        self.started = None
        self.status = "enrolling" if mode == ENROLL else "authenticating"
        self.phase = "calibrating"
        self.detail = ""
        self.guidance = None
        self.finished = False
        self.outcome = None
        self.similarity = 0.0
        self.events = []

        self._calibration = []        # (yaw_u, pitch_v) of straight frames in a row
        self._base = None             # (yaw_u, pitch_v)
        self._reference = None        # feature of the face that started the capture
        self._challenge = None
        self._challenge_passed = False
        self._tries = 0
        self._swaps = 0
        self._bad_run = 0
        self._last_matched = False
        self._replay_strikes = 0
        self._last_center = None
        self._last_seen = None
        self._samples = []            # features accepted (enrol) or matched (auth)
        self._last_sample_at = -1e9
        self._mismatches = 0
        self._compare_errors = 0
        self._fail_reason = ""

    # ------------------------------------------------------------------ helpers

    def _event(self, name, **extra):
        item = {"event": name, "flow_mode": self.mode, "occurred_at": _now_iso()}
        if self.started is not None:
            item["elapsed_seconds"] = round(self.clock() - self.started, 2)
        if self.target_name:
            item["target_name"] = self.target_name
        item.update(extra)
        self.events.append(item)

    async def _run(self, function, *args):
        return await asyncio.get_running_loop().run_in_executor(self.executor, function, *args)

    def _deadline(self):
        total = self.settings.capture_timeout
        if self.use_challenge:
            total += self.settings.challenge_timeout * MAX_CHALLENGE_TRIES
        return total

    def _finish(self, success, reason="", face_code=""):
        self.finished = True
        self.guidance = None
        if success:
            self.phase = "completed"
            self.status = "enrollment_complete" if self.mode == ENROLL else "auth_success"
            self.detail = "Face captured" if self.mode == ENROLL else "Face verified"
            self.outcome = Outcome(True, face_code=face_code)
        else:
            self.phase = "failed"
            self.status = "enrollment_failed" if self.mode == ENROLL else "auth_failed"
            self.detail = FAIL_TEXT.get(reason, FAIL_TEXT["mismatch"])
            self.outcome = Outcome(False, error_message=self.detail)
            self._fail_reason = reason
        self._event(self.status, samples=len(self._samples))

    def _challenge_state(self, now):
        if self._challenge is None and not self._challenge_passed:
            if not self.use_challenge:
                return None
            return {"type": None, "text": "", "passed": False, "score": 0.0, "reason": "calibrating",
                    "remaining_seconds": 0, "phase": "calibrating", "calibrated": False}
        if self._challenge_passed:
            kind = self._challenge.kind if self._challenge else None
            return {"type": kind, "text": self._challenge.text if self._challenge else "", "passed": True,
                    "score": 1.0, "reason": "passed", "remaining_seconds": 0, "phase": "passed",
                    "calibrated": True}
        challenge = self._challenge
        return {"type": challenge.kind, "text": challenge.text, "passed": False, "score": challenge.score(),
                "reason": challenge.reason, "remaining_seconds": round(challenge.remaining(now), 1),
                "phase": "running", "calibrated": True}

    def _liveness(self):
        if not self.use_challenge:
            return {"is_live": True, "score": 0.0, "reason": "challenge_disabled"}
        if self._challenge_passed:
            return {"is_live": True, "score": 1.0, "reason": "challenge_passed"}
        score = self._challenge.score() if self._challenge else 0.0
        return {"is_live": False, "score": score, "reason": "challenge_pending"}

    def _phase_detail(self):
        if self.finished:
            return self.detail
        if self.phase == "challenge" and self._challenge is not None:
            return self._challenge.text
        return {
            "calibrating": "Look straight at the camera",
            "challenge": "Follow the movement",
            "sampling": "Capturing face samples",
            "verifying": "Verifying your face",
        }.get(self.phase, "")

    def _flow(self, now):
        events, self.events = self.events, []
        return {
            "mode": self.mode,
            "phase": self.phase,
            "status": self.status,
            "detail": self._phase_detail(),
            "guidance": self.guidance,
            "finished": self.finished,
            "target_name": self.target_name,
            "similarity": round(self.similarity, 4),
            "threshold": self.threshold,
            "accepted_samples": len(self._samples) if self.mode == ENROLL else min(len(self._samples), self.required),
            "required_samples": self.required,
            "challenge": self._challenge_state(now),
            "liveness": self._liveness(),
            "monitor": None,
            "events": events,
        }

    def _face_item(self, detection, matched=False, similarity=0.0):
        pose = detection.pose
        return {
            "bbox": [round(v, 1) for v in detection.bbox],
            "name": self.target_name if self.mode == AUTH else "",
            "is_match": bool(matched),
            "similarity": round(float(similarity), 4),
            "detection_score": round(detection.score, 4),
            "pose": [round(pose.yaw_deg, 1), round(pose.pitch_deg, 1), round(pose.roll_deg, 1)],
            "landmarks": [[round(x, 1), round(y, 1)] for x, y in detection.landmarks],
            "liveness": self._liveness(),
        }

    def _result(self, frame, faces, now):
        height, width = frame.shape[:2]
        return {
            "type": "frame_result",
            "frame": {"width": int(width), "height": int(height)},
            "faces": faces,
            "flow": self._flow(now),
        }

    # ------------------------------------------------------------------ the frame

    async def process_frame(self, frame, raw):
        """One camera frame (BGR array) and its encoded bytes. Returns the frame_result message."""
        now = self.clock()
        if self.started is None:
            self.started = now
        if self.finished:
            return self._result(frame, [], now)
        if now - self.started > self._deadline():
            self._finish(False, "timeout")
            return self._result(frame, [], now)

        # 1. the same picture again, or a camera that sends no new picture
        reason = self.replay.check(raw, frame)
        if reason:
            self._replay_strikes += 1
            self.guidance = "replayed_frame"
            if self._replay_strikes >= MAX_REPLAY_STRIKES:
                self._finish(False, "replay")
            return self._result(frame, [], now)
        self._replay_strikes = 0

        detections = await self._run(self.analyzer.detect, frame)
        height, width = frame.shape[:2]
        faces = [self._face_item(d) for d in detections]

        # 2. how many faces, how far, how much of the face is in the picture
        big = [d for d in detections if d.width >= SECOND_FACE_RATIO * width]
        if len(big) > 1:
            self.guidance = "multiple_faces"
            self._lost(now)
            return self._result(frame, faces, now)
        if not detections:
            self.guidance = "face_missing"
            self._lost(now)
            return self._result(frame, faces, now)
        face = detections[0]
        self._last_seen = now
        x1, y1, x2, y2 = face.bbox
        margin = 0.05 * face.width
        if x1 < -margin or y1 < -margin or x2 > width + margin or y2 > height + margin:
            self.guidance = "face_missing"
            return self._result(frame, faces, now)
        too_small = face.width < self.settings.min_face_size * (width / 480.0)
        if too_small or (face.width * face.height) / float(width * height) < self.settings.min_face_ratio:
            self.guidance = "too_far"
            return self._result(frame, faces, now)

        # 3. the face must stay in one place (a swap of the picture shows as a jump)
        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        jumped = False
        if self._last_center is not None:
            jumped = np.hypot(center[0] - self._last_center[0], center[1] - self._last_center[1]) > FACE_MAX_JUMP * face.width
        self._last_center = center
        if jumped and self._challenge is not None and not self._challenge_passed:
            self._challenge.reset_progress()
            self.guidance = "liveness"
            return self._result(frame, faces, now)

        self.guidance = None
        if self.phase == "calibrating":
            await self._calibrate(frame, face, now)
        elif self.phase == "challenge":
            await self._run_challenge(frame, face, now)
        elif self.phase == "sampling":
            await self._sample(frame, face, now)
        elif self.phase == "verifying":
            await self._verify(frame, face, now)
        faces[0] = self._face_item(face, self.mode == AUTH and self._last_matched, self.similarity)
        return self._result(frame, faces, now)

    def _lost(self, now):
        """A frame without a usable face during the challenge."""
        if self.phase == "challenge" and self._last_seen is not None and not self._challenge_passed:
            if now - self._last_seen > MISSING_RESTART_SECONDS:
                self._restart_calibration()

    def _restart_calibration(self):
        self.phase = "calibrating"
        self._calibration = []
        self._challenge = None
        self._last_seen = None
        self._last_center = None

    @staticmethod
    def _frontal(pose):
        return abs(pose.yaw_deg) <= FRONTAL_YAW_DEG and abs(pose.roll_deg) <= FRONTAL_ROLL_DEG

    # ------------------------------------------------------------------ steps

    async def _calibrate(self, frame, face, now):
        pose = face.pose
        if not self._frontal(pose):
            self.guidance = "look_straight"
            self._calibration = []
            return
        self._calibration.append((pose.yaw_u, pose.pitch_v))
        if len(self._calibration) < CALIBRATION_FRAMES:
            return
        self._base = (float(np.median([p[0] for p in self._calibration])),
                      float(np.median([p[1] for p in self._calibration])))
        self._reference = await self._run(self.analyzer.embed, frame, face)
        self._calibration = []
        if self.use_challenge:
            self._start_challenge(now)
        else:
            self._after_challenge()

    def _start_challenge(self, now):
        kinds = [SHAKE, NOD]
        if self._challenge is not None and len(kinds) > 1:
            kinds = [k for k in kinds if k != self._challenge.kind] or kinds
        kind = self.rng.choice(kinds)
        self._tries += 1
        self._challenge = Challenge(kind, now, self.settings.challenge_timeout, self._base[0], self._base[1],
                                    self.yaw_turn, self.pitch_nod)
        self.phase = "challenge"
        self._event("challenge_started", challenge=kind)

    async def _run_challenge(self, frame, face, now):
        challenge = self._challenge
        just_passed = challenge.update(face.pose, now)
        if challenge.failed:
            self._event("challenge_failed", challenge=challenge.kind, reason=challenge.reason)
            if self._tries >= MAX_CHALLENGE_TRIES:
                self._finish(False, "liveness")
            else:
                self._start_challenge(now)
            return
        if not just_passed:
            return
        # the face that comes back must be the face that started the capture
        feature = await self._run(self.analyzer.embed, frame, face)
        if codes.cosine(feature, self._reference) < CONTINUITY_COSINE:
            self._swaps += 1
            self.guidance = "liveness"
            self._event("challenge_failed", challenge=challenge.kind, reason="face_changed")
            if self._swaps >= MAX_SWAPS or self._tries >= MAX_CHALLENGE_TRIES:
                self._finish(False, "liveness")
            else:
                self._challenge = None
                self._restart_calibration()
            return
        self._challenge_passed = True
        self._event("challenge_passed", challenge=challenge.kind)
        self._after_challenge()

    def _after_challenge(self):
        self.phase = "sampling" if self.mode == ENROLL else "verifying"
        self._last_sample_at = -1e9

    def _changed_face(self, feature):
        """True (and the attempt may end) when this face is not the one that started the capture."""
        if codes.cosine(feature, self._reference) >= CONTINUITY_COSINE:
            self._bad_run = 0
            return False
        self._bad_run += 1
        self.guidance = "liveness"
        if self._bad_run >= MAX_BAD_RUN:
            self._finish(False, "liveness")
        return True

    async def _sample(self, frame, face, now):
        """Enrolment: collect straight, clear, same-person samples."""
        if not self._frontal(face.pose):
            self.guidance = "look_straight"
            return
        if face.score < SAMPLE_MIN_SCORE or now - self._last_sample_at < SAMPLE_GAP_SECONDS:
            return
        feature = await self._run(self.analyzer.embed, frame, face)
        if self._changed_face(feature):
            return
        self._last_sample_at = now
        self._samples.append(feature)
        self._event("sample_accepted", samples=len(self._samples))
        if len(self._samples) >= self.required:
            code = codes.encode_code(codes.mean_feature(self._samples))
            self._finish(True, face_code=code)

    async def _verify(self, frame, face, now):
        """Sign-in: ask core whether this face is the user's, on straight frames after the challenge."""
        if not self._frontal(face.pose):
            self.guidance = "look_straight"
            return
        if now - self._last_sample_at < VERIFY_GAP_SECONDS:
            return
        feature = await self._run(self.analyzer.embed, frame, face)
        if self._changed_face(feature):
            return
        self._last_sample_at = now
        try:
            matched, similarity = await self.compare(codes.encode_code(feature))
        except CompareUnavailable:
            self._compare_errors += 1
            if self._compare_errors >= COMPARE_ERRORS_FATAL:
                raise
            return
        self._compare_errors = 0
        self.similarity = float(similarity)
        self._last_matched = bool(matched)
        if not matched:
            self._mismatches += 1
            self.guidance = "identity_mismatch"
            if self._mismatches >= AUTH_MAX_MISMATCHES:
                self._finish(False, "mismatch")
            return
        self._samples.append(feature)
        self._event("sample_accepted", samples=len(self._samples))
        if len(self._samples) >= self.required:
            # (the average of features that each match is at least as close to the stored face as their mean)
            self._finish(True, face_code=codes.encode_code(codes.mean_feature(self._samples)))
