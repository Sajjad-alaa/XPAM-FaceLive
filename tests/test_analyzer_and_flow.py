"""Recognition on real photos, and the capture flow (guidance, replay, challenge, enrol, sign-in)."""
import asyncio
import random
import unittest

import numpy as np

from facelive import codes
from facelive.analyzer import FaceAnalyzer
from facelive.flow import AUTH, ENROLL, CompareUnavailable, FlowSession
from facelive.liveness import NOD, SHAKE

from helpers import (Clock, FakeAnalyzer, face_crop, jpeg, live_variant, models_available, noisy_frame, photo,
                     rng as make_rng, settings)

THRESHOLD = 0.40   # core's default FACE_RECOGNITION_COSINE_THRESHOLD


@unittest.skipUnless(models_available(), "model files are not in FACELIVE_MODEL_DIR")
class RealPhotosTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analyzer = FaceAnalyzer(settings().model_dir)

    def feature(self, name):
        image = photo(name, 640)
        faces = self.analyzer.detect(image)
        self.assertEqual(len(faces), 1, name)
        return self.analyzer.embed(image, faces[0])

    def test_same_person_matches_and_different_person_does_not(self):
        a1, a2, b1 = self.feature("person_a_1"), self.feature("person_a_2"), self.feature("person_b_1")
        self.assertGreater(codes.cosine(a1, a2), THRESHOLD + 0.1)
        self.assertLess(codes.cosine(a1, b1), THRESHOLD)
        self.assertLess(codes.cosine(a2, b1), THRESHOLD)

    def test_a_frontal_photo_has_a_straight_pose(self):
        face = self.analyzer.detect(photo("person_a_1", 640))[0]
        self.assertLess(abs(face.pose.yaw_deg), 8)
        self.assertLess(abs(face.pose.roll_deg), 8)

    def test_a_blank_picture_has_no_face(self):
        self.assertEqual(self.analyzer.detect(np.full((360, 480, 3), 127, dtype=np.uint8)), [])

    def test_wrong_checksum_is_refused(self):
        import os
        import shutil
        import tempfile
        folder = tempfile.mkdtemp()
        for name in os.listdir(settings().model_dir):
            if name.endswith(".onnx"):
                shutil.copy(os.path.join(settings().model_dir, name), folder)
        with open(os.path.join(folder, "face_detection_yunet_2023mar.onnx"), "ab") as handle:
            handle.write(b"x")
        with self.assertRaises(RuntimeError):
            FaceAnalyzer(folder)


class Driver:
    """Feeds frames to a FlowSession with a fake clock and keeps the last result."""

    def __init__(self, analyzer, mode=AUTH, compare=None, clock=None, rng_seed=1, **changes):
        self.clock = clock or Clock()
        self.random = make_rng(rng_seed)
        self.compare_calls = []

        async def default_compare(code):
            self.compare_calls.append(code)
            return True, 0.8
        self.flow = FlowSession(settings(**changes), mode, analyzer, compare or default_compare,
                                target_name="Test User", threshold=THRESHOLD, clock=self.clock,
                                rng=random.Random(rng_seed))
        self.result = None
        self.events = []

    def step(self, image=None, tick=0.25):
        self.clock.tick(tick)
        image = noisy_frame(self.random) if image is None else image
        self.result = asyncio.run(self.flow.process_frame(image, jpeg(image)))
        self.events.extend(e["event"] for e in self.result["flow"]["events"])
        return self.result


STRAIGHT = (0.0, 0.55)


def with_pose(driver, analyzer, poses):
    analyzer.poses = list(poses)
    return [driver.step() for _ in poses]


class ChallengeTest(unittest.TestCase):
    def make(self, mode=AUTH, seed=1, **changes):
        analyzer = FakeAnalyzer()
        driver = Driver(analyzer, mode=mode, rng_seed=seed, **changes)
        return analyzer, driver

    def kind_of(self, driver):
        return driver.result["flow"]["challenge"]["type"]

    def calibrate(self, analyzer, driver):
        with_pose(driver, analyzer, [STRAIGHT] * 3)
        self.assertEqual(driver.flow.phase, "challenge")
        self.assertIn("challenge_started", driver.events)

    def test_calibration_comes_first_and_the_ui_flags_are_set(self):
        analyzer, driver = self.make()
        results = with_pose(driver, analyzer, [STRAIGHT, STRAIGHT])
        challenge = results[-1]["flow"]["challenge"]
        self.assertFalse(challenge["calibrated"])
        self.assertFalse(challenge["passed"])
        self.assertEqual(results[-1]["flow"]["status"], "authenticating")
        self.assertFalse(results[-1]["flow"]["finished"])

    def run_challenge(self, driver, analyzer, kind):
        if kind == SHAKE:
            return [(-0.2, 0.55), (-0.22, 0.55), (0.0, 0.55), (0.2, 0.55), (0.22, 0.55)], [(0.0, 0.55), (0.0, 0.55)]
        return [(0.0, 0.66), (0.0, 0.68)], [(0.0, 0.55), (0.0, 0.55)]

    def test_shake_finishes_only_after_both_sides_and_the_return(self):
        for seed in range(1, 40):
            analyzer, driver = self.make(seed=seed)
            self.calibrate(analyzer, driver)
            if self.kind_of(driver) == SHAKE:
                break
        else:
            self.fail("no seed gave a shake challenge")
        # looking straight again does nothing
        with_pose(driver, analyzer, [STRAIGHT] * 4)
        self.assertFalse(driver.result["flow"]["challenge"]["passed"])
        # only one side
        with_pose(driver, analyzer, [(-0.2, 0.55), (-0.22, 0.55), STRAIGHT])
        self.assertFalse(driver.result["flow"]["challenge"]["passed"])
        self.assertEqual(driver.flow.phase, "challenge")
        self.assertTrue(0 < driver.result["flow"]["challenge"]["score"] < 1)
        # the other side, but not back yet
        with_pose(driver, analyzer, [(0.2, 0.55), (0.22, 0.55)])
        self.assertFalse(driver.result["flow"]["challenge"]["passed"])
        self.assertNotIn("challenge_passed", driver.events)
        # back forward: passed, and sign-in verification starts
        with_pose(driver, analyzer, [STRAIGHT])
        self.assertIn("challenge_passed", driver.events)
        self.assertTrue(driver.result["flow"]["challenge"]["passed"])
        self.assertEqual(driver.flow.phase, "verifying")
        self.assertFalse(driver.flow.finished)

    def test_nod_finishes_only_after_the_movement(self):
        for seed in range(1, 40):
            analyzer, driver = self.make(seed=seed)
            self.calibrate(analyzer, driver)
            if self.kind_of(driver) == NOD:
                break
        else:
            self.fail("no seed gave a nod challenge")
        with_pose(driver, analyzer, [STRAIGHT] * 4 + [(0.0, 0.57)])   # tiny wobble is not a nod
        self.assertFalse(driver.result["flow"]["challenge"]["passed"])
        with_pose(driver, analyzer, [(0.0, 0.66), (0.0, 0.68)])
        self.assertFalse(driver.result["flow"]["challenge"]["passed"])
        with_pose(driver, analyzer, [STRAIGHT])
        self.assertTrue(driver.result["flow"]["challenge"]["passed"])

    def test_signin_completes_after_challenge_with_core_comparisons(self):
        analyzer, driver = self.make(seed=2)
        self.calibrate(analyzer, driver)
        kind = self.kind_of(driver)
        moves, back = self.run_challenge(driver, analyzer, kind)
        with_pose(driver, analyzer, moves + back)
        self.assertEqual(driver.flow.phase, "verifying")
        with_pose(driver, analyzer, [STRAIGHT] * 6)
        self.assertTrue(driver.flow.finished)
        self.assertEqual(driver.result["flow"]["status"], "auth_success")
        self.assertTrue(driver.flow.outcome.success)
        self.assertTrue(driver.flow.outcome.face_code.startswith("xpamf1:"))
        self.assertEqual(len(driver.compare_calls), 2)
        self.assertEqual(driver.result["flow"]["accepted_samples"], 2)
        self.assertIn("auth_success", driver.events)

    def test_a_different_face_after_the_challenge_is_rejected(self):
        analyzer, driver = self.make(seed=2)
        self.calibrate(analyzer, driver)
        moves, back = self.run_challenge(driver, analyzer, self.kind_of(driver))
        with_pose(driver, analyzer, moves)
        other = np.zeros(128)
        other[5] = 1.0
        analyzer.features = [other] * 20          # the face that comes back is another person
        with_pose(driver, analyzer, back[:1])
        self.assertEqual(driver.result["flow"]["guidance"], "liveness")
        self.assertIn("challenge_failed", driver.events)
        with_pose(driver, analyzer, back[1:])
        self.assertNotEqual(driver.flow.phase, "verifying")
        self.assertEqual(driver.flow.phase, "calibrating")
        self.assertFalse(driver.flow.outcome and driver.flow.outcome.success)

    def test_a_timed_out_challenge_is_retried_once_then_fails(self):
        analyzer, driver = self.make(seed=2, challenge_timeout=5.0)
        self.calibrate(analyzer, driver)
        first = self.kind_of(driver)
        with_pose(driver, analyzer, [STRAIGHT] * 21)       # 5.25 s without any movement
        self.assertIn("challenge_failed", driver.events)
        self.assertEqual(driver.flow.phase, "challenge")
        self.assertNotEqual(self.kind_of(driver), first)    # another kind the second time
        with_pose(driver, analyzer, [STRAIGHT] * 22)
        self.assertTrue(driver.flow.finished)
        self.assertEqual(driver.result["flow"]["status"], "auth_failed")
        self.assertFalse(driver.flow.outcome.success)

    def test_losing_the_face_for_long_asks_to_look_straight_again(self):
        analyzer, driver = self.make(seed=2)
        self.calibrate(analyzer, driver)
        with_pose(driver, analyzer, [None] * 3)
        self.assertEqual(driver.result["flow"]["guidance"], "face_missing")
        self.assertEqual(driver.flow.phase, "challenge")
        with_pose(driver, analyzer, [None] * 10)            # more than 3 seconds
        self.assertEqual(driver.flow.phase, "calibrating")

    def test_a_face_that_jumps_resets_the_challenge(self):
        analyzer, driver = self.make(seed=2)
        self.calibrate(analyzer, driver)
        before = driver.flow._challenge
        # a face in a different place in the picture
        original = analyzer.detect

        def jump(frame):
            faces = original(frame)
            for face in faces:
                face.bbox = (face.bbox[0] + 250, face.bbox[1], face.bbox[2] + 250, face.bbox[3])
            return faces
        analyzer.detect = jump
        with_pose(driver, analyzer, [(-0.2, 0.55)])
        self.assertEqual(driver.result["flow"]["guidance"], "liveness")
        self.assertEqual(before.min_du, 0.0)

    def test_without_liveness_the_challenge_is_skipped(self):
        analyzer, driver = self.make(liveness_mode="off")
        with_pose(driver, analyzer, [STRAIGHT] * 3)
        self.assertEqual(driver.flow.phase, "verifying")
        self.assertIsNone(driver.result["flow"]["challenge"])
        self.assertEqual(driver.result["flow"]["liveness"]["reason"], "challenge_disabled")

    def test_enrolment_collects_samples_and_averages_them(self):
        analyzer, driver = self.make(mode=ENROLL, seed=2, enrollment_samples=3)
        self.calibrate(analyzer, driver)
        moves, back = self.run_challenge(driver, analyzer, self.kind_of(driver))
        rng = np.random.default_rng(5)
        samples = [np.ones(128) + rng.normal(size=128) * 0.05 for _ in range(3)]
        analyzer.features = [np.ones(128)] + list(samples)     # (the first one is used to compare after the challenge)
        with_pose(driver, analyzer, moves + back)
        self.assertEqual(driver.flow.phase, "sampling")
        self.assertEqual(driver.result["flow"]["status"], "enrolling")
        with_pose(driver, analyzer, [STRAIGHT] * 2)
        self.assertEqual(driver.result["flow"]["accepted_samples"], 2)     # (one sample every 0.5 s: 0.25 s frames skip)
        self.assertFalse(driver.flow.finished)
        with_pose(driver, analyzer, [STRAIGHT] * 4)
        self.assertTrue(driver.flow.finished)
        self.assertEqual(driver.result["flow"]["status"], "enrollment_complete")
        self.assertEqual(driver.result["flow"]["accepted_samples"], 3)
        code = driver.flow.outcome.face_code
        expected = codes.mean_feature(samples)
        self.assertTrue(np.allclose(codes.decode_code(code), expected, atol=1e-5))
        self.assertEqual(driver.compare_calls, [])                    # enrolment never asks core to compare

    def test_signin_fails_after_too_many_mismatches(self):
        async def never(code):
            return False, 0.1
        analyzer = FakeAnalyzer()
        driver = Driver(analyzer, compare=never, liveness_mode="off")
        with_pose(driver, analyzer, [STRAIGHT] * 3)
        with_pose(driver, analyzer, [STRAIGHT] * 4)
        self.assertEqual(driver.result["flow"]["guidance"], "identity_mismatch")
        self.assertFalse(driver.flow.finished)
        with_pose(driver, analyzer, [STRAIGHT] * 4)
        self.assertTrue(driver.flow.finished)
        self.assertEqual(driver.result["flow"]["status"], "auth_failed")

    def test_core_not_answering_raises_after_three_tries(self):
        async def broken(code):
            raise CompareUnavailable()
        analyzer = FakeAnalyzer()
        driver = Driver(analyzer, compare=broken, liveness_mode="off")
        with_pose(driver, analyzer, [STRAIGHT] * 3)
        analyzer.poses = [STRAIGHT] * 3
        driver.step()
        driver.step()
        with self.assertRaises(CompareUnavailable):
            driver.step()

    def test_two_faces_and_missing_face_give_guidance_and_no_progress(self):
        analyzer, driver = self.make()
        analyzer.faces = 2
        with_pose(driver, analyzer, [STRAIGHT] * 4)
        self.assertEqual(driver.result["flow"]["guidance"], "multiple_faces")
        self.assertEqual(driver.flow.phase, "calibrating")
        analyzer.faces = 1
        with_pose(driver, analyzer, [None])
        self.assertEqual(driver.result["flow"]["guidance"], "face_missing")
        self.assertEqual(driver.result["faces"], [])

    def test_too_far_and_not_straight_give_guidance(self):
        analyzer, driver = self.make(min_face_size=300)
        with_pose(driver, analyzer, [STRAIGHT])
        self.assertEqual(driver.result["flow"]["guidance"], "too_far")
        analyzer, driver = self.make()
        with_pose(driver, analyzer, [(0.3, 0.55)])
        self.assertEqual(driver.result["flow"]["guidance"], "look_straight")

    def test_the_whole_attempt_has_a_time_limit(self):
        analyzer, driver = self.make(liveness_mode="off", capture_timeout=5.0)
        analyzer.poses = [None] * 40
        for _ in range(30):
            driver.step()
        self.assertTrue(driver.flow.finished)
        self.assertEqual(driver.result["flow"]["status"], "auth_failed")


class ReplayTest(unittest.TestCase):
    def test_identical_frames_are_rejected_and_do_not_progress(self):
        analyzer = FakeAnalyzer()
        driver = Driver(analyzer, liveness_mode="off")
        image = noisy_frame(make_rng(3))
        analyzer.poses = [STRAIGHT] * 10
        first = driver.step(image)
        self.assertIsNone(first["flow"]["guidance"])
        for _ in range(5):
            result = driver.step(image)
            self.assertEqual(result["flow"]["guidance"], "replayed_frame")
        self.assertEqual(driver.flow.phase, "calibrating")
        self.assertEqual(analyzer.detect_calls, 1)            # the repeated frames were not even analysed
        self.assertEqual(result["faces"], [])

    def test_a_looped_short_video_is_caught_on_the_second_round(self):
        analyzer = FakeAnalyzer()
        driver = Driver(analyzer, liveness_mode="off")
        rng = make_rng(4)
        loop = [noisy_frame(rng) for _ in range(4)]
        analyzer.poses = [STRAIGHT] * 20
        for image in loop:
            self.assertIsNone(driver.step(image)["flow"]["guidance"])
        for image in loop:
            self.assertEqual(driver.step(image)["flow"]["guidance"], "replayed_frame")

    def test_a_still_picture_resent_as_new_jpegs_is_static(self):
        analyzer = FakeAnalyzer()
        driver = Driver(analyzer, liveness_mode="off")
        image = noisy_frame(make_rng(5))
        analyzer.poses = [STRAIGHT] * 6
        driver.step(image)
        # same pixels, different JPEG bytes
        for quality in (70, 75, 85):
            clock_image = image.copy()
            driver.clock.tick()
            result = asyncio.run(driver.flow.process_frame(clock_image, jpeg(clock_image, quality)))
            self.assertEqual(result["flow"]["guidance"], "replayed_frame")

    def test_lasting_replay_ends_the_attempt(self):
        analyzer = FakeAnalyzer()
        driver = Driver(analyzer, liveness_mode="off")
        image = noisy_frame(make_rng(6))
        for _ in range(40):
            driver.step(image, tick=0.1)
        self.assertTrue(driver.flow.finished)
        self.assertEqual(driver.result["flow"]["status"], "auth_failed")

    def test_live_camera_noise_is_not_flagged(self):
        analyzer = FakeAnalyzer()
        driver = Driver(analyzer, liveness_mode="off")
        base = noisy_frame(make_rng(7))
        rng = make_rng(8)
        analyzer.poses = [STRAIGHT] * 8
        for _ in range(8):
            self.assertNotEqual(driver.step(live_variant(base, rng))["flow"]["guidance"], "replayed_frame")


@unittest.skipUnless(models_available(), "model files are not in FACELIVE_MODEL_DIR")
class RealPhotoFlowTest(unittest.TestCase):
    """The whole flow on real photos (liveness challenge off: a photo cannot turn its head)."""

    @classmethod
    def setUpClass(cls):
        cls.analyzer = FaceAnalyzer(settings().model_dir)

    def frames(self, name, count, seed=1):
        rng = make_rng(seed)
        base = face_crop(name, self.analyzer, 480) if name == "person_a_2" else photo(name, 640)
        return [live_variant(base, rng) for _ in range(count)]

    def run_frames(self, driver, images):
        for image in images:
            driver.step(image)
            if driver.flow.finished:
                break
        return driver

    def test_enrol_then_same_person_matches_and_other_person_does_not(self):
        enrol = self.run_frames(Driver(self.analyzer, mode=ENROLL, liveness_mode="off", enrollment_samples=3),
                                self.frames("person_a_1", 20))
        self.assertEqual(enrol.result["flow"]["status"], "enrollment_complete")
        template = codes.decode_code(enrol.flow.outcome.face_code)

        async def core_compare(code):
            similarity = codes.cosine(template, codes.decode_code(code))
            return similarity >= THRESHOLD, similarity

        same = self.run_frames(Driver(self.analyzer, compare=core_compare, liveness_mode="off"),
                               self.frames("person_a_2", 30, seed=2))
        self.assertEqual(same.result["flow"]["status"], "auth_success", same.result["flow"])
        self.assertGreaterEqual(same.result["flow"]["similarity"], THRESHOLD)

        other = self.run_frames(Driver(self.analyzer, compare=core_compare, liveness_mode="off"),
                                self.frames("person_b_1", 30, seed=3))
        self.assertEqual(other.result["flow"]["status"], "auth_failed")
        self.assertFalse(other.flow.outcome.success)

    def test_no_face_and_two_faces_give_guidance(self):
        driver = Driver(self.analyzer, liveness_mode="off")
        blank = np.full((360, 480, 3), 120, dtype=np.uint8)
        result = driver.step(blank)
        self.assertEqual(result["flow"]["guidance"], "face_missing")
        left, right = photo("person_a_1", 320), photo("person_b_1", 320)
        height = min(left.shape[0], right.shape[0])
        both = np.hstack([left[:height], right[:height]])
        result = driver.step(both)
        self.assertEqual(result["flow"]["guidance"], "multiple_faces")
        self.assertEqual(len(result["faces"]), 2)
        self.assertEqual(driver.flow.phase, "calibrating")

    def test_the_same_photo_replayed_is_rejected(self):
        driver = Driver(self.analyzer, liveness_mode="off")
        image = photo("person_a_1", 640)
        driver.step(image)
        result = driver.step(image)
        self.assertEqual(result["flow"]["guidance"], "replayed_frame")

    def test_frame_result_has_what_the_page_reads(self):
        driver = Driver(self.analyzer, liveness_mode="off")
        result = driver.step(live_variant(photo("person_a_1", 640), make_rng(1)))
        self.assertEqual(set(result), {"type", "frame", "faces", "flow"})
        self.assertEqual(result["frame"], {"width": 640, "height": result["frame"]["height"]})
        face = result["faces"][0]
        self.assertEqual(set(face), {"bbox", "name", "is_match", "similarity", "detection_score", "pose", "landmarks", "liveness"})
        self.assertEqual(len(face["bbox"]), 4)
        self.assertEqual(len(face["pose"]), 3)
        self.assertEqual(len(face["landmarks"]), 5)
        flow = result["flow"]
        for key in ("mode", "phase", "status", "detail", "guidance", "finished", "target_name", "similarity",
                    "threshold", "accepted_samples", "required_samples", "challenge", "liveness", "monitor", "events"):
            self.assertIn(key, flow)


if __name__ == "__main__":
    unittest.main()
