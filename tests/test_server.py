"""The WebSocket server end to end, with a fake core and the real models."""
import asyncio
import base64
import json
import unittest
import urllib.request
from urllib.error import HTTPError

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from facelive import codes
from facelive.analyzer import FaceAnalyzer
from facelive.core import CoreError
from facelive.server import FaceServer, decode_frame, jpeg_size, FrameError

from helpers import face_crop, jpeg, live_variant, models_available, photo, rng, settings

TOKEN = "a1b2c3d4e5f60718293a4b5c6d7e8f90"
THRESHOLD = 0.40


class FakeCore:
    def __init__(self, mode="enroll", template=None):
        self.ready = True
        self.mode = mode
        self.template = template
        self.callbacks = []
        self.compare_calls = 0
        self.known = {TOKEN}
        self.fail_callback = False

    async def context(self, token):
        if token not in self.known:
            return None
        return {"valid": True, "mode": self.mode, "target_name": "Test User", "threshold": THRESHOLD}

    async def compare(self, token, code):
        self.compare_calls += 1
        similarity = codes.cosine(self.template, codes.decode_code(code))
        return similarity >= THRESHOLD, similarity

    async def callback(self, token, success, face_code="", error_message=""):
        if self.fail_callback:
            raise CoreError("down")
        self.callbacks.append({"token": token, "success": success, "face_code": face_code, "error_message": error_message})


def data_url(image, quality=80):
    return "data:image/jpeg;base64," + base64.b64encode(jpeg(image, quality)).decode()


@unittest.skipUnless(models_available(), "model files are not in FACELIVE_MODEL_DIR")
class ServerTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.analyzer = FaceAnalyzer(settings().model_dir)

    async def start(self, core, **changes):
        options = {"liveness_mode": "off", "frame_min_interval": 0.0}
        options.update(changes)
        config = settings(**options)
        self.server = FaceServer(config, self.analyzer, core)
        self.context = self.server.listen("127.0.0.1", 0)
        running = await self.context.__aenter__()
        self.port = running.sockets[0].getsockname()[1]
        self.addAsyncCleanup(self.context.__aexit__, None, None, None)
        return self.server

    def url(self, token=TOKEN, mode="capture", path="/ws/facelive/capture/"):
        return "ws://127.0.0.1:%d%s?token=%s&mode=%s" % (self.port, path, token, mode)

    async def recv(self, ws):
        return json.loads(await asyncio.wait_for(ws.recv(), 20))

    async def test_enrol_over_the_websocket(self):
        core = FakeCore("enroll")
        await self.start(core, enrollment_samples=3)
        base = photo("person_a_1", 480)
        random = rng(1)
        async with connect(self.url()) as ws:
            ready = await self.recv(ws)
            self.assertEqual(ready["type"], "ready")
            self.assertEqual(ready["mode"], "enroll")
            self.assertEqual(ready["frame_interval_ms"], 250)
            final = None
            for _ in range(30):
                await ws.send(json.dumps({"type": "frame", "image": data_url(live_variant(base, random))}))
                message = await self.recv(ws)
                self.assertEqual(message["type"], "frame_result")
                if message["flow"]["finished"]:
                    final = message
                    break
                await asyncio.sleep(0.1)
            self.assertIsNotNone(final)
            self.assertEqual(final["flow"]["status"], "enrollment_complete")
            self.assertEqual(final["flow"]["accepted_samples"], 3)
            self.assertEqual(final["frame"]["width"], 480)
            self.assertEqual(len(core.callbacks), 1)
            callback = core.callbacks[0]
            self.assertTrue(callback["success"])
            self.assertTrue(callback["face_code"].startswith("xpamf1:"))
            self.assertEqual(callback["token"], TOKEN)
            # the face code is not in what the page receives
            self.assertNotIn("xpamf1", json.dumps(final))
            with self.assertRaises(ConnectionClosed):       # the server ends the capture
                await asyncio.wait_for(ws.recv(), 5)

    async def test_signin_success_and_failure_are_reported_to_core(self):
        # a template made by enrolling person A
        driver_core = FakeCore("enroll")
        await self.start(driver_core, enrollment_samples=2)
        async with connect(self.url()) as ws:
            await self.recv(ws)
            random = rng(2)
            base = photo("person_a_1", 480)
            for _ in range(30):
                await ws.send(json.dumps({"type": "frame", "image": data_url(live_variant(base, random))}))
                if (await self.recv(ws))["flow"]["finished"]:
                    break
                await asyncio.sleep(0.05)
        template = codes.decode_code(driver_core.callbacks[0]["face_code"])

        for name, crop, expected in (("person_a_2", True, "auth_success"), ("person_b_1", False, "auth_failed")):
            core = FakeCore("auth", template)
            self.server.core = core
            image = face_crop(name, self.analyzer, 480) if crop else photo(name, 480)
            random = rng(3)
            async with connect(self.url()) as ws:
                ready = await self.recv(ws)
                self.assertEqual(ready["mode"], "auth")
                final = None
                for _ in range(40):
                    await ws.send(json.dumps({"type": "frame", "image": data_url(live_variant(image, random))}))
                    message = await self.recv(ws)
                    if message["flow"]["finished"]:
                        final = message
                        break
                    await asyncio.sleep(0.05)
                self.assertIsNotNone(final, name)
                self.assertEqual(final["flow"]["status"], expected, name)
            self.assertEqual(len(core.callbacks), 1)
            self.assertEqual(core.callbacks[0]["success"], expected == "auth_success")
            if expected == "auth_failed":
                self.assertEqual(core.callbacks[0]["error_message"], "Facial comparison failed")
                self.assertEqual(core.callbacks[0]["face_code"], "")

    async def test_bad_tokens_close_with_4403(self):
        await self.start(FakeCore())
        for token in ("short", "x" * 200, "has space in it 0123456789", "unknown0123456789abcdef"):
            async with connect(self.url(token=token.replace(" ", "_") if token != "short" else token)) as ws:
                with self.assertRaises(ConnectionClosed) as caught:
                    await asyncio.wait_for(ws.recv(), 5)
                self.assertEqual(caught.exception.rcvd.code, 4403, token)

    async def test_monitor_mode_is_answered_with_a_clear_fatal_error(self):
        await self.start(FakeCore())
        async with connect(self.url(mode="monitor")) as ws:
            message = await self.recv(ws)
            self.assertEqual(message["type"], "error")
            self.assertTrue(message["fatal"])
            self.assertIn("not supported", message["message"])

    async def test_frames_too_fast_are_throttled_and_bad_frames_are_non_fatal_errors(self):
        await self.start(FakeCore(), frame_min_interval=5.0)
        async with connect(self.url()) as ws:
            await self.recv(ws)
            image = data_url(photo("person_a_1", 480))
            await ws.send(json.dumps({"type": "frame", "image": image}))
            first = await self.recv(ws)
            self.assertEqual(first["type"], "frame_result")
            await ws.send(json.dumps({"type": "frame", "image": image}))
            self.assertEqual(await self.recv(ws), {"type": "throttled"})
        await self.start(FakeCore())
        async with connect(self.url()) as ws:
            await self.recv(ws)
            for bad in ("not json", json.dumps({"type": "frame", "image": "data:image/png;base64,AAAA"}),
                        json.dumps({"type": "frame", "image": "data:image/jpeg;base64,@@@"}),
                        json.dumps({"type": "other"})):
                await ws.send(bad)
                message = await self.recv(ws)
                self.assertEqual(message["type"], "error")
                self.assertFalse(message["fatal"])
            await ws.send(json.dumps({"type": "frame", "image": data_url(photo("person_a_1", 480))}))
            self.assertEqual((await self.recv(ws))["type"], "frame_result")   # still works

    async def test_many_bad_frames_end_the_connection(self):
        await self.start(FakeCore())
        async with connect(self.url()) as ws:
            await self.recv(ws)
            last = None
            for _ in range(10):
                await ws.send("garbage")
                last = await self.recv(ws)
                if last.get("fatal"):
                    break
            self.assertTrue(last["fatal"])

    async def test_binary_messages_count_as_bad_frames(self):
        await self.start(FakeCore())
        async with connect(self.url()) as ws:
            await self.recv(ws)
            last = None
            for _ in range(10):
                await ws.send(b"\x00\x01")
                last = await self.recv(ws)
                if last.get("fatal"):
                    break
            self.assertTrue(last["fatal"])

    async def test_a_second_connection_with_the_same_token_replaces_the_first(self):
        await self.start(FakeCore())
        async with connect(self.url()) as first:
            await self.recv(first)
            async with connect(self.url()) as second:
                self.assertEqual((await self.recv(second))["type"], "ready")
                with self.assertRaises(ConnectionClosed) as caught:
                    await asyncio.wait_for(first.recv(), 5)
                self.assertEqual(caught.exception.rcvd.code, 4403)

    async def test_core_not_ready_and_core_down_are_fatal_errors(self):
        core = FakeCore()
        core.ready = False
        await self.start(core)
        async with connect(self.url()) as ws:
            message = await self.recv(ws)
            self.assertTrue(message["fatal"])

    async def test_result_not_delivered_to_core_is_fatal_for_the_page(self):
        core = FakeCore("enroll")
        core.fail_callback = True
        await self.start(core, enrollment_samples=1)
        base = photo("person_a_1", 480)
        random = rng(4)
        async with connect(self.url()) as ws:
            await self.recv(ws)
            seen = []
            for _ in range(20):
                await ws.send(json.dumps({"type": "frame", "image": data_url(live_variant(base, random))}))
                message = await self.recv(ws)
                seen.append(message)
                if message["type"] == "error":
                    break
            self.assertEqual(seen[-1]["type"], "error")
            self.assertTrue(seen[-1]["fatal"])
            self.assertFalse(any(m["type"] == "frame_result" and m["flow"]["finished"] for m in seen))

    async def test_connection_limits(self):
        await self.start(FakeCore(), max_connections_per_ip=1)
        async with connect(self.url()) as first:
            await self.recv(first)
            core = self.server.core
            async with connect(self.url(token="b" * 32)) as second:
                message = await self.recv(second)
                self.assertTrue(message["fatal"])
            # an address the client claims in front of the gateway's does not make it someone else
            spoofed = {"X-Forwarded-For": "203.0.113.9, 127.0.0.1"}
            async with connect(self.url(token="c" * 32), additional_headers=spoofed) as third:
                message = await self.recv(third)
                self.assertTrue(message["fatal"])

    async def test_health_and_other_paths(self):
        core = FakeCore()
        await self.start(core)
        loop = asyncio.get_running_loop()

        def get(path):
            try:
                with urllib.request.urlopen("http://127.0.0.1:%d%s" % (self.port, path), timeout=5) as reply:
                    self.assertEqual(reply.headers.get_all("Content-Type"), ["application/json"])
                    return reply.status, reply.read().decode()
            except HTTPError as error:
                return error.code, error.read().decode()
        status, body = await loop.run_in_executor(None, get, "/facelive/health")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["status"], "ok")
        core.ready = False
        status, _ = await loop.run_in_executor(None, get, "/facelive/health")
        self.assertEqual(status, 503)
        status, _ = await loop.run_in_executor(None, get, "/other")
        self.assertEqual(status, 404)
        status, _ = await loop.run_in_executor(None, get, "/ws/facelive/health")
        self.assertEqual(status, 404)
        # Cross-origin WebSocket handshake must be rejected with 403
        req = urllib.request.Request(
            "http://127.0.0.1:%d/ws/facelive/capture/?token=%s" % (self.port, TOKEN),
            headers={"Origin": "https://evil.example"},
        )
        with self.assertRaises(HTTPError) as caught:
            await loop.run_in_executor(None, lambda: urllib.request.urlopen(req, timeout=5))
        self.assertEqual(caught.exception.code, 403)

        def handshake_status(origin, host):
            req = urllib.request.Request(
                "http://127.0.0.1:%d/ws/facelive/capture/?token=%s" % (self.port, TOKEN),
                headers={"Origin": origin, "Host": host},
            )
            try:
                return urllib.request.urlopen(req, timeout=5).status
            except HTTPError as error:
                return error.code

        # behind the gateway: the Origin keeps a non-default port, the forwarded Host (nginx $host) does not
        status = await loop.run_in_executor(None, handshake_status, "https://xpam.example:8443", "xpam.example")
        self.assertNotEqual(status, 403)
        status = await loop.run_in_executor(None, handshake_status, "https://XPAM.example", "xpam.example:443")
        self.assertNotEqual(status, 403)
        status = await loop.run_in_executor(None, handshake_status, "https://[::1]:8443", "[::1]")
        self.assertNotEqual(status, 403)
        for origin, host in (("https://xpam.example.evil", "xpam.example"), ("null", "xpam.example"),
                             ("https://[::1", "xpam.example")):
            status = await loop.run_in_executor(None, handshake_status, origin, host)
            self.assertEqual(status, 403, (origin, host))


class FrameDecodingTest(unittest.TestCase):
    def test_jpeg_size_and_limits(self):
        data = jpeg(photo("person_a_1", 480))
        self.assertEqual(jpeg_size(data)[0], 480)
        self.assertIsNone(jpeg_size(b"not a jpeg"))
        with self.assertRaises(FrameError):
            decode_frame(json.dumps({"type": "frame", "image": "data:image/jpeg;base64," + base64.b64encode(data).decode()}), 1000)
        frame, raw = decode_frame(json.dumps({"type": "frame", "image": data_url(photo("person_a_1", 1200))}), 4 * 1024 * 1024)
        self.assertEqual(max(frame.shape[:2]), 640)    # larger pictures are made smaller (by the longer side)
        self.assertTrue(raw.startswith(b"\xff\xd8"))

    def test_tall_or_long_frames_are_refused_and_the_longer_side_is_limited(self):
        import numpy as np
        for shape in ((4000, 256, 3), (256, 4000, 3), (1000, 200, 3)):
            message = json.dumps({"type": "frame", "image": data_url(np.zeros(shape, np.uint8))})
            with self.assertRaises(FrameError):
                decode_frame(message, 4 * 1024 * 1024)
        frame, _ = decode_frame(json.dumps({"type": "frame", "image": data_url(np.full((1200, 800, 3), 90, np.uint8))}),
                                4 * 1024 * 1024)
        self.assertEqual(max(frame.shape[:2]), 640)
        self.assertEqual(frame.shape[1], 427)

    def test_a_jpeg_header_with_a_huge_size_is_refused_before_decoding(self):
        data = bytearray(jpeg(photo("person_a_1", 480)))
        index = data.find(b"\xff\xc0")
        data[index + 5:index + 9] = (60000).to_bytes(2, "big") + (60000).to_bytes(2, "big")
        message = json.dumps({"type": "frame", "image": "data:image/jpeg;base64," + base64.b64encode(bytes(data)).decode()})
        with self.assertRaises(FrameError):
            decode_frame(message, 4 * 1024 * 1024)
        # A fake small SOF0 inserted before a huge SOF0 must also be refused
        fake_sof = b"\xff\xc0\x00\x0b\x08\x00\x40\x00\x40\x01\x01\x11\x00"
        smuggled = bytes(data[:2]) + fake_sof + bytes(data[2:])
        self.assertIsNone(jpeg_size(smuggled))


if __name__ == "__main__":
    unittest.main()


class ClientAddressTest(unittest.TestCase):
    """The address the per-address limits count: never one the client chose."""

    class _Socket:
        def __init__(self, forwarded):
            self.request = type("R", (), {"headers": {"X-Forwarded-For": forwarded} if forwarded else {}})()
            self.remote_address = ("172.17.0.9", 50000)

    def address(self, forwarded, use_lb=None):
        import os
        old = os.environ.pop("USE_LB", None)
        try:
            if use_lb is not None:
                os.environ["USE_LB"] = use_lb
            return FaceServer.client_ip(self._Socket(forwarded))
        finally:
            os.environ.pop("USE_LB", None)
            if old is not None:
                os.environ["USE_LB"] = old

    def test_the_gateway_is_the_first_hop(self):
        self.assertEqual(self.address("203.0.113.7"), "203.0.113.7")
        self.assertEqual(self.address("6.6.6.6, 203.0.113.7"), "203.0.113.7")  # (the claimed one is ignored)

    def test_behind_a_trusted_load_balancer_the_browser_is_the_one_it_recorded(self):
        self.assertEqual(self.address("6.6.6.6, 203.0.113.7, 10.0.0.2", use_lb="0"), "203.0.113.7")
        self.assertEqual(self.address("203.0.113.7, 10.0.0.2", use_lb="0"), "203.0.113.7")
        self.assertEqual(self.address("10.0.0.2", use_lb="0"), "10.0.0.2")

    def test_without_the_header_the_socket_address_counts(self):
        self.assertEqual(self.address(""), "172.17.0.9")
