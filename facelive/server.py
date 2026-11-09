"""The WebSocket server the Workspace page talks to (/ws/facelive/capture/?token=...&mode=...).

Messages (JSON), the same as the page's useRemoteFaceCapture.ts:
  server -> {"type":"ready","mode":"enroll|auth","debug":bool,"frame_interval_ms":int}
  client -> {"type":"frame","image":"data:image/jpeg;base64,..."}
  server -> {"type":"frame_result","frame":{...},"faces":[...],"flow":{...}} | {"type":"throttled"}
            | {"type":"error","fatal":bool,"message":str}
A token that core does not know (or is finished) closes the connection with code 4403.
"""
import asyncio
import base64
import binascii
import json
import logging
import re
import struct
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from urllib.parse import parse_qs, urlsplit

import cv2
import numpy as np
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from .core import CoreClient, CoreError
from .flow import AUTH, ENROLL, CompareRejected, CompareUnavailable, FlowSession

log = logging.getLogger("facelive.server")

TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9_-]{16,128}$")  # the same rule as the page (isFaceCaptureToken)
WS_PATHS = ("/ws/facelive/capture/", "/ws/facelive/capture", "/facelive/capture/", "/facelive/capture")
HEALTH_PATHS = ("/facelive/health", "/facelive/health/")
JPEG_PREFIX = "data:image/jpeg;base64,"
MAX_SIDE = 640            # frames whose longer side is more than this are made smaller before analysis
MAX_INPUT_SIDE = 1920     # refuse pictures with a longer side than this, before decoding them
MAX_ASPECT = 3.0          # refuse pictures longer than 3:1 (either way)
MAX_PIXELS = 4096 * 4096  # refuse pictures that would be huge in memory
IDLE_SECONDS = 25         # no frame for this long: close
MAX_BAD_FRAMES = 8        # bad frames in a row before the connection is closed
LIFETIME_MARGIN = 30      # a connection lives at most the flow deadline plus this many seconds
EXPIRED_MESSAGE = "This face check has expired. Please try again."
CLOSE_INVALID_TOKEN = 4403


class FrameError(Exception):
    pass


def jpeg_size(data):
    """(width, height) from the JPEG header without decoding it, or None."""
    if len(data) < 4 or data[0:2] != b"\xff\xd8":
        return None
    index = 2
    found = None
    while index + 4 <= len(data):
        if data[index] != 0xFF:
            return None
        while index < len(data) and data[index] == 0xFF:
            index += 1
        if index >= len(data):
            return None
        marker = data[index]
        index += 1
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            continue
        if marker in (0xD9, 0xDA):  # EOI or SOS (entropy-coded data starts here)
            break
        if index + 2 > len(data):
            return None
        (length,) = struct.unpack(">H", data[index:index + 2])
        if length < 2 or index + length > len(data):
            return None
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            if length < 7 or index + 7 > len(data):
                return None
            height, width = struct.unpack(">HH", data[index + 3:index + 7])
            if found is not None and found != (width, height):
                return None  # refuse multi-frame JPEGs with conflicting SOF dimensions
            found = (width, height)
        index += length
    return found


def decode_frame(message, max_bytes):
    """Client message (text) -> (BGR array, jpeg bytes). Raises FrameError with a message for the page."""
    try:
        data = json.loads(message)
    except ValueError:
        raise FrameError("The message is not valid JSON") from None
    if not isinstance(data, dict) or data.get("type") != "frame":
        raise FrameError("Unknown message type")
    image = data.get("image")
    if not isinstance(image, str) or not image.startswith(JPEG_PREFIX):
        raise FrameError("The frame must be a JPEG data URL")
    if len(image) > max_bytes * 4 // 3 + 64:
        raise FrameError("The frame is too large")
    try:
        raw = base64.b64decode(image[len(JPEG_PREFIX):], validate=True)
    except (binascii.Error, ValueError):
        raise FrameError("The frame is not valid base64") from None
    if len(raw) > max_bytes:
        raise FrameError("The frame is too large")
    size = jpeg_size(raw)
    if not size or size[0] < 32 or size[1] < 32 or size[0] * size[1] > MAX_PIXELS:
        raise FrameError("The frame size is not accepted")
    if max(size) > MAX_INPUT_SIDE or max(size) > MAX_ASPECT * min(size):
        raise FrameError("The frame size is not accepted")
    frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise FrameError("The frame could not be decoded")
    height, width = frame.shape[:2]
    longer = max(width, height)
    if longer > MAX_SIDE:
        scale = MAX_SIDE / float(longer)
        frame = cv2.resize(frame, (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                           interpolation=cv2.INTER_AREA)
    return frame, raw


class Limits:
    """Connection limits: in total, per client address and new connections per minute."""

    def __init__(self, settings):
        self.settings = settings
        self.active = 0
        self.per_ip = defaultdict(int)
        self.recent = defaultdict(deque)

    def admit(self, ip, now=None):
        now = time.monotonic() if now is None else now
        if len(self.recent) > 2000:  # forget addresses that have been quiet for a minute
            for old in [key for key, seen in self.recent.items() if not seen or now - seen[-1] > 60]:
                if not self.per_ip.get(old):
                    del self.recent[old]
        window = self.recent[ip]
        while window and now - window[0] > 60:
            window.popleft()
        if (self.active >= self.settings.max_connections
                or self.per_ip[ip] >= self.settings.max_connections_per_ip
                or len(window) >= self.settings.connects_per_minute):
            return False
        window.append(now)
        self.active += 1
        self.per_ip[ip] += 1
        return True

    def release(self, ip):
        self.active = max(0, self.active - 1)
        self.per_ip[ip] = max(0, self.per_ip[ip] - 1)
        if not self.per_ip[ip]:
            self.per_ip.pop(ip, None)
            if not self.recent.get(ip):
                self.recent.pop(ip, None)


class FaceServer:
    def __init__(self, settings, analyzer, core=None):
        self.settings = settings
        self.analyzer = analyzer
        self.core = core or CoreClient(settings)
        self.limits = Limits(settings)
        self.executor = ThreadPoolExecutor(max_workers=settings.workers, thread_name_prefix="facelive")
        self.by_token = {}
        self.server = None

    # ---- HTTP side: health only

    async def process_request(self, connection, request):
        path = urlsplit(request.path).path
        if path in HEALTH_PATHS:
            ready = self.core.ready
            body = json.dumps({"status": "ok" if ready else "starting", "core": "connected" if ready else "waiting"})
            status = HTTPStatus.OK if ready else HTTPStatus.SERVICE_UNAVAILABLE
            response = connection.respond(status, body + "\n")
            del response.headers["Content-Type"]
            response.headers["Content-Type"] = "application/json"
            return response
        if path not in WS_PATHS:
            return connection.respond(HTTPStatus.NOT_FOUND, "Not found\n")
        # a page of another site may not open a capture: the browser's Origin must name the host it called. Host
        # names only — the gateway forwards Host without the port (nginx $host), the Origin keeps it.
        origin = (request.headers.get("Origin") or "").strip()
        host = (request.headers.get("Host") or "").strip()
        if origin and host:
            try:
                origin_name = (urlsplit(origin).hostname or "").lower()
                host_name = (urlsplit("//" + host).hostname or "").lower()
            except ValueError:  # a malformed address (e.g. an unclosed IPv6 bracket)
                origin_name = host_name = ""
            if not origin_name or origin_name != host_name:
                return connection.respond(HTTPStatus.FORBIDDEN, "Forbidden\n")
        return None

    # ---- WebSocket side

    @staticmethod
    def client_ip(websocket):
        forwarded = websocket.request.headers.get("X-Forwarded-For", "")
        if forwarded:
            # (the last address is the one the gateway saw; earlier ones are whatever the client claimed)
            return forwarded.split(",")[-1].strip()
        remote = websocket.remote_address
        return remote[0] if remote else "unknown"

    @staticmethod
    async def send(websocket, message):
        await websocket.send(json.dumps(message, separators=(",", ":")))

    async def fatal(self, websocket, message, code=1000):
        try:
            await self.send(websocket, {"type": "error", "fatal": True, "message": message})
            await websocket.close(code)
        except ConnectionClosed:
            pass

    async def handler(self, websocket):
        query = parse_qs(urlsplit(websocket.request.path).query)
        token = (query.get("token") or [""])[0].strip()
        page_mode = (query.get("mode") or ["capture"])[0].strip().lower()
        ip = self.client_ip(websocket)

        if page_mode == "monitor":
            # Online presence monitoring (pausing a live session when the person leaves) is not built in this engine
            await self.fatal(websocket, "Face monitoring is not supported by this server")
            return
        if not TOKEN_PATTERN.match(token):
            await websocket.close(CLOSE_INVALID_TOKEN)
            return
        if not self.limits.admit(ip):
            await self.fatal(websocket, "Too many face captures. Try again in a moment.", 1013)
            return
        try:
            await self.serve_capture(websocket, token)
        finally:
            self.limits.release(ip)

    async def serve_capture(self, websocket, token):
        if not self.core.ready:
            await self.fatal(websocket, "The face service is starting. Try again in a moment.", 1013)
            return
        try:
            context = await self.core.context(token)
        except CoreError as error:
            log.warning("capture token check failed: %s", error)
            await self.fatal(websocket, "The face service cannot reach the server", 1013)
            return
        if context is None:
            await websocket.close(CLOSE_INVALID_TOKEN)
            return

        # one open capture per token: a new connection replaces the old one
        previous = self.by_token.get(token)
        if previous is not None:
            try:
                await previous.close(CLOSE_INVALID_TOKEN)
            except Exception:
                pass
        self.by_token[token] = websocket

        mode = ENROLL if context["mode"] == "enroll" else AUTH
        threshold = float(context.get("threshold") or 0.4)

        async def compare(code):
            try:
                return await self.core.compare(token, code)
            except CoreError as error:
                log.warning("comparison failed: %s", error)
                if error.status in (400, 404):
                    raise CompareRejected() from None
                raise CompareUnavailable() from None

        flow = FlowSession(self.settings, mode, self.analyzer, compare,
                           target_name=str(context.get("target_name") or ""),
                           threshold=threshold, executor=self.executor,
                           yaw_turn=self.settings.yaw_turn, pitch_nod=self.settings.pitch_nod)
        log.info("capture opened (mode=%s)", mode)
        try:
            await self.send(websocket, {"type": "ready", "mode": mode, "debug": self.settings.debug,
                                        "frame_interval_ms": self.settings.frame_interval_ms})
            await self.frame_loop(websocket, token, flow)
        except ConnectionClosed:
            pass
        finally:
            if self.by_token.get(token) is websocket:
                del self.by_token[token]
            log.info("capture closed (status=%s)", flow.status)

    async def frame_loop(self, websocket, token, flow):
        last_frame = -1e9
        bad_frames = 0
        ends_at = time.monotonic() + flow._deadline() + LIFETIME_MARGIN
        while True:
            remaining = ends_at - time.monotonic()
            if remaining <= 0:
                await self.fatal(websocket, "Face capture timed out")
                return
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=min(IDLE_SECONDS, remaining))
            except asyncio.TimeoutError:
                if time.monotonic() >= ends_at:
                    await self.fatal(websocket, "Face capture timed out")
                else:
                    await self.fatal(websocket, "No camera frames were received")
                return
            if isinstance(message, bytes):
                bad_frames += 1
                if bad_frames >= MAX_BAD_FRAMES:
                    await self.fatal(websocket, "Binary frames are not accepted")
                    return
                await self.send(websocket, {"type": "error", "fatal": False, "message": "Binary frames are not accepted"})
                continue
            now = time.monotonic()
            if now - last_frame < self.settings.frame_min_interval:
                await self.send(websocket, {"type": "throttled"})
                continue
            last_frame = now
            try:
                frame, raw = await asyncio.get_running_loop().run_in_executor(
                    self.executor, decode_frame, message, self.settings.frame_max_bytes)
            except FrameError as error:
                bad_frames += 1
                if bad_frames >= MAX_BAD_FRAMES:
                    await self.fatal(websocket, str(error))
                    return
                await self.send(websocket, {"type": "error", "fatal": False, "message": str(error)})
                continue
            bad_frames = 0
            try:
                result = await flow.process_frame(frame, raw)
            except CompareRejected:
                await self.fatal(websocket, EXPIRED_MESSAGE)
                return
            except CompareUnavailable:
                await self.fatal(websocket, "The face service cannot reach the server", 1013)
                return
            except Exception:
                log.exception("frame processing failed")
                await self.fatal(websocket, "Face processing failed")
                return
            if flow.finished:
                if not await self.report(websocket, token, flow):
                    return
                await self.send(websocket, result)
                await asyncio.sleep(1.0)  # (the page closes the connection itself)
                await websocket.close(1000)
                return
            await self.send(websocket, result)

    async def report(self, websocket, token, flow):
        """Tell core the result before the page is told. False when core could not be told."""
        outcome = flow.outcome
        for attempt in range(3):
            try:
                await self.core.callback(token, outcome.success, outcome.face_code, outcome.error_message)
                return True
            except CoreError as error:
                log.warning("sending the result to core failed (%s)", error)
                if error.status in (400, 404):  # the token is finished or expired: retrying cannot help
                    await self.fatal(websocket, EXPIRED_MESSAGE)
                    return False
                await asyncio.sleep(0.5 * (attempt + 1))
        await self.fatal(websocket, "The face service cannot reach the server", 1013)
        return False

    # ---- run

    def listen(self, host=None, port=None):
        """The server as an async context manager (tests use port 0)."""
        return serve(
            self.handler, host or self.settings.host, self.settings.port if port is None else port,
            process_request=self.process_request,
            max_size=int(self.settings.frame_max_bytes * 4 / 3) + 4096,
            max_queue=2, ping_interval=20, ping_timeout=20, compression=None,
            server_header=None,
        )

    async def run(self):
        async with self.listen() as server:
            self.server = server
            log.info("listening on %s:%s", self.settings.host, self.settings.port)
            await server.serve_forever()
