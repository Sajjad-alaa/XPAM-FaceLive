"""Talking to XPAM core as a service account.

Same method as the other XPAM components (Kael, KoKo...): the first time, register with the BOOTSTRAP_TOKEN
(POST /api/v1/terminal/terminal-registrations/, "Authorization: BootstrapToken <token>") and keep the access key
that core returns in a file; after that every request is signed with that key (HTTP Signature, hmac-sha256 over
"(request-target)" and "date", as in the other components) and sent with "X-JMS-ORG: ROOT".
"""
import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import ssl
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from email.utils import formatdate

log = logging.getLogger("facelive.core")

REGISTER_PATH = "/api/v1/terminal/terminal-registrations/"
PROFILE_PATH = "/api/v1/users/profile/"
CONTEXT_PATH = "/api/v1/authentication/face/engine/context/"
COMPARE_PATH = "/api/v1/authentication/face/engine/compare/"
CALLBACK_PATH = "/api/v1/authentication/face/callback/"


class CoreError(Exception):
    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


def sign_headers(method, target, key_id, secret, now=None):
    """Headers that sign one request. `target` is the path with its query string."""
    date = formatdate(now if now is not None else time.time(), usegmt=True)
    text = "(request-target): %s %s\ndate: %s" % (method.lower(), target, date)
    digest = hmac.new(secret.encode(), text.encode(), hashlib.sha256).digest()
    signature = base64.b64encode(digest).decode()
    return {
        "Date": date,
        "X-JMS-ORG": "ROOT",
        "Authorization": 'Signature keyId="%s",algorithm="hmac-sha256",headers="(request-target) date",signature="%s"'
                         % (key_id, signature),
    }


def load_access_key(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            key_id, _, secret = handle.read().strip().partition(":")
    except FileNotFoundError:
        return None
    return (key_id, secret) if key_id and secret else None


def save_access_key(path, key_id, secret):
    folder = os.path.dirname(path)
    os.makedirs(folder, mode=0o700, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".access_key-", dir=folder)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            out.write("%s:%s" % (key_id, secret))
            out.flush()
            os.fsync(out.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


class CoreClient:
    def __init__(self, settings):
        self.settings = settings
        self._key = None
        self.ready = False
        self._lock = None
        self._context = None
        if settings.core_host.startswith("https") and not settings.core_tls_verify:
            self._context = ssl.create_default_context()
            self._context.check_hostname = False
            self._context.verify_mode = ssl.CERT_NONE

    # ---- plain HTTP (blocking; always called through a thread)

    def _http(self, method, path, body=None, headers=None):
        url = self.settings.core_host + path
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(url, data=data, method=method)
        request.add_header("Accept", "application/json")
        if data is not None:
            request.add_header("Content-Type", "application/json")
        for name, value in (headers or {}).items():
            request.add_header(name, value)
        try:
            with urllib.request.urlopen(request, timeout=self.settings.http_timeout, context=self._context) as reply:
                payload = reply.read(1024 * 1024)
                return reply.status, (json.loads(payload) if payload else None)
        except urllib.error.HTTPError as error:
            payload = error.read(65536)
            try:
                parsed = json.loads(payload) if payload else None
            except ValueError:
                parsed = None
            return error.code, parsed
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise CoreError("core is not reachable: %s" % type(error).__name__) from None

    def _signed(self, method, path, body=None):
        key_id, secret = self._key
        return self._http(method, path, body, sign_headers(method, path, key_id, secret))

    # ---- registration

    def _validate_key(self):
        status, _ = self._signed("GET", PROFILE_PATH)
        if status == 200:
            return True
        if status in (401, 403):
            return False
        raise CoreError("core answered %s while checking the access key" % status, status)

    def _register(self):
        token = self.settings.bootstrap_token
        if not token:
            raise CoreError("no access key and no BOOTSTRAP_TOKEN: cannot register with core")
        name = self.settings.name
        for attempt in range(2):
            body = {"name": name, "comment": "facelive", "type": "facelive"}
            status, reply = self._http("POST", REGISTER_PATH, body, {"Authorization": "BootstrapToken " + token})
            if status in (200, 201) and reply:
                key = (reply.get("service_account") or {}).get("access_key") or {}
                if key.get("id") and key.get("secret"):
                    return key["id"], key["secret"]
                raise CoreError("core returned an empty access key")
            if status == 400 and attempt == 0:
                name = "%s-%s" % (name[:100], secrets.token_hex(3))  # (the name is already used)
                continue
            raise CoreError("core refused the registration (HTTP %s)" % status, status)

    def _connect_blocking(self):
        key = load_access_key(self.settings.access_key_file)
        if key:
            self._key = key
            if self._validate_key():
                return
            log.warning("the stored access key is not accepted by core; registering again")
        self._key = self._register()
        save_access_key(self.settings.access_key_file, *self._key)
        if not self._validate_key():
            raise CoreError("core does not accept the new access key")

    async def connect(self):
        """Register or load the access key. Safe to call again after a 401."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            await asyncio.to_thread(self._connect_blocking)
            self.ready = True
            log.info("connected to core as a service account")

    async def keep_connected(self):
        """Run in the background: retry until core accepts us (core may start later than this service)."""
        delay = 3.0
        while not self.ready:
            try:
                await self.connect()
            except CoreError as error:
                log.warning("waiting for core: %s", error)
            except Exception as error:  # a bug must not stop the retries
                log.error("core connection failed: %s", type(error).__name__)
            if not self.ready:
                await asyncio.sleep(delay)
                delay = min(delay * 1.7, 60.0)

    # ---- calls

    async def _call(self, method, path, body=None):
        if not self.ready:
            raise CoreError("core connection is not ready")
        status, reply = await asyncio.to_thread(self._signed, method, path, body)
        if status in (401, 403) and self.settings.bootstrap_token:
            # the key may have been removed in core: register again once
            self.ready = False
            await self.connect()
            status, reply = await asyncio.to_thread(self._signed, method, path, body)
        return status, reply

    async def context(self, token):
        """What the capture token is for: {"mode", "target_name", "threshold"} or None (unknown/finished)."""
        path = CONTEXT_PATH + "?" + urllib.parse.urlencode({"token": token})
        status, reply = await self._call("GET", path)
        if status == 200 and isinstance(reply, dict) and reply.get("valid") and reply.get("mode") in ("enroll", "auth"):
            return reply
        if status in (400, 404):
            return None
        raise CoreError("core answered %s for the capture token" % status, status)

    async def compare(self, token, face_code):
        """(matched, similarity) of a face code against the user of this token. Core answers yes or no only (a score
        would help someone at the camera tune a photo), so the similarity is 1.0 or 0.0."""
        status, reply = await self._call("POST", COMPARE_PATH, {"token": token, "face_code": face_code})
        if status == 200 and isinstance(reply, dict):
            matched = bool(reply.get("matched"))
            return matched, 1.0 if matched else 0.0
        raise CoreError("core answered %s for the comparison" % status, status)

    async def callback(self, token, success, face_code="", error_message=""):
        body = {"token": token, "success": bool(success)}
        if success:
            body["face_code"] = face_code
        else:
            body["error_message"] = error_message
        status, _ = await self._call("POST", CALLBACK_PATH, body)
        if status not in (200, 201):
            raise CoreError("core answered %s for the result" % status, status)
