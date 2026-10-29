import asyncio
import importlib.util
import json
import os
import stat
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
from httpsig import HeaderVerifier

from facelive import codes
from facelive.core import CoreClient, CoreError, load_access_key, save_access_key, sign_headers

from helpers import settings

CORE_CODEC = "/opt/xpam-core/apps/users/face_sface.py"


class CodesTest(unittest.TestCase):
    def test_roundtrip_is_unit_length_and_exact(self):
        rng = np.random.default_rng(1)
        vector = rng.normal(size=128)
        code = codes.encode_code(vector)
        self.assertTrue(code.startswith("xpamf1:"))
        decoded = codes.decode_code(code)
        self.assertAlmostEqual(float(np.linalg.norm(decoded)), 1.0, places=5)
        self.assertTrue(np.allclose(decoded, vector / np.linalg.norm(vector), atol=1e-6))

    def test_mean_feature_is_unit_length(self):
        rng = np.random.default_rng(2)
        mean = codes.mean_feature([rng.normal(size=128) for _ in range(3)])
        self.assertAlmostEqual(float(np.linalg.norm(mean)), 1.0, places=5)

    def test_bad_input_is_refused(self):
        with self.assertRaises(ValueError):
            codes.unit(np.zeros(128))
        with self.assertRaises(ValueError):
            codes.unit(np.ones(12))
        with self.assertRaises(ValueError):
            codes.decode_code("nope")

    @unittest.skipUnless(os.path.isfile(CORE_CODEC), "core source is not on this machine")
    def test_core_reads_the_codes_made_here(self):
        # load core's codec module without the rest of core
        import sys
        import types
        users = types.ModuleType("users")
        users.__path__ = []
        sys.modules.setdefault("users", users)
        codec_spec = importlib.util.spec_from_file_location("users.face_codec", os.path.join(os.path.dirname(CORE_CODEC), "face_codec.py"))
        import dataclasses
        original = dataclasses.dataclass
        dataclasses.dataclass = lambda *a, slots=None, **k: original(*a, **k)  # (python 3.9 here, core uses 3.11)
        try:
            codec = importlib.util.module_from_spec(codec_spec)
            sys.modules["users.face_codec"] = codec
            codec_spec.loader.exec_module(codec)
            spec = importlib.util.spec_from_file_location("users.face_sface", CORE_CODEC)
            core = importlib.util.module_from_spec(spec)
            sys.modules["users.face_sface"] = core
            spec.loader.exec_module(core)
        finally:
            dataclasses.dataclass = original
        rng = np.random.default_rng(3)
        a, b = rng.normal(size=128), rng.normal(size=128)
        code_a, code_b = codes.encode_code(a), codes.encode_code(b)
        self.assertTrue(core.is_native_code(code_a))
        self.assertAlmostEqual(core.similarity(code_a, code_b), codes.cosine(a, b), places=5)
        self.assertTrue(np.allclose(core.decode_code(core.encode_code(list(a))), codes.decode_code(code_a), atol=1e-6))
        self.assertTrue(core.compare_codes(code_a, code_a, cosine_threshold=0.4, distance_threshold=1.0955))
        self.assertFalse(core.compare_codes(code_a, code_b, cosine_threshold=0.4, distance_threshold=1.0955))


class SignatureTest(unittest.TestCase):
    def test_core_side_verifier_accepts_the_signature(self):
        # core checks with httpsig.HeaderVerifier(headers, secret, ["(request-target)", "date"], method, path)
        headers = sign_headers("POST", "/api/v1/authentication/face/callback/", "key-id-1", "secret-1")
        self.assertTrue(HeaderVerifier(headers, "secret-1", required_headers=["(request-target)", "date"],
                                       method="post", path="/api/v1/authentication/face/callback/").verify())
        self.assertFalse(HeaderVerifier(headers, "other", required_headers=["(request-target)", "date"],
                                        method="post", path="/api/v1/authentication/face/callback/").verify())
        self.assertFalse(HeaderVerifier(headers, "secret-1", required_headers=["(request-target)", "date"],
                                        method="get", path="/api/v1/authentication/face/callback/").verify())

    def test_query_string_is_part_of_the_target(self):
        path = "/api/v1/authentication/face/engine/context/?token=abcdefabcdefabcdef"
        headers = sign_headers("GET", path, "k", "s")
        self.assertTrue(HeaderVerifier(headers, "s", required_headers=["(request-target)", "date"],
                                       method="get", path=path).verify())
        self.assertEqual(headers["X-JMS-ORG"], "ROOT")


class FakeCore(BaseHTTPRequestHandler):
    """A tiny core: registration, signed requests, context, compare and callback."""
    state = None

    def log_message(self, *args):
        pass

    def _send(self, status, body=None):
        data = json.dumps(body).encode() if body is not None else b""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        length = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(length)) if length else None

    def _signed_ok(self):
        auth = self.headers.get("Authorization", "")
        state = self.state
        if not auth.startswith("Signature "):
            return False
        if 'keyId="%s"' % state["key_id"] not in auth:
            return False
        return HeaderVerifier(dict(self.headers.items()), state["secret"], required_headers=["(request-target)", "date"],
                              method=self.command.lower(), path=self.path).verify()

    def do_POST(self):
        state = self.state
        body = self._body()
        state["calls"].append((self.command, self.path))
        if self.path.startswith("/api/v1/terminal/terminal-registrations/"):
            if self.headers.get("Authorization") != "BootstrapToken boot-token":
                return self._send(403, {"detail": "no"})
            state["registrations"] += 1
            state["key_id"], state["secret"] = "ak-%d" % state["registrations"], "sk-%d" % state["registrations"]
            state["registered_body"] = body
            return self._send(201, {"service_account": {"access_key": {"id": state["key_id"], "secret": state["secret"]}}})
        if not self._signed_ok():
            return self._send(401, {"detail": "bad signature"})
        if self.path.startswith("/api/v1/authentication/face/engine/compare/"):
            state["compared"].append(body)
            return self._send(200, {"matched": True})
        if self.path.startswith("/api/v1/authentication/face/callback/"):
            state["callbacks"].append(body)
            return self._send(201, body)
        self._send(404, {})

    def do_GET(self):
        state = self.state
        state["calls"].append((self.command, self.path))
        if not self._signed_ok():
            return self._send(401, {"detail": "bad signature"})
        if self.path.startswith("/api/v1/users/profile/"):
            return self._send(200, {"id": "service-user"})
        if self.path.startswith("/api/v1/authentication/face/engine/context/"):
            if "token=known-token-0123456789" in self.path:
                return self._send(200, {"valid": True, "mode": "auth", "target_name": "Test User", "threshold": 0.4})
            return self._send(404, {"error": "unknown"})
        self._send(404, {})


class CoreClientTest(unittest.TestCase):
    def setUp(self):
        self.state = {"registrations": 0, "calls": [], "callbacks": [], "compared": [], "key_id": None, "secret": None}
        handler = type("Handler", (FakeCore,), {"state": self.state})
        self.httpd = HTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.folder = tempfile.mkdtemp()
        self.key_file = os.path.join(self.folder, "keys", ".access_key")
        self.settings = settings(core_host="http://127.0.0.1:%d" % self.httpd.server_address[1],
                                 bootstrap_token="boot-token", access_key_file=self.key_file,
                                 name="[Facelive]-test", http_timeout=5.0)

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def run_async(self, coroutine):
        return asyncio.run(coroutine)

    def test_registers_once_saves_the_key_and_signs_calls(self):
        async def scenario():
            client = CoreClient(self.settings)
            await client.connect()
            self.assertTrue(client.ready)
            context = await client.context("known-token-0123456789")
            self.assertEqual(context["mode"], "auth")
            self.assertIsNone(await client.context("unknown-token-0123456789"))
            self.assertEqual(await client.compare("known-token-0123456789", "xpamf1:abc"), (True, 1.0))
            await client.callback("known-token-0123456789", True, face_code="xpamf1:abc")
            await client.callback("known-token-0123456789", False, error_message="Facial comparison failed")
        self.run_async(scenario())
        self.assertEqual(self.state["registrations"], 1)
        self.assertEqual(self.state["registered_body"], {"name": "[Facelive]-test", "comment": "facelive", "type": "facelive"})
        self.assertEqual(self.state["callbacks"][0], {"token": "known-token-0123456789", "success": True, "face_code": "xpamf1:abc"})
        self.assertEqual(self.state["callbacks"][1], {"token": "known-token-0123456789", "success": False,
                                                      "error_message": "Facial comparison failed"})
        self.assertEqual(load_access_key(self.key_file), ("ak-1", "sk-1"))
        mode = stat.S_IMODE(os.stat(self.key_file).st_mode)
        self.assertEqual(mode, 0o600)

    def test_a_second_start_reuses_the_stored_key(self):
        async def start():
            client = CoreClient(self.settings)
            await client.connect()
        self.run_async(start())
        self.run_async(start())
        self.assertEqual(self.state["registrations"], 1)

    def test_a_key_that_core_no_longer_knows_is_replaced(self):
        save_access_key(self.key_file, "old-id", "old-secret")
        self.run_async(CoreClient(self.settings).connect())
        self.assertEqual(self.state["registrations"], 1)
        self.assertEqual(load_access_key(self.key_file), ("ak-1", "sk-1"))

    def test_registration_without_a_bootstrap_token_fails_clearly(self):
        config = settings(core_host=self.settings.core_host, bootstrap_token="", access_key_file=self.key_file)
        with self.assertRaises(CoreError):
            self.run_async(CoreClient(config).connect())

    def test_wrong_bootstrap_token_is_refused(self):
        config = settings(core_host=self.settings.core_host, bootstrap_token="wrong", access_key_file=self.key_file)
        with self.assertRaises(CoreError):
            self.run_async(CoreClient(config).connect())
        self.assertIsNone(load_access_key(self.key_file))

    def test_core_removing_the_key_triggers_a_new_registration(self):
        async def scenario():
            client = CoreClient(self.settings)
            await client.connect()
            self.state["key_id"], self.state["secret"] = "changed", "changed"   # core forgot our key
            context = await client.context("known-token-0123456789")
            self.assertEqual(context["mode"], "auth")
        self.run_async(scenario())
        self.assertEqual(self.state["registrations"], 2)


if __name__ == "__main__":
    unittest.main()
