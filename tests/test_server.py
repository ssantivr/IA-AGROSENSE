import http.client
import json
import threading
import unittest

from agrosense_ia.server import MAX_BODY_BYTES, create_server

READING = {"id_crop": 7, "soil_moisture": 31.5, "air_temperature": 24, "ph": 6.4}


class ServerTests(unittest.TestCase):
    """The service over real HTTP, on a free port of this machine."""

    @classmethod
    def setUpClass(cls):
        cls.server = create_server()
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def call(self, method, path, body=None, headers=None, raw=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_address[1], timeout=5)
        try:
            payload = raw if raw is not None else (None if body is None else json.dumps(body).encode("utf-8"))
            sent = {"Content-Type": "application/json"} if payload is not None else {}
            sent.update(headers or {})
            connection.request(method, path, body=payload, headers=sent)
            response = connection.getresponse()
            raw = response.read()
            return response, json.loads(raw.decode("utf-8")) if raw else None
        finally:
            connection.close()

    def test_health(self):
        response, body = self.call("GET", "/health")

        self.assertEqual(response.status, 200)
        self.assertEqual(body, {"status": "UP"})
        self.assertEqual(response.getheader("Content-Type"), "application/json; charset=utf-8")

    def test_a_reading_gets_the_fields_the_backend_stores(self):
        response, body = self.call("POST", "/predict/irrigation", READING)

        self.assertEqual(response.status, 200)
        self.assertEqual(body["id_crop"], 7)
        self.assertIs(body["irrigate"], True)
        self.assertGreater(body["duration_minutes"], 0)
        self.assertIn("31.5 %", body["recommendation"])
        self.assertTrue(0 <= body["confidence"] <= 1)
        self.assertEqual(body["model_used"], "agrosense-riego-reglas-1.0")
        self.assertIn("humedad_baja", body["factors"])
        self.assertEqual(response.getheader("Cache-Control"), "no-store")

    def test_the_crop_s_ranges_are_optional_and_used_when_sent(self):
        wide = dict(READING, soil_moisture=65)
        _, default = self.call("POST", "/predict/irrigation", wide)
        _, rice = self.call("POST", "/predict/irrigation", dict(wide, humidity_min=70, humidity_max=95))
        without_crop = {key: value for key, value in READING.items() if key != "id_crop"}

        self.assertIs(default["irrigate"], False)
        self.assertIs(rice["irrigate"], True)
        self.assertIsNone(self.call("POST", "/predict/irrigation", without_crop)[1]["id_crop"])

    def test_only_the_soil_moisture_is_required(self):
        response, body = self.call("POST", "/predict/irrigation", {"soil_moisture": 31.5})

        self.assertEqual(response.status, 200)
        self.assertIs(body["irrigate"], True)
        self.assertEqual(body["factors"], ["humedad_baja"])

    def test_bad_data_is_refused_with_a_message_that_names_the_field(self):
        cases = [
            ({key: value for key, value in READING.items() if key != "soil_moisture"}, "'soil_moisture'"),
            (dict(READING, ph="ácido"), "'ph'"),
            (dict(READING, soil_moisture="mucha"), "'soil_moisture'"),
            (dict(READING, soil_moisture=True), "'soil_moisture'"),
            (dict(READING, soil_moisture=140), "entre 0 y 100"),
            (dict(READING, air_temperature=-80), "entre -50 y 80"),
            (dict(READING, ph=15), "entre 0 y 14"),
            (dict(READING, id_crop=0), "'id_crop'"),
            (dict(READING, id_crop="7"), "'id_crop'"),
            (dict(READING, humidity_min=50), "deben enviarse juntos"),
            (dict(READING, humidity_min=80, humidity_max=40), "menor que"),
            ([1, 2, 3], "objeto JSON"),
        ]
        for body, expected in cases:
            with self.subTest(body=body):
                response, answer = self.call("POST", "/predict/irrigation", body)
                self.assertEqual(response.status, 400)
                self.assertIn(expected, answer["message"])

    def test_what_is_not_json_is_refused(self):
        for raw in (b"{not json", b'{"soil_moisture": NaN, "air_temperature": 20, "ph": 6}', b"\xff\xfe"):
            with self.subTest(raw=raw):
                response, answer = self.call("POST", "/predict/irrigation", raw=raw)
                self.assertEqual(response.status, 400)
                self.assertIn("JSON", answer["message"])
        response, _ = self.call("POST", "/predict/irrigation", raw=b"soil_moisture=30",
                                headers={"Content-Type": "application/x-www-form-urlencoded"})
        self.assertEqual(response.status, 415)

    def test_a_body_that_is_too_large_is_refused_without_being_read(self):
        response, answer = self.call("POST", "/predict/irrigation", raw=b" " * (MAX_BODY_BYTES + 1))

        self.assertEqual(response.status, 413)
        self.assertIn("demasiado grande", answer["message"])

    def test_the_bare_address_says_what_the_service_is(self):
        response, body = self.call("GET", "/")

        self.assertEqual(response.status, 200)
        self.assertEqual(body["service"], "AgroSense IA")
        self.assertEqual(body["health"], "/health")

    def test_head_answers_like_get_without_a_body(self):
        for path in ("/", "/health"):
            with self.subTest(path=path):
                response, body = self.call("HEAD", path)
                self.assertEqual(response.status, 200)
                self.assertIsNone(body)
                self.assertGreater(int(response.getheader("Content-Length")), 0)

    def test_a_connection_that_is_reused_keeps_answering_with_bodies_after_a_head(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_address[1], timeout=5)
        try:
            connection.request("HEAD", "/health")
            connection.getresponse().read()
            connection.request("GET", "/health")
            self.assertEqual(json.loads(connection.getresponse().read()), {"status": "UP"})
        finally:
            connection.close()

    def test_without_a_configured_token_nobody_is_asked_for_one(self):
        self.assertEqual(self.call("POST", "/predict/irrigation", READING)[0].status, 200)

    def test_unknown_paths_and_wrong_methods(self):
        self.assertEqual(self.call("GET", "/nothing")[0].status, 404)
        self.assertEqual(self.call("POST", "/", {})[0].status, 405)
        self.assertEqual(self.call("POST", "/predict/pests", READING)[0].status, 404)
        wrong_method, _ = self.call("GET", "/predict/irrigation")
        self.assertEqual(wrong_method.status, 405)
        self.assertEqual(wrong_method.getheader("Allow"), "POST")
        self.assertEqual(self.call("POST", "/health", {})[0].status, 405)

    def test_requests_at_the_same_time_are_all_answered(self):
        results = []

        def ask(moisture):
            results.append(self.call("POST", "/predict/irrigation", dict(READING, soil_moisture=moisture))[1])

        threads = [threading.Thread(target=ask, args=(moisture,)) for moisture in range(0, 100, 5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        self.assertEqual(len(results), 20)
        self.assertEqual(sum(1 for result in results if result["irrigate"]), 8)


class ServiceTokenTests(ServerTests):
    """With a token configured: the same service, and predictions only for callers that send it."""

    TOKEN = "un-secreto-de-prueba"

    @classmethod
    def setUpClass(cls):
        cls.server = create_server(token=f"  {cls.TOKEN}  ")
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    def call(self, method, path, body=None, headers=None, raw=None):
        # Everything the open service is tested for must also hold for a caller that has the token.
        sent = {"Authorization": f"Bearer {self.TOKEN}"}
        sent.update(headers or {})
        return super().call(method, path, body, sent, raw)

    def test_without_a_configured_token_nobody_is_asked_for_one(self):
        self.skipTest("This server has a token")

    def test_predictions_are_refused_without_the_token(self):
        for header in ("", "Bearer otro-secreto", f"Basic {self.TOKEN}", self.TOKEN, "Bearer", f"Bearer {self.TOKEN}x"):
            with self.subTest(header=header):
                response, body = self.call("POST", "/predict/irrigation", READING, headers={"Authorization": header})
                self.assertEqual(response.status, 401)
                self.assertEqual(response.getheader("WWW-Authenticate"), "Bearer")
                self.assertIn("token", body["message"])
                self.assertNotIn(self.TOKEN, body["message"])

    def test_the_scheme_is_not_case_sensitive(self):
        response, _ = self.call("POST", "/predict/irrigation", READING,
                                headers={"Authorization": f"bearer {self.TOKEN}"})
        self.assertEqual(response.status, 200)

    def test_health_and_the_bare_address_stay_open(self):
        for path in ("/health", "/"):
            with self.subTest(path=path):
                self.assertEqual(self.call("GET", path, headers={"Authorization": ""})[0].status, 200)


if __name__ == "__main__":
    unittest.main()
