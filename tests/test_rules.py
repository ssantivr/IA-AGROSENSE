"""The rules that use the moisture of the last hours, the rain and past irrigations. Plain unittest and no FastAPI."""
import unittest

from agrosense_ia.config import Settings
from agrosense_ia.model import predict_irrigation
from agrosense_ia.validation import BadRequest, answer_prediction

STEADY = [(5.0, 60.0), (3.0, 60.5), (1.0, 60.0), (0.0, 60.5)]
FALLING = [(5.0, 62.0), (3.0, 56.0), (1.0, 50.0), (0.0, 47.0)]
RISING = [(3.0, 20.0), (2.0, 24.0), (1.0, 28.0), (0.0, 32.0)]


class TrendTests(unittest.TestCase):
    def test_without_the_new_fields_nothing_changes(self):
        plain = predict_irrigation(soil_moisture=30)
        self.assertEqual(plain, predict_irrigation(soil_moisture=30, moisture_history=[], rain_probability=None))
        self.assertEqual(plain.factors, ["humedad_baja"])

    def test_a_falling_moisture_is_watered_before_it_reaches_the_minimum(self):
        prediction = predict_irrigation(soil_moisture=47, moisture_history=FALLING)
        self.assertTrue(prediction.irrigate)
        self.assertIn("tendencia_a_la_baja", prediction.factors)
        self.assertIn("puntos por hora", prediction.recommendation)

    def test_a_steady_moisture_is_left_alone(self):
        prediction = predict_irrigation(soil_moisture=60, moisture_history=STEADY)
        self.assertFalse(prediction.irrigate)
        self.assertNotIn("tendencia_a_la_baja", prediction.factors)

    def test_a_trend_needs_enough_points_and_hours(self):
        self.assertEqual(predict_irrigation(soil_moisture=47, moisture_history=FALLING[:2]).factors, ["humedad_adecuada"])
        minutes = [(0.3, 62.0), (0.2, 56.0), (0.0, 47.0)]
        self.assertEqual(predict_irrigation(soil_moisture=47, moisture_history=minutes).factors, ["humedad_adecuada"])

    def test_a_rising_moisture_is_mentioned(self):
        prediction = predict_irrigation(soil_moisture=32, moisture_history=RISING)
        self.assertIn("tendencia_al_alza", prediction.factors)
        self.assertIn("sube", prediction.recommendation)


class RainTests(unittest.TestCase):
    def test_likely_rain_avoids_an_irrigation_the_soil_can_wait_for(self):
        prediction = predict_irrigation(soil_moisture=35, rain_probability=80)
        self.assertFalse(prediction.irrigate)
        self.assertEqual(prediction.duration_minutes, 0)
        self.assertIn("lluvia_prevista", prediction.factors)

    def test_rain_in_millimetres_counts_too(self):
        self.assertFalse(predict_irrigation(soil_moisture=35, rain_mm=6).irrigate)

    def test_soil_too_dry_does_not_wait(self):
        prediction = predict_irrigation(soil_moisture=20, rain_probability=90)
        self.assertTrue(prediction.irrigate)
        self.assertIn("demasiado seco", prediction.recommendation)

    def test_unlikely_rain_changes_nothing(self):
        prediction = predict_irrigation(soil_moisture=35, rain_probability=30, rain_mm=1)
        self.assertTrue(prediction.irrigate)
        self.assertNotIn("lluvia_prevista", prediction.factors)


class AnomalyTests(unittest.TestCase):
    def test_a_stuck_sensor_lowers_the_confidence_below_what_an_automatic_order_needs(self):
        stuck = [(float(hours), 100.0) for hours in range(6)]
        prediction = predict_irrigation(soil_moisture=100, moisture_history=stuck)
        self.assertIn("sensor_atascado", prediction.factors)
        self.assertLess(prediction.confidence, 0.70)

    def test_a_jump_is_named(self):
        jump = [(3.0, 60.0), (2.0, 61.0), (1.0, 20.0), (0.0, 21.0)]
        self.assertIn("salto_brusco", predict_irrigation(soil_moisture=21, moisture_history=jump).factors)


class GainTests(unittest.TestCase):
    def test_past_irrigations_set_the_duration(self):
        prediction = predict_irrigation(soil_moisture=30, moisture_gain_per_minute=1.5)
        self.assertEqual(prediction.duration_minutes, 20)  # 30 points to the middle of the range, at 1.5 per minute
        self.assertIn("duracion_segun_riegos_anteriores", prediction.factors)

    def test_the_limits_still_hold(self):
        self.assertEqual(predict_irrigation(soil_moisture=30, moisture_gain_per_minute=0.01).duration_minutes, 90)


class ContractTests(unittest.TestCase):
    def test_the_old_request_gets_the_old_answer(self):
        answer = answer_prediction({"id_crop": 1, "soil_moisture": 30})
        self.assertEqual(set(answer), {"id_crop", "irrigate", "duration_minutes", "recommendation", "confidence",
                                       "model_used", "factors"})

    def test_the_new_fields_are_read(self):
        answer = answer_prediction({"soil_moisture": 35, "rain_probability": 90, "moisture_gain_per_minute": 2,
                                    "moisture_history": [{"hours_ago": h, "value": 35} for h in (2, 1, 0)]})
        self.assertFalse(answer["irrigate"])

    def test_bad_new_fields_are_refused(self):
        for body in ({"rain_probability": 101}, {"moisture_history": "x"}, {"moisture_history": ["x"]},
                     {"moisture_history": [{"hours_ago": 1}]}, {"moisture_history": [{"hours_ago": 1, "value": 1}] * 49}):
            with self.assertRaises(BadRequest):
                answer_prediction({"soil_moisture": 35, **body})


class StartupTests(unittest.TestCase):
    def test_an_open_service_on_a_public_address_refuses_to_start(self):
        self.assertIsNotNone(Settings().refusal({}))
        self.assertIsNone(Settings(internal_key="k").refusal({}))
        self.assertIsNone(Settings(service_token="t").refusal({}))
        self.assertIsNone(Settings().refusal({"HOST": "127.0.0.1"}))
        self.assertIsNone(Settings().refusal({"AI_ALLOW_OPEN": "true"}))


if __name__ == "__main__":
    unittest.main()
