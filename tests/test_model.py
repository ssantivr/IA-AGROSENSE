import unittest

from agrosense_ia.model import MAX_CONFIDENCE, MAX_MINUTES, MIN_CONFIDENCE, MIN_MINUTES, MODEL_NAME, predict_irrigation


class IrrigationModelTests(unittest.TestCase):

    def test_dry_soil_is_irrigated_for_longer_the_drier_it_is(self):
        dry = predict_irrigation(soil_moisture=35, air_temperature=22, ph=6.5)
        drier = predict_irrigation(soil_moisture=15, air_temperature=22, ph=6.5)

        self.assertTrue(dry.irrigate)
        self.assertIn("Riega ahora", dry.recommendation)
        self.assertIn("35 %", dry.recommendation)
        self.assertIn("humedad_baja", dry.factors)
        self.assertGreater(drier.duration_minutes, dry.duration_minutes)
        self.assertGreater(drier.confidence, dry.confidence)
        self.assertEqual(dry.model_used, MODEL_NAME)

    def test_heat_lengthens_the_irrigation_and_cold_shortens_it(self):
        mild = predict_irrigation(30, 22, 6.5)
        hot = predict_irrigation(30, 34, 6.5)
        cold = predict_irrigation(30, 8, 6.5)

        self.assertGreater(hot.duration_minutes, mild.duration_minutes)
        self.assertLess(cold.duration_minutes, mild.duration_minutes)
        self.assertIn("amanecer", hot.recommendation)
        self.assertIn("temperatura_alta", hot.factors)
        self.assertIn("temperatura_baja", cold.factors)

    def test_the_duration_stays_within_its_limits(self):
        self.assertEqual(predict_irrigation(0, 45, 6.5).duration_minutes, MAX_MINUTES)
        self.assertGreaterEqual(predict_irrigation(39.9, 5, 6.5).duration_minutes, MIN_MINUTES)

    def test_moist_soil_is_left_alone(self):
        fine = predict_irrigation(65, 22, 6.5)
        soaked = predict_irrigation(92, 22, 6.5)

        self.assertFalse(fine.irrigate)
        self.assertEqual(fine.duration_minutes, 0)
        self.assertIn("No hace falta regar", fine.recommendation)
        self.assertFalse(soaked.irrigate)
        self.assertIn("drenaje", soaked.recommendation)
        self.assertIn("humedad_alta", soaked.factors)

    def test_soil_in_range_gets_a_short_irrigation_only_when_it_is_hot_and_below_the_middle(self):
        preventive = predict_irrigation(45, 33, 6.5)

        self.assertTrue(preventive.irrigate)
        self.assertEqual(preventive.duration_minutes, 10)
        self.assertIn("riego_preventivo", preventive.factors)
        self.assertFalse(predict_irrigation(45, 25, 6.5).irrigate)
        self.assertFalse(predict_irrigation(70, 33, 6.5).irrigate)

    def test_the_limits_themselves_count_as_in_range(self):
        self.assertFalse(predict_irrigation(40, 22, 6.5).irrigate)
        self.assertFalse(predict_irrigation(80, 22, 6.5).irrigate)
        self.assertEqual(predict_irrigation(40, 22, 5.5).factors, ["humedad_adecuada"])
        self.assertEqual(predict_irrigation(40, 22, 7.0).factors, ["humedad_adecuada"])

    def test_ph_and_frost_are_reported_without_changing_the_decision(self):
        acid = predict_irrigation(65, 22, 4.8)
        alkaline = predict_irrigation(65, 22, 8.1)
        frost = predict_irrigation(30, 1, 6.5)

        self.assertFalse(acid.irrigate)
        self.assertIn("ph_acido", acid.factors)
        self.assertIn("encalar", acid.recommendation)
        self.assertIn("ph_alcalino", alkaline.factors)
        self.assertTrue(frost.irrigate)
        self.assertIn("riesgo_de_helada", frost.factors)
        self.assertIn("helada", frost.recommendation)

    def test_the_crop_s_own_ranges_decide(self):
        rice = predict_irrigation(65, 22, 6.5, humidity_range=(70, 95))
        blueberry = predict_irrigation(65, 22, 5.0, ph_range=(4.5, 5.5))

        self.assertTrue(rice.irrigate)
        self.assertIn("70 %", rice.recommendation)
        self.assertEqual(blueberry.factors, ["humedad_adecuada"])
        with self.assertRaises(ValueError):
            predict_irrigation(65, 22, 6.5, humidity_range=(80, 40))

    def test_confidence_is_low_next_to_a_limit_and_never_leaves_its_bounds(self):
        borderline = predict_irrigation(39.9, 22, 6.5)
        clear = predict_irrigation(5, 22, 6.5)

        self.assertLess(borderline.confidence, 0.6)
        self.assertEqual(clear.confidence, MAX_CONFIDENCE)
        for moisture in range(0, 101, 5):
            for temperature in (-10, 5, 20, 31, 45):
                confidence = predict_irrigation(moisture, temperature, 6.5).confidence
                self.assertGreaterEqual(confidence, MIN_CONFIDENCE)
                self.assertLessEqual(confidence, MAX_CONFIDENCE)

    def test_without_temperature_or_ph_the_decision_rests_on_the_moisture_alone(self):
        dry = predict_irrigation(30)
        mild = predict_irrigation(30, 22, 6.5)

        self.assertTrue(dry.irrigate)
        self.assertEqual(dry.duration_minutes, mild.duration_minutes)
        self.assertEqual(dry.factors, ["humedad_baja"])
        self.assertEqual(predict_irrigation(45).factors, ["humedad_adecuada"])
        self.assertEqual(predict_irrigation(65, ph=4.8).factors, ["humedad_adecuada", "ph_acido"])

    def test_numbers_are_written_without_padding(self):
        self.assertIn("64.74 %", predict_irrigation(64.74, 19.26, 6.15).recommendation)
        self.assertIn("(40–80 %)", predict_irrigation(64, 20, 6.5).recommendation)


if __name__ == "__main__":
    unittest.main()
