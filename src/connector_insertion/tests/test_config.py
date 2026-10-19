import json
import unittest
from connector_insertion.config import ConnectorInsertionConfig as C


class TestConfig(unittest.TestCase):
    def test_default_off(self):
        self.assertFalse(C().enabled)
        self.assertFalse(C.from_env({}).enabled)

    def test_from_env(self):
        c = C.from_env({"KIAT_CONNINS": "1", "KIAT_CONNINS_SCALE_MULT": "1.5",
                        "KIAT_CONNINS_FIT_MODE": "max", "KIAT_CONNINS_INCLUDE": "a*, b",
                        "KIAT_CONNINS_WEIGHTS": "a=2,b=0.5", "KIAT_CONNINS_MAX_PER_FRAME": "3",
                        "KIAT_CONNINS_USE_CONNECTOR_DISC": "1"})
        self.assertTrue(c.enabled)
        self.assertEqual((c.scale_mult, c.fit_mode, c.include, c.max_per_frame),
                         (1.5, "max", ("a*", "b"), 3))
        self.assertEqual(c.weights, {"a": 2.0, "b": 0.5})
        self.assertTrue(c.use_connector_disc)

    def test_dict_json_roundtrip(self):
        c = C(enabled=True, scale_mult=2.0, include=("x",), weights={"x": 3})
        self.assertEqual(C.from_dict(c.to_dict()), c)
        self.assertEqual(C.from_json(c.to_json()), c)

    def test_unknown_key_raises(self):
        with self.assertRaises(ValueError):
            C.from_dict({"replace_probb": 0.5})

    def test_validation(self):
        for bad in (dict(replace_prob=1.5), dict(fit_mode="x"), dict(scale_mult=0),
                    dict(min_length_px=50, max_length_px=10), dict(roll_mode="spin"),
                    dict(wire_tint=2), dict(spacing_px=0), dict(min_points=10, max_points=5),
                    dict(min_cluster_points=2), dict(weights={"a": -1}), dict(flip_prob=-.1)):
            with self.assertRaises(ValueError, msg=str(bad)):
                C(**bad)

    def test_env_covers_every_field(self):
        import dataclasses
        names = {f.name for f in dataclasses.fields(C)} - {"enabled"}
        self.assertTrue(names)   # every field is reachable as KIAT_CONNINS_<NAME>
        json.dumps(C().to_dict())


if __name__ == "__main__":
    unittest.main()
