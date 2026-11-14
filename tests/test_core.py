"""Kamera ve MediaPipe gerektirmeyen çekirdek iş mantığı testleri."""
import unittest

from hand_tracking import GestureGate, StableState, angle, analyze_fingers


class GeometryTests(unittest.TestCase):
    def test_straight_angle(self):
        self.assertAlmostEqual(angle((-1, 0, 0), (0, 0, 0), (1, 0, 0)), 180)

    def test_right_angle(self):
        self.assertAlmostEqual(angle((1, 0, 0), (0, 0, 0), (0, 1, 0)), 90)

    def test_invalid_landmark_count(self):
        with self.assertRaisesRegex(ValueError, '21 adet'):
            analyze_fingers([(0., 0., 0.)]*20)


class StableStateTests(unittest.TestCase):
    def test_waits_four_frames(self):
        stable = StableState(frames=4)
        for _ in range(3):
            self.assertIsNone(stable.update((True, False, False, False, False)))
        self.assertEqual(stable.update((True, False, False, False, False)),
                         (True, False, False, False, False))

    def test_resets_when_hand_lost(self):
        stable = StableState(frames=2)
        stable.update((True,)*5)
        self.assertEqual(stable.update((True,)*5), (True,)*5)
        self.assertIsNone(stable.update(None))
        self.assertIsNone(stable.update((True,)*5))


class GestureGateTests(unittest.TestCase):
    def test_requires_open_palm_before_next(self):
        gate = GestureGate(hold=.7, cooldown=1.5)
        palm = (True,)*5
        next_slide = (False, True, False, False, False)
        self.assertIsNone(gate.update(next_slide, 1.0))  # Henüz etkin değil
        self.assertIsNone(gate.update(palm, 2.0))
        self.assertIsNone(gate.update(next_slide, 2.1))
        self.assertEqual(gate.update(next_slide, 2.9), 'right')
        self.assertIsNone(gate.update(next_slide, 5.0))  # Yeniden açık avuç gerekiyor
        self.assertIsNone(gate.update(palm, 5.1))
        self.assertIsNone(gate.update(next_slide, 5.2))
        self.assertEqual(gate.update(next_slide, 6.0), 'right')


if __name__ == '__main__':
    unittest.main()
