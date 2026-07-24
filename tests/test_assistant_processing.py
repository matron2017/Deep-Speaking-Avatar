import re
import unittest

import numpy as np

from assistant_processing import (
    clean_generated_response,
    convert_number,
    normalize_mel_spectrogram,
    remove_incomplete_sentences,
    update_audio_window,
)


class FakeNumberEngine:
    def __init__(self):
        self.received_number = None

    def number_to_words(self, number):
        self.received_number = number
        return "forty-two"


class AssistantProcessingTests(unittest.TestCase):
    def test_update_audio_window_removes_old_samples_and_appends_new_samples(self):
        window = np.array([1, 2, 3, 4])
        new_audio = np.array([5, 6])

        result = update_audio_window(window, new_audio)

        np.testing.assert_array_equal(result, np.array([3, 4, 5, 6]))

    def test_update_audio_window_handles_new_array_as_long_as_window(self):
        window = np.array([1, 2, 3])
        new_audio = np.array([4, 5, 6])

        result = update_audio_window(window, new_audio)

        np.testing.assert_array_equal(result, new_audio)

    def test_update_audio_window_does_not_modify_inputs(self):
        window = np.array([1, 2, 3, 4])
        new_audio = np.array([5, 6])
        original_window = window.copy()
        original_new_audio = new_audio.copy()

        update_audio_window(window, new_audio)

        np.testing.assert_array_equal(window, original_window)
        np.testing.assert_array_equal(new_audio, original_new_audio)

    def test_normalize_mel_spectrogram_maps_values_proportionally(self):
        values = np.array([0.0, 5.0, 10.0])

        result = normalize_mel_spectrogram(values)

        np.testing.assert_allclose(result, np.array([0.0, 0.5, 1.0]))

    def test_normalize_mel_spectrogram_preserves_two_dimensional_shape(self):
        values = np.array([[0.0, 5.0], [10.0, 2.5]])

        result = normalize_mel_spectrogram(values)

        self.assertEqual(result.shape, values.shape)
        np.testing.assert_allclose(result, np.array([[0.0, 0.5], [1.0, 0.25]]))

    def test_convert_number_converts_match_to_integer_and_returns_engine_text(self):
        engine = FakeNumberEngine()
        match = re.search(r"\d+", "value 42")

        result = convert_number(match, engine)

        self.assertEqual(engine.received_number, 42)
        self.assertIsInstance(engine.received_number, int)
        self.assertEqual(result, "forty-two")

    def test_remove_incomplete_sentences_preserves_complete_final_punctuation(self):
        for punctuation in (".", "!", "?"):
            with self.subTest(punctuation=punctuation):
                text = f"Complete{punctuation}"
                self.assertEqual(remove_incomplete_sentences(text), text)

    def test_remove_incomplete_sentences_removes_incomplete_final_sentence(self):
        self.assertEqual(
            remove_incomplete_sentences("Complete. Incomplete"),
            "Complete.",
        )

    def test_remove_incomplete_sentences_returns_empty_for_only_incomplete_text(self):
        self.assertEqual(remove_incomplete_sentences("Incomplete"), "")

    def test_clean_generated_response_removes_human_continuation(self):
        self.assertEqual(
            clean_generated_response("Answer. <human> continuation", str),
            "Answer.",
        )

    def test_clean_generated_response_removes_bot_continuation(self):
        self.assertEqual(
            clean_generated_response("Answer. <bot> continuation", str),
            "Answer.",
        )

    def test_clean_generated_response_replaces_every_number(self):
        converted = []

        def number_converter(match):
            converted.append(match.group(0))
            return f"[{match.group(0)}]"

        result = clean_generated_response("There are 2 items and 42 examples.", number_converter)

        self.assertEqual(converted, ["2", "42"])
        self.assertEqual(result, "There are [2] items and [42] examples.")

    def test_clean_generated_response_replaces_numbers_and_removes_incomplete_sentence(self):
        result = clean_generated_response(
            "There are 42 examples. Unfinished",
            lambda match: f"number-{int(match.group(0))}",
        )

        self.assertEqual(result, "There are number-42 examples.")


if __name__ == "__main__":
    unittest.main()
