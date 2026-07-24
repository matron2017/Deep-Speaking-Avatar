import re

import numpy as np


def update_audio_window(long_audio, new_audio):
    long_audio = np.delete(long_audio, [index for index in range(int(np.shape(new_audio)[0]))])
    return np.hstack((long_audio, new_audio))


def normalize_mel_spectrogram(mel_spec_db):
    minimum = np.min(mel_spec_db)
    maximum = np.max(mel_spec_db)
    return (mel_spec_db - minimum) / (maximum - minimum)


def convert_number(match, number_engine):
    number = int(match.group(0))
    return number_engine.number_to_words(number)


def remove_incomplete_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)

    if not re.search(r'[.!?]\s*$', sentences[-1]):
        sentences.pop()

    return ' '.join(sentences)


def clean_generated_response(output_text, number_converter):
    output_text = re.sub(r'<human>.*', '', output_text)
    output_text = re.sub(r'<bot>.*', '', output_text)
    output_text = re.sub(r'\d+', number_converter, output_text)
    return remove_incomplete_sentences(output_text)
