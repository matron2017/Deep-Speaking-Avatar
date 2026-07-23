import os
from faster_whisper import WhisperModel
import torch
from tensorflow.keras import models
import librosa
import numpy as np
from scipy.io.wavfile import write
import tools
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import soundfile as sf
import inflect
import re
import pygame
from num2words import num2words
import sounddevice as sd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REDPAJAMA_MODEL_ID = "togethercomputer/RedPajama-INCITE-Chat-3B-v1"
SPEECHT5_MODEL_ID = "microsoft/speecht5_tts"
SPEECHT5_VOCODER_ID = "microsoft/speecht5_hifigan"
EMBEDDINGS_DATASET_ID = "Matthijs/cmu-arctic-xvectors"
EMBEDDINGS_SPLIT = "validation"
SPEAKER_EMBEDDING_INDEX = 7306
WHISPER_MODEL_PATH = os.path.join(BASE_DIR, "small")
WAKE_WORD_MODEL_PATH = os.path.join(BASE_DIR, "Wakeword_model")
RECORDING_PATH = os.path.join(BASE_DIR, "question_recording.wav")
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"
REDPAJAMA_DEVICE = "cuda:0"
MIN_TRANSFORMERS_VERSION = "4.25.1"
SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
INITIAL_WAKE_WORD_RECORDING_DURATION = 3
WAKE_WORD_CHUNK_DURATION = 1
WAKE_WORD_THRESHOLD = 0.50
QUESTION_RECORDING_DURATION = 5
TRANSCRIPTION_LANGUAGE = "en"
TRANSCRIPTION_BEAM_SIZE = 5
FIRST_RESPONSE_PATH = "avatar_first_response.wav"
SPEECH_OUTPUT_PATH = "speech_synthesis.wav"


engine = inflect.engine()

def detect_wake_word(wake_word_model):
    recording = sd.rec(int(INITIAL_WAKE_WORD_RECORDING_DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=AUDIO_CHANNELS)
    sd.wait()
    first_audio = np.squeeze(recording)  # Remove the unnecessary dimensions

    long_audio = first_audio
    print("Say Now: ")
    while True:
        # Record 1-second audio
        recording = sd.rec(int(WAKE_WORD_CHUNK_DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=AUDIO_CHANNELS)
        sd.wait()

        audio = np.squeeze(recording)
        # Update the long_audio by removing the oldest second and appending the new second
        long_audio = np.delete(long_audio, [index for index in range(int(np.shape(audio)[0]))])
        long_audio = np.hstack((long_audio, audio))

        # Compute the mel spectrogram and normalize it
        mel_spec = librosa.feature.melspectrogram(y=long_audio, sr=SAMPLE_RATE, n_fft=512, hop_length=160, n_mels=48, fmax=8000)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).T
        mel_spec_db_norm = (mel_spec_db - np.min(mel_spec_db)) / (np.max(mel_spec_db) - np.min(mel_spec_db))

        # Predict using the wake word model
        prediction = wake_word_model.predict(np.expand_dims(mel_spec_db_norm, axis=0))
        if prediction[:, 1] >= WAKE_WORD_THRESHOLD:
            # Play the audio response if the wake word is detected
            pygame.mixer.music.load(FIRST_RESPONSE_PATH)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)  # Check every 10ms to avoid high CPU usage.
            return




def record_question(recording_path, samplerate, duration):
    print("Recording question.....")
    #choice = input("Press 'r' to record 5 second audio clip or 'q' to exit \n")
    #if choice == 'r':
    question_audio = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=AUDIO_CHANNELS)
    sd.wait()
    write(recording_path, samplerate, question_audio)
    return 



def answer_question(question, Redpajama_tokenizer, Redpajama_model):
    # Format the question for the model
    question = f"<human>: {question}\n<bot>:" 
    prompt = f"{question}:"

    # Tokenize the prompt and move it to the appropriate device
    inputs = Redpajama_tokenizer(prompt, return_tensors='pt').to(Redpajama_model.device)
    input_length = inputs.input_ids.shape[1]

    # Generate a response using the model
    outputs = Redpajama_model.generate(**inputs, max_new_tokens=65, do_sample=True, temperature=0.8, top_p=0.7, top_k=70, return_dict_in_generate=True)
    token = outputs.sequences[0, input_length:]
    output_str = Redpajama_tokenizer.decode(token)

    # Clean up the output string
    output_str = re.sub(r'<human>.*', '', output_str)
    output_str = re.sub(r'<bot>.*', '', output_str)

    # Convert numbers to words and remove incomplete sentences
    converted_text = re.sub(r'\d+', convert_number, output_str)
    cleaned_text = remove_incomplete_sentences(converted_text)

    return cleaned_text







def convert_number(match):
    # Convert a number to its word form
    number = int(match.group(0))
    return engine.number_to_words(number)






def remove_incomplete_sentences(text):
    # Split the text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Check if the last sentence ends with proper punctuation
    if not re.search(r'[.!?]\s*$', sentences[-1]):
        sentences.pop()

    # Join the sentences back together
    return ' '.join(sentences)






def speech_synthesis(answer, speaker_embeddings, synthesis_model, synthesis_processor, synthesis_vocoder):
    # Prepare the text input for the speech synthesis model
    inputs = synthesis_processor(text=answer, return_tensors="pt")

    # Generate speech from the text input
    speech = synthesis_model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=synthesis_vocoder)

    # Save the generated speech to a file and play it
    sf.write(SPEECH_OUTPUT_PATH, speech.numpy(), samplerate=SAMPLE_RATE)
    pygame.mixer.music.load(SPEECH_OUTPUT_PATH)
    pygame.mixer.music.play()

    # Wait for the audio to finish playing
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)  # Check every 10ms to avoid high CPU usage.

    return
 






def main():
    torch.cuda.empty_cache()
    pygame.mixer.init()

    # Check transformers version
    assert transformers.__version__ >= MIN_TRANSFORMERS_VERSION, f'Please upgrade transformers to version {MIN_TRANSFORMERS_VERSION} or higher.'

    # Initialize the RedPajama tokenizer and model
    Redpajama_tokenizer = AutoTokenizer.from_pretrained(REDPAJAMA_MODEL_ID)
    Redpajama_model = AutoModelForCausalLM.from_pretrained(REDPAJAMA_MODEL_ID, torch_dtype=torch.float16)
    Redpajama_model = Redpajama_model.to(REDPAJAMA_DEVICE)

    # Initialize the speech synthesis processor, model, and vocoder
    synthesis_processor = SpeechT5Processor.from_pretrained(SPEECHT5_MODEL_ID)
    synthesis_model = SpeechT5ForTextToSpeech.from_pretrained(SPEECHT5_MODEL_ID)
    synthesis_vocoder = SpeechT5HifiGan.from_pretrained(SPEECHT5_VOCODER_ID)

    # Load xvector containing speaker's voice characteristics from a dataset
    embeddings_dataset = load_dataset(EMBEDDINGS_DATASET_ID, split=EMBEDDINGS_SPLIT)
    speaker_embeddings = torch.tensor(embeddings_dataset[SPEAKER_EMBEDDING_INDEX]["xvector"]).unsqueeze(0)

    # Set the Faster Whisper model path and initialize the model
    faster_whisper_model = WhisperModel(WHISPER_MODEL_PATH, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)

    # Load the wake word model
    wake_word_model = models.load_model(WAKE_WORD_MODEL_PATH)

    while True:
        # Wake word detection
        torch.cuda.empty_cache()
        detect_wake_word(wake_word_model)

        # Recording question
        record_question(RECORDING_PATH, SAMPLE_RATE, QUESTION_RECORDING_DURATION)

        # Transcribing question
        result_text_segments, _ = faster_whisper_model.transcribe(RECORDING_PATH, language=TRANSCRIPTION_LANGUAGE, beam_size=TRANSCRIPTION_BEAM_SIZE)
        question = ' '.join([segment.text for segment in result_text_segments])

        # Answering question
        answer = answer_question(question, Redpajama_tokenizer, Redpajama_model)

        # Speech synthesis
        speech_synthesis(answer, speaker_embeddings, synthesis_model, synthesis_processor, synthesis_vocoder)

if __name__ == "__main__":
    main()
