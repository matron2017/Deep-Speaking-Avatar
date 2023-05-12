import os
from faster_whisper import WhisperModel
import torch
torch.cuda.empty_cache()
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




# Initialize the pygame mixer
pygame.mixer.init()
MIN_TRANSFORMERS_VERSION = '4.25.1'

# Check transformers version
assert transformers.__version__ >= MIN_TRANSFORMERS_VERSION, f'Please upgrade transformers to version {MIN_TRANSFORMERS_VERSION} or higher.'

# Initialize the RedPajama tokenizer and model
Redpajama_tokenizer = AutoTokenizer.from_pretrained("togethercomputer/RedPajama-INCITE-Chat-3B-v1")
Redpajama_model = AutoModelForCausalLM.from_pretrained("togethercomputer/RedPajama-INCITE-Chat-3B-v1", torch_dtype=torch.float16)
Redpajama_model = Redpajama_model.to('cuda:0')

# Initialize the speech synthesis processor, model, and vocoder
synthesis_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
synthesis_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
synthesis_vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# Load xvector containing speaker's voice characteristics from a dataset
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

# Set the Faster Whisper model path and initialize the model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
small_model_path = os.path.join(BASE_DIR, "small")
faster_whisper_model = WhisperModel(small_model_path, device="cpu", compute_type="int8")

# Initialize the inflect engine
engine = inflect.engine()

# Load the wake word model
wake_word_model_path = os.path.join(BASE_DIR, "Wakeword_model")
wake_word_model = models.load_model(wake_word_model_path)

# Set the recording path
recording_path = os.path.join(BASE_DIR, "question_recording.wav")

# Initialize the pygame mixer
pygame.mixer.init()






def detect_wake_word(wake_word_model):
    fs = 16000  # Set the sample rate
    seconds = 3  # Set the duration for the initial recording
    channel = 1  # Set the number of channels for the recording
    recording = sd.rec(int(seconds * fs), samplerate=fs, channels=channel)  # Record the initial audio
    sd.wait()
    first_audio = np.squeeze(recording)  # Remove the unnecessary dimensions

    long_audio = first_audio
    print("Say Now: ")
    while True:
        # Record 1-second audio
        recording = sd.rec(int(1 * fs), samplerate=fs, channels=channel)
        sd.wait()

        audio = np.squeeze(recording)
        # Update the long_audio by removing the oldest second and appending the new second
        long_audio = np.delete(long_audio, [index for index in range(int(np.shape(audio)[0]))])
        long_audio = np.hstack((long_audio, audio))

        # Compute the mel spectrogram and normalize it
        mel_spec = librosa.feature.melspectrogram(y=long_audio, sr=fs, n_fft=512, hop_length=160, n_mels=48, fmax=8000)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).T
        mel_spec_db_norm = (mel_spec_db - np.min(mel_spec_db)) / (np.max(mel_spec_db) - np.min(mel_spec_db))

        # Predict using the wake word model
        prediction = wake_word_model.predict(np.expand_dims(mel_spec_db_norm, axis=0))
        if prediction[:, 1] >= 0.50:
            # Play the audio response if the wake word is detected
            pygame.mixer.music.load("avatar_first_response.wav")
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)  # Check every 10ms to avoid high CPU usage.
            return




def record_question(recording_path, samplerate, duration):
    print("Recording question.....")
    #choice = input("Press 'r' to record 5 second audio clip or 'q' to exit \n")
    #if choice == 'r':
    question_audio = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1)
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
    sf.write("speech_synthesis.wav", speech.numpy(), samplerate=16000)
    pygame.mixer.music.load("speech_synthesis.wav")
    pygame.mixer.music.play()

    # Wait for the audio to finish playing
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)  # Check every 10ms to avoid high CPU usage.

    return
 






def main():
    while True:
        # Wake word detection
        torch.cuda.empty_cache()
        detect_wake_word(wake_word_model)

        # Recording question
        record_question(recording_path, 16000, 5)

        # Transcribing question
        result_text_segments, _ = faster_whisper_model.transcribe(recording_path, language="en", beam_size=5)
        question = ' '.join([segment.text for segment in result_text_segments])

        # Answering question
        answer = answer_question(question, Redpajama_tokenizer, Redpajama_model)

        # Speech synthesis
        speech_synthesis(answer, speaker_embeddings, synthesis_model, synthesis_processor, synthesis_vocoder)

main()
