# The Deep Speaking Avatar (DSA)

The Deep Speaking Avatar (DSA) is an conversational agent that combines four essential modules: wake-word recognition, speech-to-text, language model, and text-to-speech. This repository contains the source code, setup instructions, examples, and documentation to help you build, run, and customize the DSA.

**The project requires Python 3.8.x, Linux Ubuntu 20.04, Nvidia Driver Version >=510.39.01, CUDA Version: 11.6, and cuDNN 8.3.2.**
Although other version combinations might work, it is strongly recommended to use the listed versions. This project was developed and tested using the NVIDIA GTX Titan V 12 GB GPU. By default, due to the limited computational capability of the GTX Titan V GPU, this project employs TensorFlow with CPU support for wake-word detection.

## Getting Started
Clone the repository and navigate to the project directory.
```bash
git clone git@github.com:matron2017/Deep-Speaking-Avatar.git
cd Deep-Speaking-Avatar
```
Create a new conda environment from the conda file and install the pip dependencies.
```bash
conda env create --file environment.yaml
conda activate avatar_env
pip install -r requirements.txt
```
Install cuDNN 8.3.2 and PyTorch 1.13.1 with CUDA 11.6:
```bash
conda install -c conda-forge cudnn==8.3.2.44
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
```



This project relies on several large model files that are not included in the GitHub repository due to their size. Before running the project, you must download these models. Detailed instructions for obtaining and setting up the required model files can be found in the `Modules` section. Make sure you have git-lfs installed (https://git-lfs.com) before going to the module section.
```bash
git lfs install
```
 
## Modules


### 1. Speech-to-Text: Faster Whisper

Faster Whisper is an optimized version of OpenAI's Whisper Small model, designed for automatic speech recognition (ASR) and speech translation tasks. The original model has been converted to the CTranslate2 format for improved inference speed.

First you need to install [whisper](https://github.com/openai/whisper) and [faster-whisper](https://github.com/guillaumekln/faster-whisper) packages:

```bash
pip install -U openai-whisper
pip install --force-reinstall "faster-whisper @ https://github.com/guillaumekln/faster-whisper/archive/refs/heads/master.tar.gz" && pip install --force-reinstall numpy==1.22
```
Clone the converted model from [guillaumekln/faster-whisper-small](https://huggingface.co/guillaumekln/faster-whisper-small) repository:
```bash
git clone https://huggingface.co/guillaumekln/faster-whisper-small small
```


#### Example Usage

```python
# Import the WhisperModel from the faster_whisper module
from faster_whisper import WhisperModel

# Specify the path where the pre-trained model is located
small_model_path = ".../whisper-small-ct2"

# Initialize the Faster Whisper model. Set the device to CPU and the compute type to int8.
# These settings are suitable for low-compute devices or when GPU resources are limited.
faster_whisper_model = WhisperModel(small_model_path, device="cpu", compute_type="int8")

# Alternatively, if you have a capable GPU, you can use it for faster computation.
# Uncomment the following line to initialize the model with GPU support and float16 compute type.
# faster_whisper_model = WhisperModel(small_model_path, device="cuda", compute_type="float16")

# Specify the path to the audio recording file that you want to transcribe
recording_path ='.../audio_recording.wav'

# The transcribe method returns a list of text segments (result_text_segments).
result_text_segments, _ = faster_whisper_model.transcribe(recording_path, language="en", beam_size=5)

# Join the text segments into a single string to get the complete transcription.
result_text = ' '.join([segment.text for segment in result_text_segments])
```



### 2. Large Languge Model: RedPajama-INCITE-Chat-3B-v1

The RedPajama model is a powerful large-scale language model designed for natural language understanding and generation tasks. It can generate human-like responses based on given prompts.

Clone the RedPajama model [togethercomputer/RedPajama-INCITE-Chat-3B-v1](https://huggingface.co/togethercomputer/RedPajama-INCITE-Chat-3B-v1):

```bash
git clone https://huggingface.co/togethercomputer/RedPajama-INCITE-Chat-3B-v1
```

#### Example usage (with GPU):

Ensure you have a 8GB GPU available to run the model on GPU.
```python
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

# Ensure the required version of the transformers library is installed
MIN_TRANSFORMERS_VERSION = '4.25.1'
assert transformers.__version__ >= MIN_TRANSFORMERS_VERSION, f'Please upgrade transformers to version {MIN_TRANSFORMERS_VERSION} or higher.'

# Load the tokenizer and the model from the pretrained RedPajama model
Redpajama_tokenizer = AutoTokenizer.from_pretrained("togethercomputer/RedPajama-INCITE-Chat-3B-v1")
Redpajama_model = AutoModelForCausalLM.from_pretrained("togethercomputer/RedPajama-INCITE-Chat-3B-v1", torch_dtype=torch.float16)

# Move the model to GPU
Redpajama_model = Redpajama_model.to('cuda:0')

# Prepare the prompt for the model. The model expects a conversation-like input 
#  where it alternates between a 'human' and a 'bot'.
question = "What is the capital of France?"
prompt = f"<human>: {question}\n<bot>:"
inputs = Redpajama_tokenizer(prompt, return_tensors='pt').to(Redpajama_model.device)
input_length = inputs.input_ids.shape[1]
outputs = Redpajama_model.generate(**inputs, max_new_tokens=45, do_sample=True, temperature=0.8, top_p=0.7, top_k=70, return_dict_in_generate=True)
token = outputs.sequences[0, input_length:]
# Decode the tokens to text
text_answer = Redpajama_tokenizer.decode(token)
```

### 3. Text-to-Speech (SpeechT5)

The SpeechT5 model is a powerful text-to-speech model fine-tuned for speech synthesis on the LibriTTS dataset. Clone the model [microsoft/speecht5_tts](https://huggingface.co/microsoft/speecht5_tts):

```bash
git clone https://huggingface.co/microsoft/speecht5_tts
```
Lastly, you need to install `sounddevice` library with conda from this specific source to avoid audio problems with Ubuntu:

```bash
conda install -c "conda-forge/label/cf201901" python-sounddevice
```

#### Example usage

```python
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch

# Initialize the processor, model, and vocoder
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# Load speaker embeddings from the CMU ARCTIC dataset
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

# Process text input and generate speech audio
inputs = processor(text="How are you doing today?", return_tensors="pt")
speech_audio = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

```

## Running the Avatar

Activate the conda environment
```bash
conda activate avatar_env
```

Navigate to the project directory
```bash
cd /path/to/Deep-Speaking-Avatar
```
Run the Avatar script
```bash
python3 avatar4.py
```
Starting the program usually takes from 1 to 2 minutes.

1. The program begins an infinite loop.
2. Wake word detection is performed, waiting for the user to say the wake word, which is "Avatar".
3. Upon detecting the wake word, the program records a 5-second question from the user.
4. The recorded question is transcribed using the `faster_whisper_model`.
5. The transcribed question is passed to the `answer_question` function, which generates a response using the `Redpajama_tokenizer` and `Redpajama_model`.
6. The answer is then converted to speech using the `speech_synthesis` function.
7. The loop repeats, waiting for the next wake word.

To interact with the conversational avatar, simply say the wake word "Avatar" followed by your question. The avatar will then respond using speech synthesis. The process for the avatar's operation is as follows:

## Troubleshooting

Repeatedly running the 'avatar4.py' script may sometimes cause a "CUDA out of memory error" or a torch error. This can be resolved by restarting the computer or clearing the GPU memory.

This project uses TensorFlow with CPU support due to the limited computational capability of the NVIDIA GTX Titan V GPU. If you want to use TensorFlow with GPU, you need to install it separately.
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
pip install --upgrade pip
python3 -m pip install tensorflow==2.9.3
```

## Proposed Improvements

1. **Speech Synthesis Performance:**:  Implement alternative speech synthesis methods for faster response times. Another option is to upgrade the GPU for better processing speed.
2. **Improved Large Language Model**: Incorporate more advanced large language models for increased accuracy in response generation.
3. **Question Length Adaptability**: Include an automatic speech recognition model that adjusts the recorded question length based on detected voice activity, enhancing the system's flexibility.
