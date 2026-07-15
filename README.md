# Deep Speaking Avatar

Deep Speaking Avatar (DSA) is a local voice assistant that combines wake-word detection, speech recognition, response generation, and speech synthesis.

The assistant waits for the wake word **“Avatar”**, records a question, converts it to text, generates an answer, and reads the answer aloud.

## System requirements

The project was developed and tested with:

- Ubuntu 20.04
- Python 3.8
- NVIDIA driver 510.39.01 or newer
- CUDA 11.6
- cuDNN 8.3.2
- NVIDIA Titan V with 12 GB of GPU memory

Other configurations may work but have not been tested. Wake-word detection runs on the CPU by default, leaving the GPU available for the other models.

## Installation

Clone the repository:

```bash
git clone git@github.com:matron2017/Deep-Speaking-Avatar.git
cd Deep-Speaking-Avatar
```

Create the Conda environment and install the Python dependencies:

```bash
conda env create --file environment.yaml
conda activate avatar_env
pip install -r requirements.txt
```

Install cuDNN and the CUDA-enabled version of PyTorch:

```bash
conda install -c conda-forge cudnn==8.3.2.44
conda install pytorch==1.13.1 torchvision==0.14.1 \
    torchaudio==0.13.1 pytorch-cuda=11.6 \
    -c pytorch -c nvidia
```

Several model files are too large to include in this repository. Install [Git LFS](https://git-lfs.com/) before downloading them:

```bash
git lfs install
```

## Models

### 1. Speech recognition: Faster Whisper

[Faster Whisper](https://github.com/guillaumekln/faster-whisper) transcribes recorded speech using a CTranslate2 version of the Whisper Small model.

Install Whisper and Faster Whisper:

```bash
pip install -U openai-whisper
pip install --force-reinstall \
    "faster-whisper @ https://github.com/guillaumekln/faster-whisper/archive/refs/heads/master.tar.gz"
pip install --force-reinstall numpy==1.22
```

Download the converted model:

```bash
git clone https://huggingface.co/guillaumekln/faster-whisper-small small
```

#### Example

```python
from faster_whisper import WhisperModel

model_path = "small"
recording_path = "audio_recording.wav"

model = WhisperModel(
    model_path,
    device="cpu",
    compute_type="int8",
)

# GPU alternative:
# model = WhisperModel(
#     model_path,
#     device="cuda",
#     compute_type="float16",
# )

segments, _ = model.transcribe(
    recording_path,
    language="en",
    beam_size=5,
)

transcription = " ".join(segment.text for segment in segments)
print(transcription)
```

### 2. Response generation: RedPajama

[RedPajama-INCITE-Chat-3B-v1](https://huggingface.co/togethercomputer/RedPajama-INCITE-Chat-3B-v1) generates a response from the transcribed question.

Download the model:

```bash
git clone \
    https://huggingface.co/togethercomputer/RedPajama-INCITE-Chat-3B-v1
```

Running the model on a GPU requires approximately 8 GB of GPU memory.

#### Example

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "togethercomputer/RedPajama-INCITE-Chat-3B-v1"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
).to("cuda:0")

question = "What is the capital of France?"
prompt = f"<human>: {question}\n<bot>:"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
input_length = inputs.input_ids.shape[1]

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=45,
        do_sample=True,
        temperature=0.8,
        top_p=0.7,
        top_k=70,
    )

generated_tokens = outputs[0, input_length:]
answer = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True,
)

print(answer)
```

### 3. Speech synthesis: SpeechT5

[SpeechT5](https://huggingface.co/microsoft/speecht5_tts) converts the generated answer into speech. A speaker embedding from the CMU ARCTIC dataset determines the output voice.

Download the model:

```bash
git clone https://huggingface.co/microsoft/speecht5_tts
```

Install `sounddevice` from the following Conda channel to avoid audio compatibility problems on Ubuntu:

```bash
conda install -c "conda-forge/label/cf201901" python-sounddevice
```

#### Example

```python
import torch
from datasets import load_dataset
from transformers import (
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    SpeechT5Processor,
)

processor = SpeechT5Processor.from_pretrained(
    "microsoft/speecht5_tts"
)
model = SpeechT5ForTextToSpeech.from_pretrained(
    "microsoft/speecht5_tts"
)
vocoder = SpeechT5HifiGan.from_pretrained(
    "microsoft/speecht5_hifigan"
)

embeddings = load_dataset(
    "Matthijs/cmu-arctic-xvectors",
    split="validation",
)
speaker_embedding = torch.tensor(
    embeddings[7306]["xvector"]
).unsqueeze(0)

inputs = processor(
    text="How are you doing today?",
    return_tensors="pt",
)

speech_audio = model.generate_speech(
    inputs["input_ids"],
    speaker_embedding,
    vocoder=vocoder,
)
```

## Running the assistant

Activate the Conda environment:

```bash
conda activate avatar_env
```

Open the project directory:

```bash
cd /path/to/Deep-Speaking-Avatar
```

Run the main script:

```bash
python3 avatar4.py
```

Initialisation usually takes one to two minutes. After loading the models, the program:

1. Waits for the wake word **“Avatar”**.
2. Records five seconds of audio.
3. Transcribes the recording with Faster Whisper.
4. Generates an answer with RedPajama.
5. Converts the answer to speech with SpeechT5.
6. Returns to wake-word detection.

Say “Avatar” and then ask a question.

## Troubleshooting

### CUDA out-of-memory errors

Repeated runs may leave insufficient GPU memory available. Check GPU usage with:

```bash
nvidia-smi
```

Close inactive Python processes that are still using the GPU. If the memory is not released, restart the computer before running the program again.

### Optional TensorFlow GPU support

Wake-word detection uses the CPU by default. TensorFlow GPU support can be installed separately:

```bash
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/"

mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"

echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' \
    > "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh"

pip install --upgrade pip
python3 -m pip install tensorflow==2.9.3
```

## Possible improvements

- Reduce speech-synthesis latency.
- Replace RedPajama with a newer language model.
- Use voice-activity detection instead of a fixed five-second recording.
- Improve GPU memory management between models.
- Allow the wake word and recording language to be configured.
