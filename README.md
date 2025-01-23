# BreezyVoice

BreezyVoice is a voice-cloning text-to-speech system specifically adapted for Taiwanese Mandarin, highlighting phonetic control abilities via auxiliary bopomofo inputs.

[Playground](); [Model](https://huggingface.co/MediaTek-Research/BreezyVoice-300M/tree/main); [Paper]()

## Install

**Clone and install**

- Clone the repo
``` sh
git clone https://github.com/mtkresearch/BreezyVoice.git
# If you failed to clone submodule due to network failures, please run following command until success
cd BreezyVoice
```

- Install Requirements
```
pip install -r requirements.txt
```
(The model is runnable on CPU, please change onnxruntime-gpu to onnxruntime in `requirements.txt` if you do not have GPU in your environment)

## Inference

utf-8 encoding is required:

``` sh
export PYTHONUTF8=1
```

**Run single_inference.py with the following arguments:**

- `--content_to_synthesize`:
    - **Description**: Specifies the content that will be synthesized into speech. Phonetic symbols can optionally be included but should be used sparingly, as shown in the examples below:
    - Simple text: `"今天天氣真好"`
    - Text with phonetic symbols: `"今天天氣真好[:ㄏㄠ3]"`

- `--speaker_prompt_audio_path`:
  - **Description**: Specifies the path to the prompt speech audio file for setting the style of the speaker. Use your custom audio file or our example file:
    - Example audio: `./data/tc_speaker.wav`

- `--speaker_prompt_text_transcription` (optional):
  - **Description**: Specifies the transcription of the speaker prompt audio. Providing this input is highly recommended for better accuracy. If not provided, the system will automatically transcribe the audio using Whisper.
  - Example text for the audio file: `"在密碼學中，加密是將明文資訊改變為難以讀取的密文內容，使之不可讀的方法。"`

- `--output_path` (optional):
  - **Description**: Specifies the name and path for the output `.wav` file. If not provided, the default path is used.
  - **Default Value**: `results/output.wav`
  - Example: `[your_file_name].wav`

- `--model_path` (optional):
  - **Description**: Specifies the pre-trained model used for speech synthesis.
  - **Default Value**: `MediaTek-Research/BreezyVoice-300M`

**Example Usage:**

``` python
# python simple_use.py --text_to_speech [text to be converted into audio] --text_prompt [the prompt of that audio file] --audio_path [reference audio file]
python simple_use.py --text_to_speech "今天天氣真好" --text_prompt "在密碼學中，加密是將明文資訊改變為難以讀取的密文內容，使之不可讀的方法。" --audio_path "./data/tc_speaker.wav"
```

``` python
# python simple_use.py --text_to_speech [text to be converted into audio] --audio_path [reference audio file]
python simple_use.py --text_to_speech "今天天氣真好[:ㄏㄠ3]" --audio_path "./data/tc_speaker.wav"
```