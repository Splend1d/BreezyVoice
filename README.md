# BreezyVoice

BreezyVoice is a voice-cloning text-to-speech system specifically adapted for Taiwanese Mandarin, highlighting phonetic control abilities via auxiliary 注音 (bopomofo) inputs. BreezyVoice is partially derived from [CosyVoice](https://github.com/FunAudioLLM/CosyVoice).

[Playground](); [Model](https://huggingface.co/MediaTek-Research/BreezyVoice-300M/tree/main); [Paper](https://arxiv.org/abs/2501.17790)

Repo Main Contributors: Chia-Chun Lin, Chan-Jan Hsu

## Install

**Clone and install**

- Clone the repo
``` sh
git clone https://github.com/mtkresearch/BreezyVoice.git
# If you failed to clone submodule due to network failures, please run following command until success
cd BreezyVoice
```

- Install Requirements (requires Python3.10)
```
pip install -r requirements.txt
```
(The model is runnable on CPU, please change onnxruntime-gpu to onnxruntime in `requirements.txt` if you do not have GPU in your environment)

## Inference

UTF8 encoding is required:

``` sh
export PYTHONUTF8=1
```
---
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

``` bash
bash run_single_inference.sh
```

``` python
# python single_inference.py --text_to_speech [text to be converted into audio] --text_prompt [the prompt of that audio file] --audio_path [reference audio file]
python single_inference.py --text_to_speech "今天天氣真好" --text_prompt "在密碼學中，加密是將明文資訊改變為難以讀取的密文內容，使之不可讀的方法。" --audio_path "./data/tc_speaker.wav"
```

``` python
# python single_inference.py --text_to_speech [text to be converted into audio] --audio_path [reference audio file]
python single_inference.py --text_to_speech "今天天氣真好[:ㄏㄠ3]" --audio_path "./data/tc_speaker.wav"
```

---

**Run `batch_inference.py` with the following arguments:**

- `--csv_file`:
  - **Description**: Path to the CSV file that contains the input data for batch processing.
  - **Example**: `./data/batch_files.csv`

- `--speaker_prompt_audio_folder`:
  - **Description**: Path to the folder containing the speaker prompt audio files. The files in this folder are used to set the style of the speaker for each synthesis task.
  - **Example**: `./data`

- `--output_audio_folder`:
  - **Description**: Path to the folder where the output audio files will be saved. Each processed row in the CSV will result in a synthesized audio file stored in this folder.
  - **Example**: `./results`

**CSV File Structure:**

The CSV file should contain the following columns:

- **`speaker_prompt_audio_filename`**:
  - **Description**: The filename (without extension) of the speaker prompt audio file that will be used to guide the style of the generated speech.
  - **Example**: `example`

- **`speaker_prompt_text_transcription`**:
  - **Description**: The transcription of the speaker prompt audio. This field is optional but highly recommended to improve transcription accuracy. If not provided, the system will attempt to transcribe the audio using Whisper.
  - **Example**: `"在密碼學中，加密是將明文資訊改變為難以讀取的密文內容，使之不可讀的方法。"`

- **`content_to_synthesize`**:
  - **Description**: The content that will be synthesized into speech. You can include phonetic symbols if needed, though they should be used sparingly.
  - **Example**: `"今天天氣真好"`

- **`output_audio_filename`**:
  - **Description**: The filename (without extension) for the generated output audio. The audio will be saved as a `.wav` file in the output folder.
  - **Example**: `output`

**Example Usage:**

``` bash
bash run_batch_inference.sh
```
```bash
python batch_inference.py \
  --csv_file ./data/batch_files.csv \
  --speaker_prompt_audio_folder ./data \
  --output_audio_folder ./results
```

---

If you like our work, please cite:

```
@article{hsu2025breezyvoice,
  title={BreezyVoice: Adapting TTS for Taiwanese Mandarin with Enhanced Polyphone Disambiguation--Challenges and Insights},
  author={Hsu, Chan-Jan and Lin, Yi-Cheng and Lin, Chia-Chun and Chen, Wei-Chih and Chung, Ho Lam and Li, Chen-An and Chen, Yi-Chang and Yu, Chien-Yu and Lee, Ming-Ji and Chen, Chien-Cheng and others},
  journal={arXiv preprint arXiv:2501.17790},
  year={2025}
}
```
