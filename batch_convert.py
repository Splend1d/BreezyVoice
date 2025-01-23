import csv
import os
import subprocess

BATCH_FOLDER = "data"
RESULT_FOLDER = "results"

os.makedirs(RESULT_FOLDER, exist_ok=True)

CSV_FILE = "mtk_batch_transform.csv"

SINGLE_INFERENCE_SCRIPT = "single_inference.py"

def process_batch(csv_file, batch_folder, result_folder, script_path):
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            audio_path = os.path.join(batch_folder, f"{row['speaker_audio']}.wav")
            text_prompt = row['speaker_label']
            text_to_speech = row['label_of_gen_audio']
            gen_voice_file_name = os.path.join(result_folder, f"{row['gen_audio']}.wav")
            if not os.path.exists(audio_path):
                print(f"File{audio_path} not exist")
                continue

            command = [
                "python", script_path,
                "--speaker_prompt_audio_path", audio_path,
                "--speaker_prompt_text_transcription", text_prompt,
                "--content_to_synthesize", text_to_speech,
                "--output_path", gen_voice_file_name
            ]
            try:
                print(f"In Process...:{audio_path}")
                subprocess.run(command, check=True)
                print(f"Generate Complete:{gen_voice_file_name}")
            except subprocess.CalledProcessError as e:
                print(f"Fail to generate{audio_path}, error:{e}")

if __name__ == "__main__":
    process_batch(CSV_FILE, BATCH_FOLDER, RESULT_FOLDER, SINGLE_INFERENCE_SCRIPT)
