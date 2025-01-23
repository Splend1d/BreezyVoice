import argparse
import os
import sys
import re
from functools import partial
import time

import torch
import torchaudio
import whisper
import opencc
from hyperpyyaml import load_hyperpyyaml
from huggingface_hub import snapshot_download
from g2pw import G2PWConverter

from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.cli.model import CosyVoiceModel
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.utils.frontend_utils import (contains_chinese, replace_blank, replace_corner_mark,remove_bracket, spell_out_number, split_paragraph)
from utils.word_utils import word_to_dataset_frequency, char2phn

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))

####new normalize
class CustomCosyVoiceFrontEnd(CosyVoiceFrontEnd):
    def text_normalize_new(self,text, split=True):
        text = text.strip()
        def normalize_outside_brackets(match):
            inner = match.group(1)  
            outer = match.group(2)  
            if contains_chinese(outer):
                if self.use_ttsfrd:
                    outer = self.frd.get_frd_extra_info(outer, 'input')
                else:
                    outer = self.zh_tn_model.normalize(outer)
                outer = outer.replace("\n", "")
                outer = replace_blank(outer)
                outer = replace_corner_mark(outer)
                outer = outer.replace(".", "、")
                outer = outer.replace(" - ", "，")
                outer = remove_bracket(outer)
                outer = re.sub(r'[，,]+$', '。', outer)
            else:
                if self.use_ttsfrd:
                    outer = self.frd.get_frd_extra_info(outer, 'input')
                else:
                    outer = self.en_tn_model.normalize(outer)
                outer = spell_out_number(outer, self.inflect_parser)
            return inner + outer

        text = re.sub(r'(\[[^\]]*\])(.*?)', normalize_outside_brackets, text)

        if contains_chinese(text):
            texts = [i for i in split_paragraph(
                text, partial(self.tokenizer.encode, allowed_special=self.allowed_special),
                "zh", token_max_n=80, token_min_n=60, merge_len=20, comma_split=False
            )]
        else:
            texts = [i for i in split_paragraph(
                text, partial(self.tokenizer.encode, allowed_special=self.allowed_special),
                "en", token_max_n=80, token_min_n=60, merge_len=20, comma_split=False
            )]

        if split is False:
            return text
        return texts

####model
class CosyVoiceModelv2(CosyVoiceModel):

    def __init__(self,
                 llm: torch.nn.Module,
                 flow: torch.nn.Module,
                 hift: torch.nn.Module):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.llm = llm
        self.flow = flow
        self.hift = hift

    def load(self, llm_model, flow_model, hift_model):
        self.llm.load_state_dict(torch.load(llm_model, map_location=self.device))
        self.llm.to(self.device).eval()
        self.flow.load_state_dict(torch.load(flow_model, map_location=self.device))
        self.flow.to(self.device).eval()
        self.hift.load_state_dict(torch.load(hift_model, map_location=self.device))
        self.hift.to(self.device).eval()

    def inference(self, text, text_len, flow_embedding, llm_embedding=torch.zeros(0, 192),
                  prompt_text=torch.zeros(1, 0, dtype=torch.int32), prompt_text_len=torch.zeros(1, dtype=torch.int32),
                  llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32), llm_prompt_speech_token_len=torch.zeros(1, dtype=torch.int32),
                  flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32), flow_prompt_speech_token_len=torch.zeros(1, dtype=torch.int32),
                  prompt_speech_feat=torch.zeros(1, 0, 80), prompt_speech_feat_len=torch.zeros(1, dtype=torch.int32)):
        tts_speech_token = self.llm.inference(text=text.to(self.device),
                                              text_len=text_len.to(self.device),
                                              prompt_text=prompt_text.to(self.device),
                                              prompt_text_len=prompt_text_len.to(self.device),
                                              prompt_speech_token=llm_prompt_speech_token.to(self.device),
                                              prompt_speech_token_len=llm_prompt_speech_token_len.to(self.device),
                                              embedding=llm_embedding.to(self.device),
                                              beam_size=1,
                                              sampling=25,
                                              max_token_text_ratio=30,
                                              min_token_text_ratio=3)
        tts_mel = self.flow.inference(token=tts_speech_token,
                                      token_len=torch.tensor([tts_speech_token.size(1)], dtype=torch.int32).to(self.device),
                                      prompt_token=flow_prompt_speech_token.to(self.device),
                                      prompt_token_len=flow_prompt_speech_token_len.to(self.device),
                                      prompt_feat=prompt_speech_feat.to(self.device),
                                      prompt_feat_len=prompt_speech_feat_len.to(self.device),
                                      embedding=flow_embedding.to(self.device))
        tts_speech = self.hift.inference(mel=tts_mel).cpu()
        torch.cuda.empty_cache()
        return {'tts_speech': tts_speech}
     
###CosyVoice
class CustomCosyVoice:

    def __init__(self, model_dir):
        #assert os.path.exists(model_dir), f"model path '{model_dir}' not exist, please check the path: pretrained_models/CosyVoice-300M-zhtw"
        instruct = False
        
        if not os.path.exists(model_dir):
            model_dir = snapshot_download(model_dir)
        print("model", model_dir)
        self.model_dir = model_dir
        
        with open('{}/cosyvoice.yaml'.format(model_dir), 'r') as f:
            configs = load_hyperpyyaml(f)
        self.frontend = CustomCosyVoiceFrontEnd(configs['get_tokenizer'],
                                          configs['feat_extractor'],
                                          model_dir,
                                          '{}/campplus.onnx'.format(model_dir),
                                          '{}/speech_tokenizer_v1.onnx'.format(model_dir),
                                          '{}/spk2info.pt'.format(model_dir),
                                          instruct,
                                          configs['allowed_special'])
        self.model = CosyVoiceModel(configs['llm'], configs['flow'], configs['hift'])
        self.model.load('{}/llm.pt'.format(model_dir),
                        '{}/flow.pt'.format(model_dir),
                        '{}/hift.pt'.format(model_dir))
        del configs

    def list_avaliable_spks(self):
        spks = list(self.frontend.spk2info.keys())
        return spks

    def inference_sft(self, tts_text, spk_id):
        tts_speeches = []
        for i in self.frontend.text_normalize(tts_text, split=True):
            model_input = self.frontend.frontend_sft(i, spk_id)
            model_output = self.model.inference(**model_input)
            tts_speeches.append(model_output['tts_speech'])
        return {'tts_speech': torch.concat(tts_speeches, dim=1)}

    def inference_zero_shot(self, tts_text, prompt_text, prompt_speech_16k):
        prompt_text = self.frontend.text_normalize(prompt_text, split=False)
        tts_speeches = []
        for i in self.frontend.text_normalize(tts_text, split=True):
            model_input = self.frontend.frontend_zero_shot(i, prompt_text, prompt_speech_16k)
            model_output = self.model.inference(**model_input)
            tts_speeches.append(model_output['tts_speech'])
        return {'tts_speech': torch.concat(tts_speeches, dim=1)}
        
    def inference_zero_shot_no_normalize(self, tts_text, prompt_text, prompt_speech_16k):
        prompt_text = prompt_text
        tts_speeches = []
        for i in re.split(r'(?<=[？！，。.?!])\s*', tts_text):
            model_input = self.frontend.frontend_zero_shot(i, prompt_text, prompt_speech_16k)
            model_output = self.model.inference(**model_input)
            tts_speeches.append(model_output['tts_speech'])
        return {'tts_speech': torch.concat(tts_speeches, dim=1)}
        
####wav2text
def transcribe_audio(audio_file):
    model = whisper.load_model("base")
    result = model.transcribe(audio_file)
    converter = opencc.OpenCC('s2t')
    traditional_text = converter.convert(result["text"])
    return traditional_text

def get_bopomofo_rare(text, converter):
    res = converter(text)
    text_w_bopomofo = [x for x in zip(list(text), res[0])]
    reconstructed_text = ""
    
    for t in text_w_bopomofo:
        #print(t[0], word_to_dataset_frequency[t[0]], t[1])
        
        if word_to_dataset_frequency[t[0]] < 500 and t[1] != None:
            # Add the char and the pronunciation
            reconstructed_text += t[0] + f"[:{t[1]}]"
        
        elif len(char2phn[t[0]]) >= 2:
            if t[1] != char2phn[t[0]][0] and word_to_dataset_frequency[t[0]] < 10000:  # Not most common pronunciation
                # Add the char and the pronunciation
                reconstructed_text += t[0] + f"[:{t[1]}]"
            else:
                reconstructed_text += t[0]
            #print("DEBUG, multiphone char", t[0], char2phn[t[0]])
        else:
            # Add only the char
            reconstructed_text += t[0]
    
    #print("Reconstructed:", reconstructed_text)
    return reconstructed_text

def main():
    ####args
    parser = argparse.ArgumentParser(description="Run BreezyVoice text-to-speech with custom inputs")
    parser.add_argument("--content_to_synthesize", type=str, required=True, help="Specifies the content that will be synthesized into speech.")
    parser.add_argument("--speaker_prompt_audio_path", type=str, required=True, help="Specifies the path to the prompt speech audio file of the speaker.")
    parser.add_argument("--speaker_prompt_text_transcription", type=str, required=False, help="Specifies the transcription of the speaker prompt audio (Highly Recommended, if not provided, the system will fall back to transcribing with Whisper.)")
    
    parser.add_argument("--output_path", type=str, required=False, default="results/output.wav", help="Specifies the name and path for the output .wav file.")
    
    parser.add_argument("--model_path", type=str, required=False, default = "Splend1dchan/BreezyVoice-300M",help="Specifies the model used for speech synthesis.")
    args = parser.parse_args()
    
    
    cosyvoice = CustomCosyVoice(args.model_path)
    prompt_speech_16k = load_wav(args.speaker_prompt_audio_path, 16000)
    content_to_synthesize = args.content_to_synthesize
    output_path = args.output_path.strip()

    if args.speaker_prompt_text_transcription:
        speaker_prompt_text_transcription = args.speaker_prompt_text_transcription
    else:
        speaker_prompt_text_transcription = transcribe_audio(args.speaker_prompt_audio_path)
    
    bopomofo_converter = G2PWConverter()
    
    ###normalization
    speaker_prompt_text_transcription = cosyvoice.frontend.text_normalize_new(
        speaker_prompt_text_transcription, 
        split=False
    )
    content_to_synthesize = cosyvoice.frontend.text_normalize_new(
        content_to_synthesize, 
        split=False
    )
    speaker_prompt_text_transcription_bopomo = get_bopomofo_rare(speaker_prompt_text_transcription, bopomofo_converter)
    print("Speaker prompt audio transcription:",speaker_prompt_text_transcription_bopomo)

    content_to_synthesize_bopomo = get_bopomofo_rare(content_to_synthesize, bopomofo_converter)
    print("Content to be synthesized:",content_to_synthesize_bopomo)
    start = time.time()
    output = cosyvoice.inference_zero_shot_no_normalize(content_to_synthesize_bopomo, speaker_prompt_text_transcription_bopomo, prompt_speech_16k)
    end = time.time()
    print("Elapsed time:",end - start)
    print("Generated audio length:", output['tts_speech'].shape[1]/22050, "seconds")
    torchaudio.save(output_path, output['tts_speech'], 22050)
    print(f"Generated voice saved to {output_path}")

if __name__ == "__main__":
    main()
