import torch
import time
import concurrent.futures

from nltk import download
from nltk.tokenize import sent_tokenize
from pydub import AudioSegment
from pydub.playback import play
from glados.utils.cleaners import Cleaner
from glados.utils.tokenizer import Tokenizer

class TextToSpeech:
        
    __SAMPLE_RATE=22050
    __WAVE_16_BIT_MAX_AMPLITUDE=2**15

    def __get_best_torch_device(self):
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.is_vulkan_available():
            return 'vulkan'
        else:
            return 'cpu'

    def __init__(self, emb_file: str='models/emb/glados_p2.pt', 
                 glados_file: str='models/glados-new.pt', 
                 vocoder_file: str='models/vocoder-gpu.pt',
                 corrections: dict=None,
                 log: bool=False):
        self.log = log
        self.device = self.__get_best_torch_device()
        self.emb = torch.load(emb_file)
        self.glados = torch.jit.load(glados_file)
        self.vocoder = torch.jit.load(vocoder_file, map_location=self.device)
        self.executor = concurrent.futures.ThreadPoolExecutor()
        self.last_line_future = None
        self.cleaner = Cleaner(cleaner_name='english_cleaners', use_phonemes=True,
                               lang='en-us', user_corrections=corrections)
        for i in range(2):
            init = self.glados.generate_jit(self.__prepare_text(str(i)), self.emb, 1.0)
            init_mel = init['mel_post'].to(self.device)
            self.vocoder(init_mel)    
        download('punkt',quiet=self.log)

    def speak(self, text: str, delay: float=0.1, filename: str=None):
        sentences = sent_tokenize(text)
        pause = AudioSegment.silent(duration=delay, frame_rate=self.__SAMPLE_RATE)
        recording = AudioSegment.empty()

        if self.log:
            print("Speaking: ")
            print(sentences)

        sentence_audio_futures = []
        for sentence in sentences:
            sentence_audio_futures += [self.executor.submit(self.__run_tts, text=sentence)]

        for sentence_audio_future in sentence_audio_futures:
            sentence_audio = sentence_audio_future.result()
            if len(sentence_audio) > 0:
                play(sentence_audio)
            play(pause)
            recording = recording + sentence_audio + pause

        if not filename is None and len(filename) > 0:
            recording.export(filename +".wav", format="wav")

    def line(self, line: str, delay: float=0.1):
        pause = AudioSegment.silent(duration=delay, frame_rate=self.__SAMPLE_RATE)
        audio = self.__run_tts(line)
        
        self.__wait_for_previous_line_to_complete()
        self.last_line_future = self.executor.submit(self.__play_audio, audio, pause)

    def __play_audio(self, audio: AudioSegment, pause: AudioSegment):
        if len(audio) > 0:
            play(audio)
        play(pause)

    def __wait_for_previous_line_to_complete(self):
        if not self.last_line_future is None:
            self.last_line_future.result()
            self.last_line_future = None

    def __run_tts(self, text, alpha: float=1.0) -> AudioSegment:
        if self.log:
            print("Generating: " + text)
        x = self.__prepare_text(text)

        if x is None:
            return AudioSegment.empty()
        
        with torch.no_grad():
            # Generate generic TTS-output
            old_time = time.time()
            tts_output = self.glados.generate_jit(x, self.emb, alpha)
            if self.log:
                print("Forward Tacotron took " + str((time.time() - old_time) * 1000) + "ms")

            # Use HiFiGAN as vocoder to make output sound like GLaDOS
            old_time = time.time()
            mel = tts_output['mel_post'].to(self.device)
            audio = self.vocoder(mel)
            if self.log:
                print("HiFiGAN took " + str((time.time() - old_time) * 1000) + "ms")

            # Normalize audio to fit in wav-file
            audio = audio.squeeze()
            audio = audio * self.__WAVE_16_BIT_MAX_AMPLITUDE
            audio = audio.cpu().numpy().astype('int16')

            return AudioSegment(data=audio.tobytes(), sample_width=2, frame_rate=self.__SAMPLE_RATE, channels=1)

    def __prepare_text(self, text: str)->str:
        if not (text[-1] in ".?!"):
            text = text + '.'
        
        if len(text) <= 1:
            # Only punctuation can cause the model to explode
            return None
        
        tokenizer = Tokenizer()
        return torch.as_tensor(tokenizer(self.cleaner(text)), dtype=torch.long, device=self.device).unsqueeze(0)
