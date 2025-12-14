import librosa
import resampy
import soundfile
import loguru

import torch
import numpy

from modules.nsf_hifigan.models import load_model
from modules.rmvpe.inference import RMVPE
from modules.shifter.mel_extractor import MelExtractor
from modules.shifter.utils import *

class Shift:
    def __init__(self, nsf_hifigan: str, pitch_extractor: str = None, device: str = "cuda", sample_rate: int = 44100):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.sample_rate = sample_rate

        loguru.logger.info(f"Using device: {self.device}")

        loguru.logger.info("Loading NSF HiFiGAN...")
        self.generator, self.config = load_model(nsf_hifigan, device=str(self.device))
        self.hop_length = self.config.hop_size

        self.mel_extractor = MelExtractor(sample_rate, self.config.n_fft, self.config.win_size, self.config.hop_size, self.config.fmin, self.config.fmax, self.config.num_mels,)

        loguru.logger.info("Loading pitch extractor...")
        self.rmvpe = None

        if pitch_extractor:
            self.rmvpe = RMVPE(pitch_extractor, hop_length=160)
            self.rmvpe.model = self.rmvpe.model.to(self.device)
            loguru.logger.info("Using pitch extractor: RMVPE")
        else:
            loguru.logger.warning("No pitch extractor provided - pitch extraction will fail!")

    def process_audio(self, audio: numpy.ndarray, key_shift: float,) -> numpy.ndarray:
        max_amplitude = numpy.max(numpy.abs(audio))
        if max_amplitude > 1.0:
            audio = audio / max_amplitude
        
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)

        mel_spectrogram = self.mel_extractor(audio_tensor)
        mel_length = mel_spectrogram.shape[-1]

        if self.sample_rate != 16000:
            audio_16k = resampy.resample(audio, self.sample_rate, 16000)
        else:
            audio_16k = audio
        
        f0 = self.rmvpe.infer_from_audio(audio_16k, 16000, self.device, 0.03)
        uv = f0 == 0

        if len(f0[~uv]) > 0:
            f0_continuous = f0.copy()
            f0_continuous[uv] = numpy.interp(numpy.where(uv)[0], numpy.where(~uv)[0], f0[~uv])
        else:
            f0_continuous = f0
        
        original_time = 0.01 * numpy.arange(len(f0))
        target_time = (numpy.arange(mel_length) * self.hop_length) / self.sample_rate
        
        f0_interpolated = numpy.interp(target_time, original_time, f0_continuous)
        uv_interpolated = numpy.interp(target_time, original_time, uv.astype(float)) > 0.5

        pitch_factor = 2 ** (key_shift / 12)
        f0_shifted = f0_interpolated * pitch_factor
        f0_shifted[uv_interpolated] = 0
        
        f0_tensor = torch.from_numpy(f0_shifted).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            output_audio = self.generator(mel_spectrogram, f0_tensor)
        
        output_audio = output_audio.squeeze().cpu().numpy()
        output_audio = output_audio / (numpy.max(numpy.abs(output_audio)) + 1e-5) * 0.95
        
        return output_audio
    
    def process_file(self, input_path: str, output_path: str, key_shift: float) -> numpy.ndarray:
        loguru.logger.info(f"Processing: {input_path}")
        loguru.logger.info(f"Pitch shifting: {key_shift:+.1f} semitones ()")

        audio, _ = librosa.load(input_path, sr=self.sample_rate, mono=True)

        if self.sample_rate != 16000:
            audio_16k = resampy.resample(audio, self.sample_rate, 16000)
        else:
            audio_16k = audio
        
        f0 = self.rmvpe.infer_from_audio(audio_16k, 16000, self.device, 0.03)
        uv = f0 == 0
        
        if len(f0[~uv]) > 0:
            f0_min, f0_max = f0[~uv].min(), f0[~uv].max()
            f0_mean = numpy.mean(f0[~uv])
            
            loguru.logger.info(f"F0 range: {f0_shift(f0_min, f0_max, key_shift)}")
            loguru.logger.info(f"F0 mean: {format_hz(f0_mean)} → {format_hz(f0_mean * (2 ** (key_shift / 12)))}")

        output_audio = self.process_audio(audio, key_shift)

        loguru.logger.info(f"Saving output: {output_path}")
        soundfile.write(output_path, output_audio, self.sample_rate)
        loguru.logger.success("Process completed successfully!")

        return output_audio
    
    def process_file_silent(self, input_path: str, output_path: str, key_shift: float) -> numpy.ndarray:
        audio, _ = librosa.load(input_path, sr=self.sample_rate, mono=True)
        output_audio = self.process_audio(audio, key_shift)
        soundfile.write(output_path, output_audio, self.sample_rate)
        
        return output_audio
    
    @staticmethod
    def shift_audio(input: str, output: str, key_shift: float, nsf_hifigan: str, pitch_extractor: str = None, device: str = "cuda", sample_rate: int = 44100) -> numpy.ndarray:
        shifter = Shift(nsf_hifigan, pitch_extractor, device, sample_rate)

        return shifter.process_file(input, output, key_shift)
