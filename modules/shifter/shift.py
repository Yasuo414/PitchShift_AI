import librosa
import resampy
import loguru
import soundfile

import torch
import numpy

from modules.nsf_hifigan.models import load_model
from modules.rmvpe.inference import RMVPE
from modules.shifter.mel_extractor import MelExtractor

class Shift:
    def shift_audio(input: str, output: str, key_shift: float, nsf_hifigan: str, pitch_extractor: str = None, device: str = "cuda", sample_rate: int = 44100,):
        device = torch.device(device if torch.cuda.is_available() else "cpu")
        loguru.logger.info(f"Using device: {device}")

        loguru.logger.info("Loading NSF HiFiGAN...")
        generator, h = load_model(nsf_hifigan, device=str(device))
        hop_length = h.hop_size

        loguru.logger.info("Loading pitch extractor...")
        if pitch_extractor:
            rmvpe = RMVPE(pitch_extractor, hop_length=160)
            rmvpe.model = rmvpe.model.to(device)
            loguru.logger.info("Using pitch extractor: RMVPE")
        else:
            loguru.logger.info("Using pitch extractor: Need to be implemented")
        
        loguru.logger.info(f"Loading audio: {input}")
        audio, _ = librosa.load(input, sr=sample_rate, mono=True)

        max_amplitude = numpy.max(numpy.abs(audio))
        if max_amplitude > 1.0:
            audio = audio / max_amplitude
        
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(device)

        loguru.logger.info("Extracting mel-spectrogram...")
        mel_extractor = MelExtractor(sample_rate, h.n_fft, h.win_size, hop_length, h.fmin, h.fmax, h.num_mels,)

        mel = mel_extractor(audio_tensor)
        mel_length = mel.shape[-1]
        loguru.logger.info(f" - Mel-Spectrogram shape: {mel.shape}")

        loguru.logger.info(f"Extracting pitch...")
        if sample_rate != 16000:
            audio_16k = resampy.resample(audio, sample_rate, 16000)
        else:
            audio_16k = audio
        
        f0 = rmvpe.infer_from_audio(audio_16k, 16000, device, 0.03)
        uv = f0 == 0

        if len(f0[~uv]) > 0:
            f0_continuous = f0.copy()
            f0_continuous[uv] = numpy.interp(numpy.where(uv)[0], numpy.where(~uv)[0], f0[~uv])
        else:
            f0_continuous = f0
        
        original_time = 0.01 * numpy.arange(len(f0))
        target_time = (numpy.arange(mel_length) * hop_length) / sample_rate
        
        f0_interpolated = numpy.interp(target_time, original_time, f0_continuous)
        uv_interpolated = numpy.interp(target_time, original_time, uv.astype(float)) > 0.5

        pitch_factor = 2 ** (key_shift / 12)
        f0_shifted = f0_interpolated * pitch_factor

        f0_shifted[uv_interpolated] = 0
        f0_tensor = torch.from_numpy(f0_shifted).float().unsqueeze(0).to(device)

        loguru.logger.info(f" - Pitch shift: {key_shift:+.1f} semitones (factor: {pitch_factor:.4f})")
        loguru.logger.info(f" - Original F0 range: {f0[~uv].min():.1f} - {f0[~uv].max():.1f} Hz" if len(f0[~uv]) > 0 else "  No voiced frames")
        loguru.logger.info(f" - Shifted F0 range: {f0_shifted[~uv_interpolated].min():.1f} - {f0_shifted[~uv_interpolated].max():.1f} Hz" if len(f0_shifted[~uv_interpolated]) > 0 else "  No voiced frames")
        loguru.logger.info(f" - F0 shape: {f0_tensor.shape}")

        loguru.logger.info("Generating audio...")
        with torch.no_grad():
            output_audio = generator(mel, f0_tensor)
        
        output_audio = output_audio.squeeze().cpu().numpy()
        output_audio = output_audio / (numpy.max(numpy.abs(output_audio)) + 1e-5) * 0.95

        loguru.logger.info(f"Saving output: {output}")
        soundfile.write(output, output_audio, sample_rate)
        loguru.logger.success("Process completed successfully!")

        return output_audio