import librosa.filters

import torch

class MelExtractor:
    def __init__(self, sample_rate: int = 44100, num_fft: int = 2048, win_length: int = 2048, hop_length: int = 512, fmin: int = 40, fmax: int = 16000, num_mels: int = 128, center: bool = False,):
        self.sample_rate = sample_rate
        self.num_fft = num_fft
        self.win_size = win_length
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.num_mels = num_mels
        self.center = center
        self.mel_basis = {}
        self.hann_window = {}
    
    def __call__(self, audio):
        mel_basis_key = f"{self.fmax}_{audio.device}"

        if mel_basis_key not in self.mel_basis:
            mel = librosa.filters.mel(sr=self.sample_rate, n_fft=self.num_fft, n_mels=self.num_mels, fmin=self.fmin, fmax=self.fmax,)
            self.mel_basis[mel_basis_key] = torch.from_numpy(mel).float().to(audio.device)
        
        hann_window_key = f"{audio.device}"
        if hann_window_key not in self.hann_window:
            self.hann_window[hann_window_key] = torch.hann_window(self.win_size, device=audio.device)
        
        padding_amount = int((self.win_size - self.hop_length) / 2)
        audio = torch.nn.functional.pad(audio.unsqueeze(1), (padding_amount, padding_amount), mode="reflect",)
        audio = audio.squeeze(1)

        spectrogram = torch.stft(audio, self.num_fft, self.hop_length, self.win_size, self.hann_window[hann_window_key], self.center, "reflect", False, True, True,)
        spectrogram = torch.abs(spectrogram)
        spectrogram = torch.matmul(self.mel_basis[mel_basis_key], spectrogram)
        spectrogram = torch.log(torch.clamp(spectrogram, min=1e-5))

        return spectrogram