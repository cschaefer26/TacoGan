import librosa
import numpy as np

from utils.config import Config


class Audio:

    def __init__(self, cfg: Config):
        self.n_mels = cfg.n_mels
        self.sample_rate = cfg.sample_rate
        self.hop_length = cfg.hop_length
        self.win_length = cfg.win_length
        self.n_fft = cfg.n_fft
        self.fmin = cfg.fmin
        self.min_db = cfg.min_db
        self.ref_db = cfg.ref_db

    def load_wav(self, path):
        wav, _ = librosa.load(path, sr=self.sample_rate)
        return wav

    def save_wav(self, wav, path):
        wav = wav.astype(np.float32)
        librosa.output.write_wav(path, wav, sr=self.sample_rate)

    def wav_to_mel(self, y):
        spec = librosa.stft(
            y=y,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length)
        spec = np.abs(spec)
        mel = librosa.feature.melspectrogram(
            S=spec,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.fmin)
        mel = mel.T
        mel = self._compress(mel)
        return self._normalize(mel)

    def griffinlim(self, mel, n_iter=32):
        mel = mel.T
        denormalized = self._denormalize(mel)
        amp_mel = self._decompress(denormalized)
        S = librosa.feature.inverse.mel_to_stft(
            amp_mel,
            power=1,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            fmin=self.fmin)
        wav = librosa.core.griffinlim(
            S,
            n_iter=n_iter,
            hop_length=self.hop_length,
            win_length=self.win_length)
        return wav

    def _normalize(self, mel):
        mel = (mel - self.min_db) / -self.min_db
        mel =  np.clip(mel, 0, 1)
        return mel * 2. - 1.

    def _denormalize(self, mel):
        mel = (mel + 1.) / 2.
        return np.clip(mel, 0, 1) * -self.min_db + self.min_db

    def _compress(self, mel):
        mel = np.maximum(1e-5, mel)
        return self.ref_db * np.log10(mel)

    def _decompress(self, mel):
        return np.power(10.0, mel / self.ref_db)


if __name__ == '__main__':
    cfg = Config.load('config.yaml')
    audio = Audio(cfg)
    wav = audio.load_wav('/Users/cschaefe/datasets/LJSpeech/LJSpeech-1.1/wavs/LJ050-0278.wav')
    mel = audio.wav_to_mel(wav)
    wav = audio.griffinlim(mel)
    audio.save_wav(wav, '/tmp/testwav.wav')