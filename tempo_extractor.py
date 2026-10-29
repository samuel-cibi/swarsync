"""
tempo_extractor.py
-------------------
DSP project: extract tempo (BPM), beat time-instances and the time signature
of a song.

Pipeline (classic beat-tracking DSP chain):
  1. Load audio, resample/mono-mix, normalize.
  2. Compute a spectral onset-strength envelope (novelty function):
       - STFT -> mel spectrogram -> log compression -> first-order
         difference across time (spectral flux). Peaks in this curve
         correspond to note/drum onsets.
  3. Estimate global tempo via autocorrelation / Fourier tempogram of
     the onset envelope (finds the periodicity that best explains the
     onsets).
  4. Run dynamic-programming beat tracking (Ellis, 2007) that picks a
     sequence of beat instants which are (a) close to onset peaks and
     (b) spaced consistently with the estimated tempo.
  5. Also compute a local/time-varying tempo curve using a tempogram,
     since real songs often drift in tempo.
  6. Estimate the time signature (meter) with an open-ended search:
       - find the pulse (tatum): the fastest regular grid the notes sit on;
       - describe every pulse by accent features (band-wise onset strength,
         harmonic/bass/timbre change, loudness);
       - test EVERY bar length from 2 up to --max-bar-pulses pulses for how
         well "position inside the bar" explains those features;
       - split the winning bar into beat groups of 2s and 3s (any combination,
         e.g. 2+2+3, 3+3+2+2), then name it in standard notation.
     Nothing is restricted to a list like {2/4, 3/4, 4/4, 6/8}: 5/4, 7/8 (2+2+3),
     11/8, 13/8, 15/16 ... come out of the same search.

Usage:
    python tempo_extractor.py --input song.mp3
    python tempo_extractor.py --url "https://example.com/song.mp3"
    python tempo_extractor.py --input song.mp3 --plot --csv beats.csv
    python tempo_extractor.py --input song.mp3 --downbeats-csv bars.csv
"""

import argparse
import os
import sys
import tempfile
import urllib.parse
from dataclasses import dataclass, field

import numpy as np
import librosa
import requests
import scipy.ndimage as ndi
import scipy.signal as sps
from scipy.fft import dct


# --------------------------------------------------------------------------
# 1. Audio acquisition
# --------------------------------------------------------------------------
def download_audio(url: str, dest_dir: str) -> str:

    parsed = urllib.parse.urlparse(url)
    filename = os.path.basename(parsed.path) or "downloaded_audio"
    if "." not in filename:
        filename += ".mp3"
    dest_path = os.path.join(dest_dir, filename)

    print(f"[download] fetching {url}")
    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()
    content_type = resp.headers.get("Content-Type", "")
    if "audio" not in content_type and not filename.lower().endswith(
        (".mp3", ".wav", ".flac", ".ogg", ".m4a")
    ):
        print(
            f"[warning] Content-Type='{content_type}' doesn't look like audio. "
            "If this is a YouTube/streaming page (not a direct file link), "
            "download the audio separately first (e.g. with yt-dlp) and "
            "pass it via --input instead."
        )
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"[download] saved to {dest_path}")
    return dest_path


def load_audio(path: str, sr: int = 48000): #22050
    """Load audio as mono float waveform at sample rate sr."""
    y, sr = librosa.load(path, sr=sr, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    print(f"[load] duration={duration:.2f}s, sample_rate={sr}Hz, samples={len(y)}")
    return y, sr


# --------------------------------------------------------------------------
# 2-4. Onset envelope, global tempo, beat instants
# --------------------------------------------------------------------------
def analyze_beats(y: np.ndarray, sr: int, hop_length: int = 512):
    """
    Returns:
        tempo_bpm      : float, global tempo estimate
        beat_times     : np.ndarray of beat instants in seconds
        onset_env      : onset strength envelope (for plotting)
        onset_times    : time axis for onset_env
    """
    # Spectral-flux onset strength envelope (the "novelty function")
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset_times = librosa.frames_to_time(
        np.arange(len(onset_env)), sr=sr, hop_length=hop_length
    )

    # Dynamic-programming beat tracker: returns tempo + beat frame indices
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, hop_length=hop_length, trim=False
    )
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)

    tempo_bpm = float(np.atleast_1d(tempo)[0])
    return tempo_bpm, beat_times, onset_env, onset_times


def time_varying_tempo(onset_env: np.ndarray, sr: int, hop_length: int = 512):
    """
    Local tempo curve using a tempogram (autocorrelation of the onset
    envelope in short windows). Useful to see tempo drift over the song,
    unlike the single global BPM number.
    """
    tempogram = librosa.feature.tempogram(
        onset_envelope=onset_env, sr=sr, hop_length=hop_length
    )
    tempo_curve = librosa.feature.tempo(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        aggregate=None,  # per-frame tempo instead of one global value
    )
    times = librosa.frames_to_time(
        np.arange(len(tempo_curve)), sr=sr, hop_length=hop_length
    )
    return times, tempo_curve, tempogram


# --------------------------------------------------------------------------
# 5. Output helpers
# --------------------------------------------------------------------------
def save_csv(beat_times: np.ndarray, path: str):
    import csv

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["beat_index", "time_seconds"])
        for i, t in enumerate(beat_times):
            writer.writerow([i, f"{t:.4f}"])
    print(f"[csv] wrote {len(beat_times)} beat times to {path}")


def save_downbeats_csv(downbeat_times: np.ndarray, path: str):
    import csv

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bar_index", "downbeat_time_seconds"])
        for i, t in enumerate(downbeat_times):
            writer.writerow([i, f"{t:.4f}"])
    print(f"[csv] wrote {len(downbeat_times)} bar (downbeat) times to {path}")


def make_plot(y, sr, onset_env, onset_times, beat_times, tempo_bpm, out_path,
              downbeat_times=None, time_sig=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    # Waveform + beat markers
    librosa.display.waveshow(y, sr=sr, ax=axes[0], alpha=0.6)
    axes[0].vlines(beat_times, -1, 1, color="r", linestyle="--", alpha=0.8, label="beats")
    if downbeat_times is not None and len(downbeat_times):
        axes[0].vlines(downbeat_times, -1, 1, color="k", linewidth=2, alpha=0.9, label="downbeats")
    ts = f", time signature ≈ {time_sig}" if time_sig is not None else ""
    axes[0].set_title(f"Waveform with detected beats  (tempo ≈ {tempo_bpm:.1f} BPM{ts})")
    axes[0].legend(loc="upper right")

    # Onset strength envelope + beat markers
    axes[1].plot(onset_times, onset_env, label="onset strength", color="tab:blue")
    axes[1].vlines(
        beat_times,
        0,
        onset_env.max() if len(onset_env) else 1,
        color="r",
        linestyle="--",
        alpha=0.8,
    )
    axes[1].set_title("Onset strength envelope (novelty function)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"[plot] saved figure to {out_path}")


# --------------------------------------------------------------------------
# 6. Time signature (meter) estimation
# --------------------------------------------------------------------------
#
# A time signature is a statement about *grouping*: how many pulses make a bar,
# and how those pulses cluster into felt beats.  Audio has no bar lines, so the
# estimate is built bottom-up:
#
#   a) Pulse / tatum.  The averaged Fourier tempogram of the onset envelope has
#      peaks at every periodicity in the music.  The strongest one is usually
#      the pulse; we step down if it is only a harmonic of a slower level (no
#      onsets at its in-between points) and step up if the pulse has genuine
#      triplet or (for slow pulses) duple subdivisions.  The pulse grid is then
#      tracked with the same Ellis DP used for beats.  Additive meters (2+2+3)
#      have no isochronous beat, but they always have an isochronous pulse,
#      which is why the analysis happens at this level.
#
#   b) Accent features per pulse: onset strength (full band and 4 sub-bands:
#      kick / low-mid / snare / hats), harmonic change (chroma), bass-note
#      change, timbre change (MFCC) and loudness.  Downbeats tend to coincide
#      with chord / bass changes and kick accents; beat starts with onsets.
#
#   c) Bar length.  For every candidate M = 2 .. max_bar_pulses we fit the
#      model  feature[i] = mean_{i mod M} + noise  and measure the adjusted R^2
#      (windowed, so tempo-tracking slips only affect one window).  Multiples of
#      the true bar also fit (2 bars, 4-bar phrases), so each M is scored by
#      its *gain* over its own divisors, weighted by a broad, meter-agnostic
#      prior on bar duration (log-normal around ~2 s).  Every integer M is a
#      candidate: 7, 11, 13, 15 ... are tested exactly like 3 and 4.
#
#   d) Downbeat phase: the bar position with the strongest downbeat cues.
#
#   e) Beat grouping: the bar's accent profile is split into groups of 2s and
#      3s by an exact dynamic program over all compositions (2+2+3, 3+2+2,
#      3+3+2+2, ...).  Regular groupings are preferred when the whole song
#      repeats at that sub-period or the beat tracker's tactus tiles the bar;
#      an uneven grouping must win by a clear margin.
#
#   f) Notation: groups of 2 pulses -> n/4 (simple), groups of 3 -> 3n/8
#      (compound), mixed 2s/3s -> M/8 with the grouping (additive),
#      16th-note pulses -> /16.  Note that the denominator is a notational
#      convention: 3/8 vs 3/4, or 4/4 vs 2/2, sound identical at the right
#      tempo, so the denominator is chosen from the pulse/beat tempo.
#
_FEAT_SR = 22050          # internal analysis rate for the accent features
_FEAT_HOP = 512
_CHANGE_CONTEXT_BEATS = 2     # length of the "before" window for change cues, in beats
_FEATURE_NAMES = ["onset", "flux_low", "flux_lowmid", "flux_mid", "flux_high",
                  "harm_change", "bass_change", "timbre_change", "loudness"]
# downbeat cues: kick / low-end onsets, chord and bass changes, timbre novelty,
# loudness.  (Snare and hi-hat bands mark backbeats and every pulse, so they
# are not used to locate bar starts.)
_DOWNBEAT_W = np.array([0.5, 1.0, 0.25, 0.0, 0.0, 2.0, 2.0, 1.5, 1.0])
_BEAT_W = np.array([1.5, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.25, 0.75])


@dataclass
class TimeSignature:
    numerator: int
    denominator: int
    grouping: tuple            # beat groups, in pulses (e.g. (2, 2, 3) for 7/8)
    kind: str                  # simple | compound | additive | irregular simple | irregular compound
    beats_per_bar: int         # number of felt beats in a bar
    beat_bpm: float            # tempo of the felt beat
    pulse_bpm: float           # tempo of the pulse (tatum)
    bar_pulses: int            # bar length in pulses
    bar_seconds: float
    confidence: float          # 0..1
    downbeat_times: np.ndarray = field(repr=False)
    alternatives: list = field(default_factory=list)   # [(label, probability), ...]

    @property
    def label(self) -> str:
        return f"{self.numerator}/{self.denominator}"

    @property
    def grouping_str(self) -> str:
        return "+".join(str(g) for g in self.grouping)

    def __str__(self) -> str:
        s = self.label
        if self.kind == "additive" or (len(set(self.grouping)) > 1):
            s += f" ({self.grouping_str})"
        return s


# ---- a) pulse ------------------------------------------------------------
def _periodicity_spectrum(onset_env, sr, hop_length, win_sec=8.0):
    """
    Time-averaged Fourier tempogram (Welch-style: |FFT| of overlapping 8 s
    windows of the onset envelope, averaged).  Peaks = periodicities in BPM.
    """
    win = int(round(win_sec * sr / hop_length))
    win = max(16, min(win, len(onset_env)))
    env = np.pad(onset_env, (win // 2, win // 2), mode="constant")
    frames = librosa.util.frame(env, frame_length=win, hop_length=max(1, win // 8))
    A = np.abs(np.fft.rfft(frames * np.hanning(win)[:, None], axis=0)).mean(axis=1)
    freqs = np.fft.rfftfreq(win, d=hop_length / sr) * 60.0       # cycles per minute
    return freqs, A


def _salience_curve(freqs, A, lo, hi):
    """Periodicity strength relative to the local spectral floor (+-25 %)."""
    m = (freqs >= lo) & (freqs <= hi)
    fr, a = freqs[m], A[m]
    sal = np.empty_like(a)
    for i in range(len(a)):
        ring = (fr > fr[i] / 1.25) & (fr < fr[i] * 1.25)
        sal[i] = a[i] / (np.median(a[ring]) + 1e-12)
    return fr, sal


def _sal_at(fr, sal, f0, tol=0.04):
    m = (fr > f0 * (1 - tol)) & (fr < f0 * (1 + tol))
    if not m.any():
        return 0.0, f0
    i = np.flatnonzero(m)[np.argmax(sal[m])]
    return float(sal[i]), float(fr[i])


def _track_grid(onset_env, sr, hop_length, bpm, tightness=30):
    """Ellis DP tracking at a fixed rate; low tightness lets it follow drift."""
    _, frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, hop_length=hop_length,
        bpm=bpm, tightness=tightness, trim=False)
    return np.asarray(frames, dtype=float)


def _phase_profile(onset_env, grid_frames, n_bins=48):
    """Average onset strength as a function of phase between consecutive grid points."""
    env = ndi.maximum_filter1d(onset_env, 3)
    g = np.asarray(grid_frames, float)
    d = np.diff(g)
    if len(d) < 4:
        return None
    med = np.median(d)
    x = np.arange(len(env))
    H, c = np.zeros(n_bins), 0
    for a, L in zip(g[:-1], d):
        if abs(L - med) > 0.25 * med:
            continue
        H += np.interp(a + np.arange(n_bins) / n_bins * L, x, env)
        c += 1
    return H / max(c, 1)


def _subdivision_strength(prof, k):
    """Onset energy at the k-ary subdivision points relative to the on-grid peak."""
    n = len(prof)
    floor = np.percentile(prof, 20)
    h = lambda f: prof[[(int(round(f * n)) + j) % n for j in (-1, 0, 1)]].max()
    on = max(h(0) - floor, 1e-9)
    return float(np.mean([h(j / k) - floor for j in range(1, k)]) / on)


def _estimate_pulse(onset_env, sr, hop_length, tactus_bpm,
                    min_bpm=40.0, max_bpm=1000.0, sub_thresh=0.2):
    """Returns (pulse_bpm, weak) where weak=True means no clear periodicity was found."""
    freqs, A = _periodicity_spectrum(onset_env, sr, hop_length)
    hi = min(max_bpm, freqs[-1] * 0.95)
    fr, sal = _salience_curve(freqs, A, lo=min_bpm * 0.8, hi=hi)
    pk, _ = sps.find_peaks(sal, height=2.0)
    pk = pk[(fr[pk] >= min_bpm) & (fr[pk] <= hi)]
    if len(pk) == 0:
        return float(tactus_bpm), True
    p = float(fr[pk[np.argmax(sal[pk])]])
    p_sal = float(sal[pk].max())
    frame_rate = sr / hop_length * 60.0          # frames per minute

    # walk down: is p only a harmonic of a slower periodicity?  Sharp, isolated
    # clicks (a metronome, a rim-shot) put strong energy at 2x, 3x, 4x the real
    # pulse rate, sometimes stronger than the pulse itself.  The arbiter is the
    # onset content at the would-be subdivision points: none at all -> harmonic.
    changed = True
    while changed:
        changed = False
        for k in (2, 3):
            s_c, c = _sal_at(fr, sal, p / k)
            if c < min_bpm or s_c < 2.0:
                continue
            prof = _phase_profile(onset_env, _track_grid(onset_env, sr, hop_length, c))
            if prof is None:
                continue
            strength = _subdivision_strength(prof, k)
            if strength < 0.05 or (s_c >= 0.7 * p_sal and strength < sub_thresh):
                p, p_sal, changed = c, max(s_c, 0.5 * p_sal), True
                break

    # walk up: genuine triplet subdivisions (or duple ones for slow pulses)?
    prof = _phase_profile(onset_env, _track_grid(onset_env, sr, hop_length, p))
    if prof is not None:
        r = {k: _subdivision_strength(prof, k) for k in (2, 3)}
        for k in (3, 2):
            fine = p * k
            if fine > max_bpm or frame_rate / fine < 4:      # needs >= 4 frames per pulse
                continue
            if r[k] >= sub_thresh and r[k] > r[5 - k] and (k == 3 or p < 200):
                p = fine
                break

    # the spectrum is coarsely sampled (~7 BPM bins); snap to an exact multiple
    # of the tracked tactus when it is one, so the pulse grid does not drift
    ratio = p / tactus_bpm
    for k in (1, 2, 3, 4, 6, 8):
        if abs(ratio - k) / k < 0.04:
            return k * tactus_bpm, False
        if abs(ratio - 1.0 / k) * k < 0.04:
            return tactus_bpm / k, False
    return p, False


def _pulse_grid(onset_env, sr, hop_length, pulse_bpm, beat_frames):
    """
    Two candidate grids: (a) DP-tracked at the pulse rate, (b) the tracked beats
    subdivided evenly (follows tempo changes, but wrong for additive meters).
    Keep the one whose points sit on stronger onsets.
    """
    env = onset_env / (ndi.maximum_filter1d(onset_env, 64) + 1e-9)
    env = ndi.maximum_filter1d(env, 3)
    cands = {"dp": _track_grid(onset_env, sr, hop_length, pulse_bpm)}
    bf = np.asarray(beat_frames, float)
    if len(bf) > 4:
        k = np.median(np.diff(bf)) / (60.0 / pulse_bpm * sr / hop_length)
        if round(k) >= 1 and abs(k - round(k)) < 0.12:
            k = int(round(k))
            sub = [b0 + j * (b1 - b0) / k for b0, b1 in zip(bf[:-1], bf[1:]) for j in range(k)]
            cands["subdiv"] = np.array(sub + [bf[-1]])
    score = {}
    for name, g in cands.items():
        gi = np.clip(np.round(g).astype(int), 0, len(env) - 1)
        score[name] = float(np.mean(env[gi])) if len(gi) else -1.0
    best = max(score, key=score.get)
    return np.round(cands[best]).astype(int)


# ---- b) accent features ----------------------------------------------------
def _window_means(Z, starts, ends):
    """Mean of Z[:, a:b] for many (a, b) at once via cumulative sums.  Z: (D, T)."""
    T = Z.shape[1]
    cs = np.concatenate([np.zeros((Z.shape[0], 1)), np.cumsum(Z, axis=1)], axis=1)
    a = np.clip(starts, 0, T)
    b = np.clip(ends, 0, T)
    cnt = np.maximum(b - a, 1)
    return ((cs[:, b] - cs[:, a]) / cnt).T          # (n, D)


def _pulse_features(y, sr, pulse_times, beat_sec):
    """Accent feature matrix (n_pulses, 9), computed at an internal 22.05 kHz."""
    y22 = librosa.resample(y, orig_sr=sr, target_sr=_FEAT_SR) if sr != _FEAT_SR else y
    n_fft = 2048
    S = np.abs(librosa.stft(y22, n_fft=n_fft, hop_length=_FEAT_HOP)).astype(np.float32)
    T = S.shape[1]
    fft_f = librosa.fft_frequencies(sr=_FEAT_SR, n_fft=n_fft)
    # cheap harmonic enhancement for the pitch features: a short median filter
    # along time removes drum transients (much faster than full HPSS)
    kb = int(np.searchsorted(fft_f, 5000.0))
    H = ndi.median_filter(S[:kb], size=(1, 9))

    mel = librosa.feature.melspectrogram(S=S ** 2, sr=_FEAT_SR, n_mels=96, fmax=11000)

    # Onset / loudness features on a finer grid (46 ms window, 11.6 ms hop).
    # They are measured as the *total* rise of log-energy around each pulse
    # (sum of positive spectral flux over a short window) and mean *linear*
    # power, both of which are independent of where an onset falls inside an
    # STFT frame.  (Taking the flux peak instead makes perfectly regular audio
    # such as a metronome show a fake periodicity from frame-phase aliasing.)
    hop1 = _FEAT_HOP // 2
    P1 = np.abs(librosa.stft(y22, n_fft=1024, hop_length=hop1)).astype(np.float32) ** 2
    T1 = P1.shape[1]
    mel1 = librosa.feature.melspectrogram(S=P1, sr=_FEAT_SR, n_fft=1024, n_mels=64, fmax=11000)
    logmel1 = librosa.power_to_db(mel1, ref=np.max, top_db=80)
    mel_f1 = librosa.mel_frequencies(n_mels=64, fmax=11000)
    flux = np.maximum(0, np.diff(logmel1, axis=1, prepend=logmel1[:, :1]))
    onset = flux.mean(axis=0)
    bands = [(0, 150), (150, 600), (600, 3000), (3000, 1e9)]
    band_flux = [flux[(mel_f1 >= lo) & (mel_f1 < hi)].mean(axis=0) for lo, hi in bands]
    power = P1.sum(axis=0)
    del mel1, logmel1, P1

    cfilt = librosa.filters.chroma(sr=_FEAT_SR, n_fft=n_fft)[:, :kb]
    H2 = H ** 2
    chroma = cfilt @ H2
    chroma /= chroma.max(axis=0, keepdims=True) + 1e-9
    bfilt = cfilt * (fft_f[:kb] <= 300)[None, :]
    bass_chroma = bfilt @ H2
    mel_floor = float(mel.max()) * 1e-8                   # 80 dB below the peak
    del S, H, H2

    pulse_times = np.asarray(pulse_times)
    pf = np.clip(np.round(pulse_times * _FEAT_SR / _FEAT_HOP).astype(int), 0, T - 1)
    pf1 = np.clip(np.round(pulse_times * _FEAT_SR / hop1).astype(int), 0, T1 - 1)
    n = len(pf)
    per1 = int(np.median(np.diff(pf1))) if n > 1 else 8
    a1 = min(5, max(2, int(0.4 * per1)))              # window around each pulse (fine frames)
    b1 = max(3, int(round(0.3 * per1)))
    L = max(2, int(round(beat_sec * _FEAT_SR / _FEAT_HOP)))
    # pulse times sit on onset-envelope peaks, which lag the true attack by a
    # frame or two: start the 'after' windows slightly earlier
    # (window offsets also keep each analysis window clear of the STFT smear of
    # neighbouring onsets, which would otherwise alias with the frame grid)
    d = int(round(0.07 * _FEAT_SR / _FEAT_HOP))
    d1 = int(round(0.045 * _FEAT_SR / hop1))

    def rise_at(x):
        return _window_means(x[None], pf1 - a1, pf1 + b1)[:, 0]

    def timbre_change():
        # MFCCs of the window-averaged *linear* mel spectrum: averaging power
        # before the log keeps short sounds from aliasing with the frame grid
        u = _window_means(mel, pf - d - _CHANGE_CONTEXT_BEATS * L, pf - d)
        v = _window_means(mel, pf - d, pf - d + L)
        cu = dct(np.log10(np.maximum(u, mel_floor)), axis=1, norm="ortho")[:, 1:14]
        cv = dct(np.log10(np.maximum(v, mel_floor)), axis=1, norm="ortho")[:, 1:14]
        scale = np.concatenate([cu, cv]).std(axis=0) + 1e-9
        return np.linalg.norm((cu - cv) / scale, axis=1)

    def change(Z, metric):
        # what starts at the pulse vs. what came before it.  The 'before' window
        # is two beats long so the change is credited to the pulse where the new
        # material begins, not also to the one after it (matters for short sounds
        # such as an accented metronome click).
        u = _window_means(Z, pf - d - _CHANGE_CONTEXT_BEATS * L, pf - d)
        v = _window_means(Z, pf - d, pf - d + L)
        cos = 1 - (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-9)
        if metric == "cos":
            return cos
        if metric == "bass":    # bass pitch-class change, weighted by bass energy
            return cos * np.sqrt(np.minimum(u.sum(1), v.sum(1)) + 1e-12)
        return np.linalg.norm(u - v, axis=1)

    feats = [rise_at(onset)] + [rise_at(b) for b in band_flux]
    feats += [change(chroma, "cos"), change(bass_chroma, "bass"), timbre_change()]
    loud = _window_means(power[None], pf1 - d1, pf1 - d1 + max(1, per1))[:, 0]
    feats.append(10 * np.log10(np.maximum(loud, loud.max() * 1e-6) + 1e-20))   # 60 dB floor
    return np.stack(feats, axis=1).astype(float)


# smallest difference per feature that is worth anything (log-ratio for the
# onset bands, cosine distance, MFCC distance, dB).  Without these floors a
# perfectly steady signal (metronome) has a near-zero spread, and inaudible
# 0.3 dB wobbles would be blown up into 'strong accents'.
_FEATURE_FLOORS = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.02, 0.0, 0.5, 1.0])


def _normalise_features(X, pulse_sec, detrend_sec=8.0):
    X = X.copy()
    for j in range(5):    # compress heavy-tailed onset features
        X[:, j] = np.log1p(X[:, j] / (np.median(X[:, j]) + 1e-9))
    floors = _FEATURE_FLOORS.copy()
    floors[5:8] = np.maximum(floors[5:8], 0.1 * np.median(np.abs(X[:, 5:8]), axis=0))
    k = max(3, int(round(detrend_sec / pulse_sec)) | 1)
    for j in range(X.shape[1]):
        X[:, j] -= ndi.median_filter(X[:, j], size=k, mode="nearest")   # remove section dynamics
        mad = np.median(np.abs(X[:, j] - np.median(X[:, j]))) * 1.4826
        X[:, j] /= max(mad, 0.25 * X[:, j].std(), floors[j]) + 1e-9
    return np.clip(X, -6, 6)


# ---- c) bar length -------------------------------------------------------
def _cyclic_adj_r2(X, M, win, hop):
    """Adjusted R^2 of feature ~ position-in-cycle (period M), averaged over windows."""
    n = len(X)
    out = []
    for s0 in (list(range(0, max(1, n - win + 1), hop)) or [0]):
        Z = X[s0:s0 + win]
        m = len(Z)
        if m < 2 * M + 2:
            continue
        pos = np.arange(m) % M
        sst = ((Z - Z.mean(0)) ** 2).sum(0)
        cnt = np.bincount(pos, minlength=M)[:, None]
        sums = np.zeros((M, Z.shape[1]))
        np.add.at(sums, pos, Z)
        ssw = ((Z - (sums / np.maximum(cnt, 1))[pos]) ** 2).sum(0)
        out.append(1 - (ssw / (m - M)) / (sst / (m - 1) + 1e-12))
    return np.mean(out, axis=0) if out else np.full(X.shape[1], np.nan)


def _bar_scores(X, pulse_sec, max_bar_pulses, window_sec=30.0):
    n = len(X)
    Mmax = int(min(max_bar_pulses, n // 4))
    win = int(min(n, max(window_sec / pulse_sec, 4 * Mmax)))
    hop = max(1, win // 2)
    Rf = np.zeros((Mmax + 1, X.shape[1]))
    for M in range(2, Mmax + 1):
        Rf[M] = _cyclic_adj_r2(X, M, win, hop)
    R = np.nanmean(Rf, axis=1)
    G = np.full(Mmax + 1, -np.inf)
    for M in range(2, Mmax + 1):
        G[M] = R[M] - max(R[d] for d in range(1, M) if M % d == 0)
    return R, G, win, hop, Rf


def _bar_candidates(R, G, pulse_sec, beat_pulses, min_sec, max_sec, mu=2.0, sigma=0.5):
    cands = []
    for M in range(2, len(R)):
        dur = M * pulse_sec
        if dur < min_sec or dur > max_sec or M <= beat_pulses:
            continue
        prior = np.exp(-np.log(dur / mu) ** 2 / (2 * sigma ** 2))
        cands.append((max(G[M], 0.0) * prior, M))
    cands.sort(reverse=True)
    return cands


def _promote_accent_cycle(X, cands, G, pulse_sec, win, hop, max_sec, max_promote_sec=6.0):
    """
    The gain criterion splits a single accent cycle between the bar and its
    half: with one accent every 6 beats, a 3-beat model already 'explains' 40 %
    of it.  If a 2x or 3x longer cycle has ONE clearly dominant accent (its
    downbeat stands above every other position, including the would-be half-bar
    downbeat), that longer cycle is the bar -- e.g. a metronome accenting beat 1
    of 6, or a 6/8 click with a strong '1' and a weaker '4'.  Two equal peaks
    (hypermeter, 2-bar patterns) do not qualify.
    """
    M = cands[0][1]
    by_m = {m: s for s, m in cands}
    changed = True
    while changed:
        changed = False
        for k in (2, 3):
            Mk = M * k
            if Mk not in by_m or G[Mk] <= 0 or Mk * pulse_sec > min(max_sec, max_promote_sec):
                continue
            phase = _downbeat_phase(X, Mk, win, hop)
            s = _bar_profile(X, Mk, phase, _DOWNBEAT_W)
            top, second, med = s[0], np.max(s[1:]), np.median(s)
            if top - second >= max(0.3, 0.5 * (top - med)):
                M, changed = Mk, True
                break
    if M != cands[0][1]:
        cands = [c for c in cands if c[1] == M] + [c for c in cands if c[1] != M]
    return cands


# ---- d) downbeat phase -----------------------------------------------------
def _downbeat_phase(X, M, win, hop):
    """Phase (pulse index mod M of the downbeat) for every pulse, decided per window."""
    n = len(X)
    w = _DOWNBEAT_W / _DOWNBEAT_W.sum()
    phases, centers = [], []
    for s0 in (list(range(0, max(1, n - win + 1), hop)) or [0]):
        Z = X[s0:s0 + win]
        pos = np.arange(s0, s0 + len(Z)) % M
        prof = np.array([Z[pos == p].mean(0) if np.any(pos == p) else np.zeros(X.shape[1])
                         for p in range(M)])
        # features are already on a common robust scale: keep real effect sizes
        # (z-scoring across positions would blow tiny differences up)
        prof = prof - prof.mean(0)
        phases.append(int(np.argmax(prof @ w)))
        centers.append(s0 + len(Z) / 2)
    phases = np.array(phases)
    for k in range(1, len(phases) - 1):           # drop isolated outliers
        if phases[k - 1] == phases[k + 1] != phases[k]:
            phases[k] = phases[k - 1]
    idx = np.abs(np.arange(n)[:, None] - np.array(centers)[None, :]).argmin(axis=1)
    return phases[idx]


def _bar_profile(X, M, phase, weights=_BEAT_W):
    rel = (np.arange(len(X)) - phase) % M
    s = X @ (weights / weights.sum())
    return np.array([s[rel == p].mean() if np.any(rel == p) else 0.0 for p in range(M)])


# ---- e) beat grouping ------------------------------------------------------
def _grouping_r2(profile, groups):
    """
    Variance of the bar profile explained by 'group start vs. not'.  The
    downbeat is left out: it starts a group in every grouping, so only the
    other positions can tell 3+3 from 2+2+2 (or 2+2+3 from 3+2+2).
    """
    x = profile[1:] - profile[1:].mean()
    n = len(x)
    sst = float((x ** 2).sum()) + 1e-12
    starts = np.cumsum(groups)[:-1].astype(int)          # group starts after the downbeat
    k = len(starts)
    if not 1 <= k < n:
        return 0.0
    S = float(x[starts - 1].sum())
    return (S ** 2) * n / (k * (n - k)) / sst


def _all_groupings(profile, parts=(2, 3)):
    """
    Exact DP over every composition of the bar into groups with sizes in `parts`
    (for each possible number of groups, the start positions maximising accent).
    Returns [(r2, groups), ...] sorted best first.
    """
    M = len(profile)
    x = profile - profile[1:].mean()
    x[0] = 0.0                                             # downbeat carries no information
    NEG = -1e18
    maxk = M // min(parts)
    f = np.full((maxk + 1, M), NEG)
    back = np.zeros((maxk + 1, M), dtype=int)
    f[1][0] = 0.0
    for k in range(1, maxk):
        for p in range(M):
            if f[k][p] <= NEG / 2:
                continue
            for g in parts:
                q = p + g
                if q < M and f[k][p] + x[q] > f[k + 1][q]:
                    f[k + 1][q] = f[k][p] + x[q]
                    back[k + 1][q] = p
    seen, out = set(), []
    for k in range(1, maxk + 1):
        for p in range(M):
            if f[k][p] <= NEG / 2 or (M - p) not in parts:
                continue
            starts, kk, pp = [p], k, p
            while kk > 1:
                pp = back[kk][pp]
                kk -= 1
                starts.append(pp)
            groups = tuple(np.diff(sorted(starts) + [M]).tolist())
            if groups not in seen:
                seen.add(groups)
                out.append((_grouping_r2(profile, groups), groups))
    out.sort(reverse=True)
    return out


def _subbar_periodicity(X, M, phase):
    """
    How strongly the song repeats with each proper divisor L of the bar, using
    only the pulses that are NOT downbeats (otherwise the downbeat accent alone
    leaks into every divisor: a click accented once per 6 beats would look
    3-periodic and 2-periodic).  Returns {L: strength relative to the bar}.
    """
    rel = (np.arange(len(X)) - phase) % M
    keep = rel != 0
    Xs, rs = X[keep], rel[keep]
    m = len(Xs)

    def adj_r2(pos, n_groups):
        if m <= n_groups + 2:
            return 0.0
        sst = ((Xs - Xs.mean(0)) ** 2).sum(0)
        cnt = np.bincount(pos, minlength=pos.max() + 1)[:, None]
        sums = np.zeros((len(cnt), Xs.shape[1]))
        np.add.at(sums, pos, Xs)
        ssw = ((Xs - (sums / np.maximum(cnt, 1))[pos]) ** 2).sum(0)
        return float(np.mean(1 - (ssw / (m - n_groups)) / (sst / (m - 1) + 1e-12)))

    r_bar = max(adj_r2(rs, M - 1), 0.05)
    return {L: max(adj_r2(rs % L, L), 0.0) / r_bar for L in range(2, M) if M % L == 0}


def _choose_grouping(M, profile, rho, beat_pulses, pulse_bpm, mixed_margin=0.3, tiling_bonus=0.3):
    """
    Regular groupings (2s, 3s, 4s) are scored by the accent profile AND by how
    strongly the whole song repeats at a sub-period that is a multiple of the
    group size (robust to syncopation); the beat tracker's tactus gets a bonus
    when it tiles the bar.  An uneven grouping must beat them by a clear margin.
    """
    uni = []
    for g in (2, 3, 4):
        if M % g or M // g < 2:
            continue
        levels = [L for L in rho if L % g == 0]
        rg = max(rho[L] for L in levels) if levels else 0.0
        if g == beat_pulses:
            bonus = tiling_bonus
        elif beat_pulses > g and beat_pulses % g == 0:
            bonus = tiling_bonus / 2
        else:
            bonus = 0.0
        pr = _grouping_r2(profile, (g,) * (M // g))
        uni.append((0.5 * pr + 0.5 * rg + bonus, (g,) * (M // g), 0.5 * pr + 0.5 * rg))
    uni.sort(reverse=True)
    # with 16th-note pulses a beat can also be 4 pulses long (e.g. 15/16 = 4+4+4+3)
    parts = (2, 3, 4) if pulse_bpm > 500 else (2, 3)
    mixed = [(r, g) for r, g in _all_groupings(profile, parts) if len(set(g)) > 1]
    if beat_pulses <= 1:
        # the pulse is already the beat: only group beats when the music says so
        # (e.g. a metronome accenting just beat 1 of 6 is 6/4, not 6/8 or 3/2)
        u_ev = max((u[2] for u in uni), default=0.0)
        m_ev = mixed[0][0] if mixed else 0.0
        # R^2 is scale-free, so also require the non-downbeat positions to
        # actually differ (in robust-z units) before reading a grouping into them
        flat = np.std(profile[1:]) < 0.15
        if flat or (u_ev < 0.2 and m_ev < 0.35):
            return (M,)
    if uni and (not mixed or mixed[0][0] < uni[0][0] + mixed_margin):
        return uni[0][1]
    if mixed:
        return mixed[0][1]
    return (M,)


# ---- f) notation -----------------------------------------------------------
def _kind(n_beats, base):
    """'simple' / 'compound', or 'irregular ...' for a prime number of beats > 3 (5, 7, 11 ...)."""
    prime = n_beats > 3 and all(n_beats % q for q in range(2, int(n_beats ** 0.5) + 1))
    return f"irregular {base}" if prime else base


def _to_signature(M, groups, pulse_bpm, pulse_is_beat):
    """-> numerator, denominator, kind, grouping (in pulses), beats per bar, beat bpm"""
    groups = tuple(int(g) for g in groups)
    uniform = len(set(groups)) == 1
    if pulse_is_beat and not (uniform and groups[0] == 3 and len(groups) > 1):
        # the pulse itself is the counted (quarter-note) beat, e.g. 5/4 felt as 3+2
        if uniform or len(groups) == 1:
            return M, 4, _kind(M, "simple"), (1,) * M, M, pulse_bpm
        return M, 4, "additive", groups, M, pulse_bpm
    if len(groups) == 1:
        den = 4 if pulse_bpm <= 220 else (8 if pulse_bpm <= 600 else 16)
        return M, den, _kind(M, "simple"), (1,) * M, M, pulse_bpm
    if uniform and groups[0] in (2, 4):
        g, n = groups[0], len(groups)
        beat_bpm = pulse_bpm / g
        while beat_bpm > 200 and n % 2 == 0:            # 16th-note pulses: merge pairs
            g, n, beat_bpm = g * 2, n // 2, beat_bpm / 2
        while n >= 8 and n % 2 == 0 and beat_bpm >= 110:  # 8 fast beats -> 4 (half-time)
            g, n, beat_bpm = g * 2, n // 2, beat_bpm / 2
        return n, 4, _kind(n, "simple"), (g,) * n, n, beat_bpm
    if uniform and groups[0] == 3:
        den = 8 if pulse_bpm <= 600 else 16
        n = len(groups)
        return M, den, _kind(n, "compound"), groups, n, pulse_bpm / 3
    den = 4 if pulse_bpm <= 200 else (8 if pulse_bpm <= 600 else 16)
    return M, den, "additive", groups, len(groups), pulse_bpm / np.mean(groups)


def estimate_time_signature(y, sr, onset_env, beat_times, hop_length=512,
                            max_bar_pulses=64, min_bar_sec=0.6, max_bar_sec=10.0,
                            n_alternatives=5):
    """
    Estimate the time signature of a song.  The search space is open-ended:
    every bar length from 2 to `max_bar_pulses` pulses and every grouping of the
    bar into 2s and 3s, mapped to standard notation (n/4, n/8, n/16 ...).

    Args:
        y, sr          : mono audio and sample rate
        onset_env      : onset strength envelope (from analyze_beats)
        beat_times     : beat instants in seconds (from analyze_beats)
        hop_length     : hop used for onset_env
        max_bar_pulses : largest bar to test, in pulses (64 covers e.g. 32/8, 21/16)
        min_bar_sec, max_bar_sec : plausible bar duration range

    Returns:
        TimeSignature, or None if the audio is too short / has no rhythm.
    """
    beat_times = np.asarray(beat_times, float)
    if len(beat_times) < 8:
        return None
    beat_frames = beat_times * sr / hop_length
    # beat period from 8-beat spans: frame quantisation makes single intervals
    # jitter by a frame (e.g. 0.597 / 0.608 s for a 0.600 s beat)
    k = min(8, len(beat_times) - 1)
    beat_sec = float(np.median((beat_times[k:] - beat_times[:-k]) / k))
    tactus_bpm = 60.0 / beat_sec

    # a) pulse
    pulse_bpm, weak = _estimate_pulse(onset_env, sr, hop_length, tactus_bpm)
    pulse_frames = _pulse_grid(onset_env, sr, hop_length, pulse_bpm, beat_frames)
    pulse_times = pulse_frames * hop_length / sr
    # drop grid points in leading / trailing silence (the trackers run on with
    # trim=False); a silent 'pulse' would look like a huge accent in the stats
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    active = np.flatnonzero(rms > rms.max() * 10 ** (-45 / 20))
    if len(active):
        t0, t1 = active[0] * hop_length / sr, active[-1] * hop_length / sr
        pulse_times = pulse_times[(pulse_times >= t0 - 0.05) & (pulse_times <= t1)]
    if len(pulse_times) < 16:
        return None
    pulse_bpm = 60.0 / float(np.median(np.diff(pulse_times)))
    pulse_sec = 60.0 / pulse_bpm
    k = beat_sec / pulse_sec
    beat_pulses = int(round(k)) if (not weak and round(k) >= 1 and abs(k - round(k)) < 0.15) else 1
    pulse_is_beat = beat_pulses == 1 and pulse_bpm <= 200

    # b) features, c) bar length
    X = _normalise_features(_pulse_features(y, sr, pulse_times, beat_sec), pulse_sec)
    R, G, win, hop, Rf = _bar_scores(X, pulse_sec, max_bar_pulses)
    cands = _bar_candidates(R, G, pulse_sec, beat_pulses, min_bar_sec, max_bar_sec)
    if not cands:
        return None
    cands = _promote_accent_cycle(X, cands, G, pulse_sec, win, hop, max_bar_sec)

    # d) + e) + f) for the best few bar lengths (the runners-up become alternatives)
    results = []
    for score, M in cands[:n_alternatives]:
        phase = _downbeat_phase(X, M, win, hop)
        groups = _choose_grouping(M, _bar_profile(X, M, phase), _subbar_periodicity(X, M, phase),
                                  beat_pulses, pulse_bpm)
        num, den, kind, grouping, n_beats, beat_bpm = _to_signature(M, groups, pulse_bpm, pulse_is_beat)
        results.append(dict(score=score, M=M, phase=phase, num=num, den=den, kind=kind,
                            grouping=grouping, n_beats=n_beats, beat_bpm=beat_bpm))
    total = sum(r["score"] for r in results) + 1e-12
    best = results[0]

    alts = {}
    for r in results:
        lab = f"{r['num']}/{r['den']}"
        alts[lab] = alts.get(lab, 0.0) + r["score"] / total
    p_best = alts[f"{best['num']}/{best['den']}"]
    # clarity: how strongly the best-explained cues repeat with this bar length
    # (a metronome may differ only in timbre on beat 1, so don't average it away)
    top3 = np.sort(np.nan_to_num(Rf[best["M"]]))[-3:].mean()
    clarity = float(np.clip(top3 / 0.4, 0.0, 1.0))
    # alternatives: other readings with real support (phrase-level multiples dropped)
    best_lab = f"{best['num']}/{best['den']}"
    phrase = {f"{r['num']}/{r['den']}" for r in results
              if r["M"] % best["M"] == 0 and r["M"] // best["M"] >= 3}
    alternatives = sorted(((lab, p) for lab, p in alts.items()
                           if lab != best_lab and lab not in phrase and p >= 0.08),
                          key=lambda t: -t[1])

    M, phase = best["M"], best["phase"]
    is_down = (np.arange(len(pulse_times)) - phase) % M == 0
    return TimeSignature(
        numerator=int(best["num"]), denominator=int(best["den"]),
        grouping=tuple(int(g) for g in best["grouping"]), kind=best["kind"],
        beats_per_bar=int(best["n_beats"]),
        beat_bpm=float(best["beat_bpm"]), pulse_bpm=float(pulse_bpm),
        bar_pulses=int(M), bar_seconds=float(M * pulse_sec),
        confidence=float(p_best * clarity),
        downbeat_times=pulse_times[is_down],
        alternatives=alternatives,
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Extract tempo, beat instants and time signature from a song.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="Path to a local audio file (mp3/wav/flac/...)")
    src.add_argument("--url", help="Direct URL to an audio file")

    parser.add_argument("--csv", default="beat_times.csv", help="Output CSV path for beat times")
    parser.add_argument("--plot", action="store_true", help="Save a waveform+onset plot (PNG)")
    parser.add_argument("--plot-out", default="beat_analysis.png", help="Plot output path")
    parser.add_argument("--hop-length", type=int, default=512, help="STFT hop length in samples")
    parser.add_argument(
        "--time-varying", action="store_true", help="Also print a time-varying tempo curve"
    )
    parser.add_argument(
        "--max-bar-pulses", type=int, default=64,
        help="Largest bar length (in pulses) tested by the time-signature search",
    )
    parser.add_argument(
        "--downbeats-csv", default=None, help="Optional CSV path for bar (downbeat) times"
    )
    args = parser.parse_args()

    # Resolve source to a local file path
    if args.url:
        tmp_dir = tempfile.mkdtemp()
        audio_path = download_audio(args.url, tmp_dir)
    else:
        audio_path = args.input
        if not os.path.exists(audio_path):
            print(f"[error] file not found: {audio_path}")
            sys.exit(1)

    y, sr = load_audio(audio_path)

    tempo_bpm, beat_times, onset_env, onset_times = analyze_beats(
        y, sr, hop_length=args.hop_length
    )

    time_sig = estimate_time_signature(
        y, sr, onset_env, beat_times, hop_length=args.hop_length,
        max_bar_pulses=args.max_bar_pulses,
    )

    print("\n===== RESULTS =====")
    print(f"Estimated global tempo : {tempo_bpm:.2f} BPM")
    print(f"Number of beats found  : {len(beat_times)}")
    print("First 10 beat instants (s):", np.round(beat_times[:10], 3).tolist())

    if time_sig is None:
        print("Time signature         : could not be estimated (too short or no clear rhythm)")
    else:
        level = "high" if time_sig.confidence >= 0.6 else ("medium" if time_sig.confidence >= 0.3 else "LOW")
        print(f"Time signature         : {time_sig}   [{time_sig.kind} meter]")
        groups = ""
        if len(set(time_sig.grouping)) > 1 or time_sig.grouping[0] == 3:
            groups = f", beat groups in pulses: {time_sig.grouping_str}"
        print(f"  beats per bar        : {time_sig.beats_per_bar}{groups}, "
              f"felt beat ≈ {time_sig.beat_bpm:.1f} BPM")
        print(f"  bar length           : {time_sig.bar_pulses} pulses = {time_sig.bar_seconds:.2f} s "
              f"(pulse ≈ {time_sig.pulse_bpm:.1f} per minute)")
        print(f"  confidence           : {time_sig.confidence:.2f} ({level})")
        if time_sig.alternatives:
            alts = ", ".join(f"{lab} ({p:.0%})" for lab, p in time_sig.alternatives)
            print(f"  alternatives         : {alts}")
        print("  first 5 downbeats (s):", np.round(time_sig.downbeat_times[:5], 3).tolist())
        if args.downbeats_csv:
            save_downbeats_csv(time_sig.downbeat_times, args.downbeats_csv)

    save_csv(beat_times, args.csv)

    if args.time_varying:
        t_times, t_curve, _ = time_varying_tempo(onset_env, sr, hop_length=args.hop_length)
        print("\nTime-varying tempo (sampled every ~5s):")
        for i in range(0, len(t_times), max(1, len(t_times) // 500)):
            print(f"  t={t_times[i]:6.2f}s  tempo≈{t_curve[i]:6.2f} BPM")

    if args.plot:
        import librosa.display  # noqa: F401  (registers waveshow)

        make_plot(y, sr, onset_env, onset_times, beat_times, tempo_bpm, args.plot_out,
                  downbeat_times=None if time_sig is None else time_sig.downbeat_times,
                  time_sig=time_sig)


if __name__ == "__main__":
    main()