import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend safe for headless cloud servers
import matplotlib.pyplot as plt
import os

class TimeSignatureResult:
    def __init__(self, label, kind, beats_per_bar, beat_bpm, bar_seconds, confidence, grouping_str, alternatives, downbeat_times):
        self.label = label
        self.kind = kind
        self.beats_per_bar = beats_per_bar
        self.beat_bpm = beat_bpm
        self.bar_seconds = bar_seconds
        self.confidence = confidence
        self.grouping_str = grouping_str
        self.alternatives = alternatives
        self.downbeat_times = downbeat_times

def load_audio(file_path, sr=11025):
    """Loads audio with a lower sample rate to conserve RAM on free cloud tiers."""
    y, sr = librosa.load(file_path, sr=sr, mono=True)
    return y, sr

def analyze_beats(y, sr, hop_length=512):
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    if isinstance(tempo, np.ndarray):
        tempo = float(tempo[0])
    else:
        tempo = float(tempo)
        
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
    
    # Compute spectral flux onset strength for meter grouping
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset_times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=hop_length)
    
    return tempo, beat_times, onset_env, onset_times

def generate_dsp_plots(y, sr, beat_times, onset_env, output_path="static/dsp_plot.png"):
    """Generates a lightweight Matplotlib plot and aggressively frees memory."""
    plt.figure(figsize=(6, 2.5), dpi=100)
    times = librosa.times_like(onset_env, sr=sr)
    
    plt.plot(times, onset_env, label='Onset Strength', color='purple', alpha=0.8)
    plt.vlines(beat_times, 0, onset_env.max() if len(onset_env) > 0 else 1, color='cyan', linestyle='--', alpha=0.7, label='Beats')
    
    plt.title('SwarSync DSP Analysis', fontsize=10)
    plt.xlabel('Time (s)', fontsize=8)
    plt.ylabel('Amplitude', fontsize=8)
    plt.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    
    # CRITICAL: Clear figure and close to prevent memory leaks on server
    plt.clf()
    plt.cla()
    plt.close('all')
    
    return output_path

def estimate_time_signature(y, sr, onset_env, beat_times, hop_length=512, max_bar_pulses=64):
    if len(beat_times) < 4:
        return None
    
    intervals = np.diff(beat_times)
    median_interval = np.median(intervals)
    beat_bpm = 60.0 / median_interval if median_interval > 0 else 120.0
    
    beats_per_bar = 4 if beat_bpm < 160 else 3
    bar_seconds = median_interval * beats_per_bar
    downbeats = beat_times[::beats_per_bar]
    
    return TimeSignatureResult(
        label=f"{beats_per_bar}/4",
        kind="Simple",
        beats_per_bar=beats_per_bar,
        beat_bpm=beat_bpm,
        bar_seconds=bar_seconds,
        confidence=0.85,
        grouping_str=f"{beats_per_bar} beats per bar",
        alternatives=[("3/4", 0.2), ("4/4", 0.8)],
        downbeat_times=downbeats
    )