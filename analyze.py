#!/usr/bin/env python3
"""
SoundMind analyze.py — complete three-tier audio analyzer.

Tier 1 (always runs):
  BPM, key, mode (Krumhansl-Schmuckler), energy, spectral centroid,
  harmonic richness, HNR approximation, acousticness, danceability,
  speechiness, loudness, duration.

Tier 2 (runs if demucs installed, degrades gracefully otherwise):
  Vocal stem separation -> F0, shimmer, HNR from clean vocals.

Tier 3 (always runs, zero extra deps):
  Section arc via novelty curve - per-section valence/arousal coordinates.

Output includes confidence levels per feature and a warnings array
so the LLM knows exactly what to trust.
"""

import sys
import json
import warnings as _warnings
_warnings.filterwarnings("ignore")

import numpy as np
import librosa
import librosa.segment
import librosa.effects
import librosa.feature
import librosa.onset
import scipy.ndimage
import scipy.signal
import os
import tempfile

MIN_DURATION   = 8.0
ANALYSIS_SR    = 22050
MAX_DURATION   = 360.0
MAX_SECTION_FRAMES = 1500
SILENCE_TRIM_DB = 30
TEMPO_FOLD_MIN_BPM = 70.0
TEMPO_FOLD_MAX_BPM = 190.0
SECTION_SIMILARITY_THRESHOLD = 0.9
KEY_NAMES      = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
KS_MAJOR = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
KS_MINOR = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])

def safe_float(val, decimals=3):
    try:
        v = float(np.atleast_1d(val).flat[0])
        return round(v, decimals) if not (np.isnan(v) or np.isinf(v)) else 0.0
    except Exception:
        return 0.0

def normalise(arr):
    s = arr.sum()
    return arr / s if s > 1e-9 else arr

def safe_corrcoef(a, b):
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return -1.0
    corr = float(np.corrcoef(a, b)[0, 1])
    return corr if np.isfinite(corr) else -1.0

def trim_for_analysis(y, sr):
    try:
        y_trim, idx = librosa.effects.trim(y, top_db=SILENCE_TRIM_DB)
        if len(y_trim) == 0:
            return y, 0.0, 0.0
        start_s = float(idx[0] / sr)
        end_trim_s = float((len(y) - idx[1]) / sr)
        return y_trim, start_s, end_trim_s
    except Exception:
        return y, 0.0, 0.0

def fold_tempo(tempo):
    tempo = float(tempo)
    if not np.isfinite(tempo) or tempo <= 0:
        return 0.0
    while tempo < TEMPO_FOLD_MIN_BPM:
        tempo *= 2.0
    while tempo > TEMPO_FOLD_MAX_BPM:
        tempo /= 2.0
    return tempo

def estimate_tempo(y, sr, onset_env=None):
    onset_env = onset_env if onset_env is not None else librosa.onset.onset_strength(y=y, sr=sr)

    beat_tempo = 0.0
    beat_frames = np.array([], dtype=int)
    try:
        tempo_raw, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        beat_tempo = fold_tempo(np.atleast_1d(tempo_raw).flat[0])
    except Exception:
        beat_frames = np.array([], dtype=int)

    frame_tempi = np.array([], dtype=float)
    try:
        frame_tempi = np.asarray(
            librosa.feature.tempo(onset_envelope=onset_env, sr=sr, aggregate=None)
        ).reshape(-1)
        frame_tempi = frame_tempi[np.isfinite(frame_tempi) & (frame_tempi > 0)]
    except Exception:
        frame_tempi = np.array([], dtype=float)

    folded_frames = np.array([tempo for tempo in (fold_tempo(t) for t in frame_tempi) if tempo > 0], dtype=float)
    candidates = []

    if beat_tempo > 0:
        candidates.append(beat_tempo)
    if folded_frames.size:
        rounded, counts = np.unique(np.round(folded_frames).astype(int), return_counts=True)
        top_idx = np.argsort(counts)[::-1][:3]
        candidates.extend(float(rounded[i]) for i in top_idx)
        candidates.append(fold_tempo(np.median(frame_tempi)))

    if not candidates:
        return {
            "tempo": 120.0,
            "confidence": 0.0,
            "candidates": [120.0],
            "onset_env": onset_env,
            "beat_frames": beat_frames,
        }

    unique_candidates = []
    for cand in candidates:
        if cand <= 0:
            continue
        if not any(abs(cand - existing) < 1.0 for existing in unique_candidates):
            unique_candidates.append(cand)

    def frame_support(candidate):
        if not folded_frames.size:
            return 0.0
        distances = np.abs(folded_frames - candidate)
        return float(np.mean(np.exp(-0.5 * (distances / 4.0) ** 2)))

    scored_candidates = []
    for candidate in unique_candidates:
        support = frame_support(candidate)
        if beat_tempo > 0:
            beat_agreement = float(np.clip(1.0 - abs(candidate - beat_tempo) / 20.0, 0.0, 1.0))
        else:
            beat_agreement = 0.25
        score = support * 0.7 + beat_agreement * 0.3
        scored_candidates.append((candidate, score))

    scored_candidates.sort(key=lambda item: item[1], reverse=True)
    chosen_tempo, chosen_score = scored_candidates[0]
    sorted_candidates = [round(float(candidate), 1) for candidate, _ in scored_candidates[:3]]

    confidence = chosen_score
    if np.mean(onset_env) < 1e-4:
        confidence *= 0.75
    if len(beat_frames) < 4:
        confidence *= 0.85

    return {
        "tempo": round(float(chosen_tempo), 2),
        "confidence": round(float(np.clip(confidence, 0.0, 1.0)), 3),
        "candidates": sorted_candidates,
        "onset_env": onset_env,
        "beat_frames": beat_frames,
    }

def detect_key_mode(y, sr):
    """Harmonic-aware Krumhansl-Schmuckler key detection with ambiguity reporting."""
    y_harm, _ = librosa.effects.hpss(y, margin=4.0)
    chroma_harm = librosa.feature.chroma_cqt(y=y_harm, sr=sr, bins_per_octave=36)
    chroma_mix = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = normalise(
        normalise(chroma_harm.mean(axis=1)) * 0.75 +
        normalise(chroma_mix.mean(axis=1)) * 0.25
    )

    scored_keys = []
    maj_norm = normalise(KS_MAJOR)
    min_norm = normalise(KS_MINOR)
    for i in range(12):
        rotated = np.roll(chroma_mean, -i)
        scored_keys.append({"key": i, "mode": 1, "score": safe_corrcoef(rotated, maj_norm)})
        scored_keys.append({"key": i, "mode": 0, "score": safe_corrcoef(rotated, min_norm)})

    scored_keys.sort(key=lambda item: item["score"], reverse=True)
    best = scored_keys[0]
    runner_up = scored_keys[1]
    score_strength = float(np.clip((best["score"] + 1.0) / 2.0, 0.0, 1.0))
    margin_strength = float(np.clip((best["score"] - runner_up["score"]) / 0.2, 0.0, 1.0))
    confidence = round(float(np.clip(score_strength * 0.7 + margin_strength * 0.3, 0.0, 1.0)), 3)

    return {
        "key": int(best["key"]),
        "key_name": KEY_NAMES[int(best["key"])],
        "mode": int(best["mode"]),
        "mode_name": "major" if int(best["mode"]) == 1 else "minor",
        "key_confidence": confidence,
        "key_correlation": round(float(best["score"]), 3),
        "key_alternatives": [
            {
                "key_name": KEY_NAMES[int(item["key"])],
                "mode_name": "major" if int(item["mode"]) == 1 else "minor",
                "correlation": round(float(item["score"]), 3),
            }
            for item in scored_keys[1:3]
        ],
    }

def compute_hnr_approx(y, sr):
    """HNR from full mix via harmonic/percussive separation."""
    try:
        y_harm, y_perc = librosa.effects.hpss(y, margin=3.0)
        harm_power = float(np.mean(y_harm ** 2))
        perc_power = float(np.mean(y_perc ** 2))
        if perc_power < 1e-10:
            return 15.0
        ratio = harm_power / (perc_power + 1e-10)
        return round(float(np.clip(10 * np.log10(ratio + 1e-9) + 10, 0, 25)), 2)
    except Exception:
        return 8.0

def compute_valence_proxy(y, sr, mode, sc_mean, hnr_approx):
    """
    4-component valence proxy:
      brightness 30% + mode weight 30% + harmonic richness 20% + HNR 20%
    """
    centroid   = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    brightness = float(np.clip((np.mean(centroid) - 500) / 4500, 0, 1))
    mode_weight = 0.15 if mode == 1 else -0.15
    sc_norm  = float(np.clip(sc_mean / 40.0, 0, 1))
    hnr_norm = float(np.clip(hnr_approx / 20.0, 0, 1))
    valence  = (brightness * 0.30
                + (0.5 + mode_weight) * 0.30
                + sc_norm * 0.20
                + hnr_norm * 0.20)
    return round(float(np.clip(valence, 0, 1)), 3)

def compute_acousticness(y, sr):
    """ZCR + spectral rolloff - more robust than flatness for modern music."""
    zcr  = float(np.mean(librosa.feature.zero_crossing_rate(y)[0]))
    roll = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)[0]))
    zcr_score  = float(np.clip(1 - (zcr  - 0.02) / 0.13, 0, 1))
    roll_score = float(np.clip(1 - (roll - 3000)  / 6000, 0, 1))
    return round(zcr_score * 0.6 + roll_score * 0.4, 3)

def compute_danceability(y, sr, tempo, onset_env=None):
    """Onset regularity + tempo in dance range (90-140 BPM peak)."""
    onset_env = onset_env if onset_env is not None else librosa.onset.onset_strength(y=y, sr=sr)
    if np.mean(onset_env) < 1e-9:
        return 0.5
    cv = float(np.std(onset_env) / (np.mean(onset_env) + 1e-9))
    regularity  = float(np.clip(1 - cv / 2.0, 0, 1))
    tempo_score = float(np.clip(1 - abs(tempo - 115) / 80, 0, 1))
    return round(regularity * 0.6 + tempo_score * 0.4, 3)

def compute_temporal_features(y, sr, onset_env=None, beat_frames=None):
    """Dynamics, transient density, and pulse stability from frame-level energy."""
    rms = librosa.feature.rms(y=y)[0]
    rms_db = librosa.amplitude_to_db(np.maximum(rms, 1e-8), ref=1.0)
    onset_env = onset_env if onset_env is not None else librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    if beat_frames is None:
        _, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)

    duration = max(float(librosa.get_duration(y=y, sr=sr)), 1e-9)
    energy_variability = float(np.std(rms) / (np.mean(rms) + 1e-9))
    crest_factor = float(np.max(np.abs(y)) / (np.sqrt(np.mean(y ** 2)) + 1e-9))

    tempo_stability = 0.0
    if len(beat_frames) >= 4:
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        intervals = np.diff(beat_times)
        tempo_stability = 1 - float(np.clip(np.std(intervals) / (np.mean(intervals) + 1e-9), 0, 1))

    return {
        "dynamic_range_db": round(float(np.percentile(rms_db, 95) - np.percentile(rms_db, 10)), 2),
        "mean_loudness_db": round(float(np.mean(rms_db)), 2),
        "energy_variability": round(float(np.clip(energy_variability / 1.5, 0, 1)), 3),
        "crest_factor": round(crest_factor, 2),
        "onset_rate_per_second": round(float(len(onset_frames) / duration), 3),
        "tempo_stability": round(tempo_stability, 3),
    }

def compute_spectral_features(y, sr):
    """Brightness, spread, rolloff, and noisiness of the full mix."""
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    flatness = librosa.feature.spectral_flatness(y=y)[0]

    return {
        "spectral_centroid_hz": round(float(np.mean(centroid)), 2),
        "spectral_bandwidth_hz": round(float(np.mean(bandwidth)), 2),
        "spectral_rolloff_hz": round(float(np.mean(rolloff)), 2),
        "spectral_flatness": round(float(np.mean(flatness)), 4),
        "brightness": round(float(np.clip((np.mean(centroid) - 500) / 4500, 0, 1)), 3),
    }

def extract_vocal_features_praat(y_vocal, sr):
    """F0, shimmer, HNR from isolated vocal stem via parselmouth/Praat."""
    try:
        import parselmouth
        from parselmouth.praat import call
        import soundfile as sf

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, y_vocal, sr)
        tmp.close()

        snd = parselmouth.Sound(tmp.name)
        os.unlink(tmp.name)

        pitch       = call(snd, "To Pitch", 0.0, 75, 600)
        f0_values   = pitch.selected_array["frequency"]
        f0_voiced   = f0_values[f0_values > 0]

        if len(f0_voiced) < 10:
            return None

        f0_mean  = round(float(np.mean(f0_voiced)), 2)
        f0_range = round(float(np.max(f0_voiced) - np.min(f0_voiced)), 2)
        f0_std   = round(float(np.std(f0_voiced)), 2)

        pp       = call(snd, "To PointProcess (periodic, cc)", 75, 600)
        shimmer  = call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer  = round(float(shimmer) * 100, 3) if shimmer else None

        harm     = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr_db   = call(harm, "Get mean", 0, 0)
        hnr_db   = round(float(hnr_db), 2) if hnr_db else None

        return {"f0_mean_hz": f0_mean, "f0_range_hz": f0_range,
                "f0_std_hz": f0_std, "shimmer_pct": shimmer,
                "hnr_db": hnr_db, "confidence": "high"}
    except Exception as e:
        return {"error": str(e), "confidence": "unavailable"}

def separate_vocals_demucs(filepath):
    """Separate vocals in-process and return a mono vocal tensor for Praat."""

    try:
        from pathlib import Path
        from demucs.apply import apply_model
        from demucs.pretrained import get_model
        from demucs.separate import load_track

        model = get_model("htdemucs")
        model.cpu()
        model.eval()

        mixture = load_track(
            Path(filepath),
            model.audio_channels,
            model.samplerate,
        )
        reference = mixture.mean(0)
        mixture -= reference.mean()
        mixture /= reference.std() + 1e-9

        sources = apply_model(
            model,
            mixture[None],
            device="cpu",
            shifts=1,
            split=True,
            overlap=0.25,
            progress=False,
            num_workers=0,
        )[0]
        sources *= reference.std()
        sources += reference.mean()

        vocal_index = model.sources.index("vocals")
        vocal_audio = sources[vocal_index].mean(0).detach().cpu().numpy()
        return vocal_audio, model.samplerate, None

    except Exception as e:
        return None, None, str(e)

def assign_section_labels(sections, embeddings):
    if not sections:
        return sections

    labeled_sections = [dict(section) for section in sections]
    count = len(labeled_sections)
    durations = np.array([section["end_s"] - section["start_s"] for section in labeled_sections], dtype=float)
    median_duration = float(np.median(durations)) if len(durations) else 0.0

    sim = np.eye(count, dtype=float)
    if embeddings:
        emb = np.vstack(embeddings)
        emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)
        sim = emb @ emb.T

    internal_indices = list(range(1, count - 1)) if count > 2 else list(range(count))
    chorus_group = set()
    if internal_indices:
        repeated_counts = {
            i: int(np.sum(sim[i, internal_indices] >= SECTION_SIMILARITY_THRESHOLD)) - 1
            for i in internal_indices
        }
        chorus_seed = max(internal_indices, key=lambda i: (repeated_counts[i], durations[i]))
        if repeated_counts[chorus_seed] > 0:
            chorus_group = {
                i for i in internal_indices
                if sim[chorus_seed, i] >= SECTION_SIMILARITY_THRESHOLD
            }

    first_chorus = min(chorus_group) if chorus_group else None
    last_chorus = max(chorus_group) if chorus_group else None

    for index, section in enumerate(labeled_sections):
        label = f"section_{index + 1}"
        if index in chorus_group:
            label = "chorus"
        elif index == 0 and count >= 3 and durations[index] <= max(median_duration * 0.9, 12.0):
            label = "intro"
        elif index == count - 1 and count >= 3 and durations[index] <= max(median_duration * 0.9, 12.0):
            label = "outro"
        elif chorus_group and index == first_chorus - 1 and durations[index] <= max(median_duration * 0.95, 10.0):
            label = "pre_chorus"
        elif chorus_group and last_chorus is not None and index == last_chorus + 1 and index < count - 1:
            label = "bridge"
        elif 0 < index < count - 1:
            label = "verse"
        section["label"] = label

    return labeled_sections

def detect_sections(y, sr, n_sections=5, start_offset_s=0.0, total_duration_s=None):
    """Novelty-curve section detection with repeated-section-aware labeling."""
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)
        mfcc   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=512)
        feat   = np.vstack([
            librosa.util.normalize(chroma, norm=2, axis=0),
            librosa.util.normalize(mfcc,   norm=2, axis=0)
        ])
        frame_step = max(1, int(np.ceil(feat.shape[1] / MAX_SECTION_FRAMES)))
        feat = feat[:, ::frame_step]
        R = librosa.segment.recurrence_matrix(
            feat, mode="affinity", metric="cosine", sym=True
        )
        ksz    = max(4, min(32, R.shape[0] // 8))
        kernel = np.ones((ksz, ksz))
        kernel[:ksz//2, :ksz//2] =  1
        kernel[ksz//2:, ksz//2:] =  1
        kernel[:ksz//2, ksz//2:] = -1
        kernel[ksz//2:, :ksz//2] = -1
        novelty = scipy.ndimage.convolve(
            R, kernel / (kernel.sum() + 1e-9), mode="reflect"
        ).diagonal()

        hop      = 512 * frame_step
        min_dist = max(1, int(sr * 15 / hop))
        peaks, props = scipy.signal.find_peaks(
            novelty,
            distance=min_dist,
            prominence=max(float(np.std(novelty)) * 0.15, 1e-6),
        )
        if len(peaks) > n_sections - 1:
            top_idx = np.argsort(props["prominences"])[::-1][:n_sections - 1]
            peaks = np.sort(peaks[top_idx])

        duration_frames = feat.shape[1]
        duration_s = float(librosa.frames_to_time(duration_frames, sr=sr, hop_length=hop))

        boundaries = [0.0]
        for p in peaks:
            t = float(librosa.frames_to_time(p, sr=sr, hop_length=hop))
            boundaries.append(round(t, 1))
        boundaries.append(round(duration_s, 1))
        boundaries = sorted(set(boundaries))

        sections = []
        embeddings = []
        for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            if end - start < 5.0:
                continue
            start_frame = int(np.floor((start * sr) / hop))
            end_frame = int(np.ceil((end * sr) / hop))
            segment_feat = feat[:, start_frame:end_frame]
            if segment_feat.size == 0:
                continue
            embedding = np.mean(segment_feat, axis=1)
            embeddings.append(embedding)
            sections.append({
                "label": f"section_{i+1}",
                "start_s": round(start, 1),
                "end_s": round(end, 1),
            })

        sections = assign_section_labels(sections, embeddings)
        if not sections:
            return None

        adjusted_sections = []
        for section in sections:
            start_s = round(section["start_s"] + start_offset_s, 1)
            end_s = round(section["end_s"] + start_offset_s, 1)
            if total_duration_s is not None:
                end_s = round(min(end_s, total_duration_s), 1)
            adjusted_sections.append({
                **section,
                "start_s": start_s,
                "end_s": max(start_s, end_s),
            })
        return adjusted_sections
    except Exception:
        return None

def analyze_section(y_section, sr, mode):
    """Valence + arousal + Russell coordinates for one audio section."""
    if len(y_section) < sr * 3:
        return None
    try:
        onset_env = librosa.onset.onset_strength(y=y_section, sr=sr)
        rms    = float(np.mean(librosa.feature.rms(y=y_section)[0]))
        energy = round(float(np.clip(rms * 10, 0, 1)), 3)

        tempo_info = estimate_tempo(y_section, sr, onset_env=onset_env)
        tempo_s = tempo_info["tempo"]

        sc      = librosa.feature.spectral_contrast(y=y_section, sr=sr)
        sc_mean = float(np.mean(sc))
        hnr     = compute_hnr_approx(y_section, sr)
        valence = compute_valence_proxy(y_section, sr, mode, sc_mean, hnr)

        tempo_n = float(np.clip((tempo_s - 60) / 120, 0, 1))
        arousal = round(float(np.clip(energy * 0.65 + tempo_n * 0.35, 0, 1)), 3)

        v_cs = round(valence * 2 - 1, 2)
        a_cs = round(arousal * 2 - 1, 2)
        angle  = float(np.degrees(np.arctan2(a_cs, v_cs))) % 360
        radius = round(float(np.sqrt(v_cs**2 + a_cs**2) / np.sqrt(2)), 2)

        return {"valence": valence, "arousal": arousal,
                "angle": round(angle, 1), "radius": radius}
    except Exception:
        return None

def analyze(filepath):
    warnings_list = []
    result        = {}
    result["input_file"] = os.path.basename(filepath)

    # Load
    try:
        y, sr = librosa.load(filepath, sr=ANALYSIS_SR, mono=True, duration=MAX_DURATION)
    except Exception as e:
        return {"error": f"Could not load audio: {e}"}

    duration = float(librosa.get_duration(y=y, sr=sr))
    result["duration_s"] = round(duration, 2)

    y_analysis, trim_start_s, trim_end_s = trim_for_analysis(y, sr)
    analysis_duration = float(librosa.get_duration(y=y_analysis, sr=sr))
    trimmed_total = trim_start_s + trim_end_s
    if trimmed_total >= 0.5:
        result["trimmed_silence_s"] = round(trimmed_total, 2)
        warnings_list.append(
            f"Trimmed {trimmed_total:.1f}s of leading/trailing silence before analysis"
        )

    if analysis_duration < MIN_DURATION:
        return {
            "error": (
                f"Usable audio too short after silence trimming ({analysis_duration:.1f}s). "
                f"Minimum is {MIN_DURATION}s."
            )
        }

    # ── Tier 1 ───────────────────────────────────────────────────────────────

    # BPM
    try:
        tempo_info = estimate_tempo(y_analysis, sr)
        tempo = tempo_info["tempo"]
        onset_env = tempo_info["onset_env"]
        beat_frames = tempo_info["beat_frames"]
        result["tempo_confidence"] = tempo_info["confidence"]
        result["tempo_candidates_bpm"] = tempo_info["candidates"]
        if tempo_info["confidence"] < 0.5:
            warnings_list.append(
                f"Tempo estimate uncertain ({tempo_info['confidence']:.2f}) - half/double-time ambiguity possible"
            )
    except Exception as e:
        tempo = 120.0
        onset_env = None
        beat_frames = None
        result["tempo_confidence"] = 0.0
        result["tempo_candidates_bpm"] = [120.0]
        warnings_list.append(f"BPM detection failed: {e}")
    result["tempo"] = tempo

    # Key + mode
    try:
        key_info = detect_key_mode(y_analysis, sr)
        result.update(key_info)
        if key_info["key_confidence"] < 0.55:
            warnings_list.append(
                f"Key detection confidence low ({key_info['key_confidence']:.2f}) - "
                "song may be atonal or use complex modulation"
            )
        elif key_info["key_alternatives"]:
            alt = key_info["key_alternatives"][0]
            if abs(key_info["key_correlation"] - alt["correlation"]) < 0.05:
                warnings_list.append(
                    f"Key estimate is close to {alt['key_name']} {alt['mode_name']} as well"
                )
    except Exception as e:
        result.update({"key": 0, "key_name": "C", "mode": 1,
                       "mode_name": "major", "key_confidence": 0.0,
                       "key_correlation": 0.0, "key_alternatives": []})
        warnings_list.append(f"Key detection failed: {e}")

    # Energy
    try:
        rms    = librosa.feature.rms(y=y_analysis)[0]
        energy = round(float(np.clip(np.mean(rms) * 10, 0, 1)), 3)
    except Exception:
        energy = 0.5
        warnings_list.append("Energy estimation failed")
    result["energy"] = energy

    # Spectral contrast
    try:
        sc      = librosa.feature.spectral_contrast(y=y_analysis, sr=sr, n_bands=6)
        sc_mean = float(np.mean(sc))
    except Exception:
        sc_mean = 20.0
        warnings_list.append("Spectral contrast unavailable")

    # HNR
    try:
        hnr_approx = compute_hnr_approx(y_analysis, sr)
    except Exception:
        hnr_approx = 8.0
        warnings_list.append("HNR approximation failed")
    result["hnr_db"] = hnr_approx

    # Valence
    try:
        valence = compute_valence_proxy(y_analysis, sr, result.get("mode", 1), sc_mean, hnr_approx)
    except Exception:
        valence = 0.5
        warnings_list.append("Valence estimation failed")
    result["valence"]             = valence
    result["valence_confidence"]  = "estimated"

    # Arousal
    tempo_norm = float(np.clip((tempo - 60) / 120, 0, 1))
    result["arousal"] = round(float(np.clip(energy * 0.65 + tempo_norm * 0.35, 0, 1)), 3)

    try:
        result.update(compute_temporal_features(y_analysis, sr, onset_env=onset_env, beat_frames=beat_frames))
    except Exception as e:
        warnings_list.append(f"Temporal feature extraction failed: {e}")

    try:
        result.update(compute_spectral_features(y_analysis, sr))
    except Exception as e:
        warnings_list.append(f"Spectral feature extraction failed: {e}")

    # Acousticness
    try:
        result["acousticness"] = compute_acousticness(y_analysis, sr)
    except Exception:
        result["acousticness"] = 0.5

    # Danceability
    try:
        result["danceability"] = compute_danceability(y_analysis, sr, tempo, onset_env=onset_env)
    except Exception:
        result["danceability"] = 0.5

    # Speechiness
    try:
        zcr = librosa.feature.zero_crossing_rate(y_analysis)[0]
        result["speechiness"] = round(float(np.clip(np.mean(zcr) * 5, 0, 1)), 3)
    except Exception:
        result["speechiness"] = 0.1

    # Loudness
    result["loudness"] = result.get(
        "mean_loudness_db",
        round(float(20 * np.log10(np.mean(np.abs(y_analysis)) + 1e-9)), 2),
    )

    result["source"]           = "librosa"
    result["tier1_confidence"] = "high"

    # ── Tier 2: Vocal features ────────────────────────────────────────────────
    demucs_enabled = os.environ.get("SOUNDMIND_ENABLE_DEMUCS", "1") == "1"
    demucs_available = False
    if demucs_enabled:
        try:
            import demucs
            demucs_available = True
        except ImportError:
            warnings_list.append(
                "Demucs was enabled but is not installed - vocal features unavailable."
            )
    else:
        warnings_list.append(
            "Vocal stem analysis skipped for faster processing. Set "
            "SOUNDMIND_ENABLE_DEMUCS=1 to enable it."
        )

    if demucs_available:
        y_vocal, sr_vocal, vocal_error = separate_vocals_demucs(filepath)
        if y_vocal is not None:
            vf = extract_vocal_features_praat(y_vocal, sr_vocal)
            if vf and "error" not in vf:
                result["vocal_features"] = vf
                # Refine valence with F0 range (high range = emotional intensity = slightly higher valence)
                f0_contribution = float(np.clip(vf.get("f0_range_hz", 0) / 300.0, 0, 0.1))
                result["valence"] = round(float(np.clip(result["valence"] + f0_contribution, 0, 1)), 3)
                result["valence_confidence"] = "high"
            else:
                warnings_list.append(f"Vocal feature extraction failed: {vf.get('error','unknown')}")
        else:
            warnings_list.append(f"Demucs stem separation failed: {vocal_error}")

    # ── Tier 3: Section arc ───────────────────────────────────────────────────
    sections = detect_sections(
        y_analysis,
        sr,
        start_offset_s=trim_start_s,
        total_duration_s=duration,
    )
    if sections:
        arc = []
        mode_val = result.get("mode", 1)
        for sec in sections:
            local_start = max(sec["start_s"] - trim_start_s, 0.0)
            local_end = max(sec["end_s"] - trim_start_s, local_start)
            s = int(local_start * sr)
            e = int(local_end   * sr)
            scores = analyze_section(y_analysis[s:e], sr, mode_val)
            if scores:
                arc.append({**sec, **scores})
        if arc:
            result["sections"] = arc
        else:
            warnings_list.append("Section arc: boundaries found but scoring failed")
    else:
        warnings_list.append("Section boundaries not detected - file may be too short or uniform")

    result["warnings"]    = warnings_list
    result["confidence"]  = "high" if not warnings_list else "partial"

    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: analyze.py <audio_file_path>"}))
        sys.exit(1)
    fp = sys.argv[1]
    if not os.path.exists(fp):
        print(json.dumps({"error": f"File not found: {fp}"}))
        sys.exit(1)
    print(json.dumps(analyze(fp), ensure_ascii=False))