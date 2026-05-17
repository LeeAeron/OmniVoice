#!/usr/bin/env python3

import argparse
import logging
import logging as py_logging
import gc
import os
import subprocess
import tempfile
import zipfile
import requests
import sys
import threading
import time
import json
import warnings
import random
import hashlib
import pickle
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Iterator, Optional


os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

transformers_logger = py_logging.getLogger("transformers")
transformers_logger.setLevel(py_logging.ERROR)
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
warnings.filterwarnings("ignore", message=".*HF Hub.*")
warnings.filterwarnings("ignore", message=".*forced_decoder_ids.*")
warnings.filterwarnings("ignore", message=".*multilingual Whisper.*")
warnings.filterwarnings("ignore", message=".*custom logits processor.*")
warnings.filterwarnings("ignore", message=".*SuppressTokensLogitsProcessor.*")
warnings.filterwarnings("ignore", message=".*SuppressTokensAtBeginLogitsProcessor.*")

import gradio as gr
import numpy as np
import soundfile as sf
import torch
import matplotlib
matplotlib.use('Agg')  # Using a non-interactive backend
import matplotlib.pyplot as plt
import librosa
import librosa.display

CWD = Path(os.getcwd()).absolute()
OUTPUTS_DIR = CWD / "outputs"
REFERENCE_AUDIO_DIR = CWD / "reference_audio"
MODELS_DIR = CWD / "models"
SETTINGS_FILE = CWD / "settings.txt"
CHECKPOINT_DIR = CWD / "checkpoints"

OUTPUTS_DIR.mkdir(exist_ok=True)
REFERENCE_AUDIO_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)

# SETTING UP THE HF HUB CACHE
os.environ["HF_HOME"] = str(MODELS_DIR / "hf_cache")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(MODELS_DIR / "hf_cache")
os.environ["TRANSFORMERS_CACHE"] = str(MODELS_DIR / "transformers_cache")
os.environ["HF_DATASETS_CACHE"] = str(MODELS_DIR / "datasets_cache")

logging.info(f"Working directory: {CWD}")
logging.info(f"Outputs will be saved to: {OUTPUTS_DIR}")
logging.info(f"Reference voices folder: {REFERENCE_AUDIO_DIR}")
logging.info(f"Models folder: {MODELS_DIR}")
logging.info(f"HF Cache: {MODELS_DIR / 'hf_cache'}")
logging.info(f"Checkpoints folder: {CHECKPOINT_DIR}")
logging.getLogger("modelscope").setLevel(logging.ERROR)

from huggingface_hub import constants as hf_constants
hf_constants.HF_HUB_CACHE = MODELS_DIR / "hf_cache"
hf_constants.DEFAULT_CACHE_PATH = MODELS_DIR / "hf_cache"

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name

_CATEGORIES = {
    "Gender": ["Male", "Female"],
    "Age": ["Child", "Teenager", "Young Adult", "Middle-aged", "Elderly"],
    "Pitch": ["Very Low", "Low", "Moderate", "High", "Very High"],
    "Style": ["Whisper"],
    "English Accent": ["American", "British", "Chinese"],
    "Chinese Dialect": ["Sichuan", "Northeast"],
}

_ALL_LANGUAGES = ["Auto"] + sorted(lang_display_name(n) for n in LANG_NAMES)

AUDIO_FORMATS = ["wav", "mp3", "flac", "ogg", "m4a", "aac"]
TARGET_SAMPLE_RATE = 48000

# Normalization levels
NORMALIZATION_LEVELS = [-20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Bitrate options for lossy formats (kbps)
BITRATE_OPTIONS = [64, 96, 128, 160, 192, 224, 256, 320]

# Output sample rate options
OUTPUT_SAMPLE_RATES = [24000, 32000, 44100, 48000]

# Chunking modes
CHUNKING_MODES = ["lines", "sentences", "characters"]

# Crossfade range (ms)
CROSSFADE_MS_RANGE = [0, 10, 20, 30, 40, 50, 60, 80, 100, 150, 200]

# Max lines per chunk options
MAX_LINES_OPTIONS = list(range(1, 21))

# Max sentences per chunk options
MAX_SENTENCES_OPTIONS = list(range(1, 21))

# Max chars per chunk options
MAX_CHARS_OPTIONS = list(range(100, 3001, 50))


def get_best_device():
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.95)
        torch.backends.cudnn.benchmark = True
        return "cuda"
    return "cpu"


def save_audio_with_ffmpeg(waveform: np.ndarray, sample_rate: int, fmt: str = "wav", prefix: str = "omnivoice", target_sr: int = 48000, bitrate: int = 192) -> str:
    """Saving audio via ffmpeg with configurable sample rate and bitrate."""
    fmt = fmt.lower().replace(".", "")
    if fmt not in AUDIO_FORMATS:
        fmt = "wav"

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_filename = f"{prefix}_{timestamp}.{fmt}"
    out_path = OUTPUTS_DIR / out_filename

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        sf.write(tmp_wav.name, waveform, sample_rate)
        tmp_wav_path = tmp_wav.name

    try:
        cmd = ["ffmpeg", "-y", "-i", tmp_wav_path, "-ar", str(target_sr)]

        if fmt == "mp3":
            cmd.extend(["-c:a", "libmp3lame", "-b:a", f"{bitrate}k"])
        elif fmt == "ogg":
            cmd.extend(["-c:a", "libvorbis", "-q:a", "4"])
        elif fmt == "m4a" or fmt == "aac":
            cmd.extend(["-c:a", "aac", "-b:a", f"{bitrate}k"])
        elif fmt == "flac":
            cmd.extend(["-c:a", "flac"])
        else:
            cmd.extend(["-c:a", "pcm_s16le"])

        cmd.append(str(out_path))

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logging.warning(f"FFmpeg error: {result.stderr}. Falling back to wav.")
            out_path = out_path.with_suffix(".wav")
            sf.write(out_path, waveform, sample_rate)
        else:
            logging.info(f"Saved (resampled to {target_sr}Hz): {out_path}")

    finally:
        os.unlink(tmp_wav_path)

    return str(out_path)


def save_spectrogram(waveform: np.ndarray, sample_rate: int, output_path: str = None) -> str:
    """Generate and save a spectrogram image from audio waveform."""
    if output_path is None:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            output_path = tmp.name

    fig, ax = plt.subplots(figsize=(12, 4))

    # Convert to float if needed
    if waveform.dtype == np.int16:
        waveform_float = waveform.astype(np.float32) / 32768.0
    else:
        waveform_float = waveform.astype(np.float32)

    # Compute spectrogram
    D = librosa.amplitude_to_db(
        np.abs(librosa.stft(waveform_float)), 
        ref=np.max
    )

    img = librosa.display.specshow(
        D, 
        sr=sample_rate, 
        x_axis='time', 
        y_axis='hz', 
        ax=ax
    )

    ax.set_title('Spectrogram')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)

    return output_path


def get_voice_choices() -> List[str]:
    """Get a list of reference audios from the reference_audio folder."""
    if not REFERENCE_AUDIO_DIR.exists():
        return ["-NONE-"]

    files = []
    for f in REFERENCE_AUDIO_DIR.iterdir():
        if f.suffix.lower() in [".wav", ".mp3", ".aac", ".m4a", ".m4b", ".ogg", ".flac", ".opus"]:
            files.append(f.name)

    return ["-NONE-"] + sorted(files)


def download_reference_voices(url: str = "https://huggingface.co/datasets/LeeAeron/F5TTSx/resolve/main/ref_voices.zip") -> gr.update:
    """Loading reference voices."""
    try:
        archive_path = REFERENCE_AUDIO_DIR / "temp_voices.zip"

        logging.info(f"Downloading voices from {url}...")
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()

        with open(archive_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(REFERENCE_AUDIO_DIR)

        archive_path.unlink()
        logging.info("Voices downloaded successfully!")

    except Exception as e:
        logging.error(f"Failed to download voices: {e}")
        gr.Warning(f"Failed to download voices: {e}")

    return gr.update(choices=get_voice_choices(), value="-NONE-")


def set_voice_file(selected_voice: str) -> str:
    """Set the path to the selected reference audio."""
    if selected_voice and selected_voice != "-NONE-":
        file_path = REFERENCE_AUDIO_DIR / selected_voice
        if file_path.exists():
            return str(file_path.absolute())
    return None


def ensure_model_downloaded(repo_id: str = "LeeAeron/OmniVoice", local_dir: Path = None) -> Path:
    """
    Checks if the model exists locally and downloads it if not.
    Returns the path to the local folder containing the model.
    """
    if local_dir is None:
        local_dir = MODELS_DIR / repo_id.replace("/", "--")

    # Check if the model files have already been downloaded
    marker_file = local_dir / ".download_complete"

    # Check the marker file and the presence of config
    if marker_file.exists() and (local_dir / "config.json").exists():
        logging.info(f"Model found locally: {local_dir}")
        return local_dir

    # Need to download
    logging.info(f"Model not found locally. Downloading from HuggingFace: {repo_id}")
    logging.info(f"Target directory: {local_dir}")

    try:
        from huggingface_hub import snapshot_download

        local_dir.mkdir(parents=True, exist_ok=True)

        # Download everything via snapshot_download
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,  # Copy files, not symlinks
            resume_download=True,
        )

        # Create a marker file
        marker_file.touch()

        # Saving version information
        info_file = local_dir / ".repo_info"
        info_file.write_text(json.dumps({
            "repo_id": repo_id,
            "downloaded_at": datetime.now().isoformat(),
        }))

        logging.info(f"Model downloaded successfully to: {local_dir}")
        return Path(downloaded_path)

    except Exception as e:
        logging.error(f"Failed to download model: {e}")
        # If there are partial files, delete them.
        if local_dir.exists():
            import shutil
            shutil.rmtree(local_dir, ignore_errors=True)
        raise RuntimeError(f"Failed to download model {repo_id}: {e}")


def download_whisper_model(model_name: str = "openai/whisper-large-v3", local_dir: Path = None) -> Path:
    """
    Downloads the Whisper model for ASR to a local folder.
    """
    if local_dir is None:
        local_dir = MODELS_DIR / model_name.replace("/", "--")

    marker_file = local_dir / ".download_complete"

    if marker_file.exists():
        logging.info(f"Whisper model found locally: {local_dir}")
        return local_dir

    logging.info(f"Downloading Whisper model: {model_name}")

    try:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        local_dir.mkdir(parents=True, exist_ok=True)

        # Download the model and processor
        model = WhisperForConditionalGeneration.from_pretrained(
            model_name,
            cache_dir=str(MODELS_DIR / "transformers_cache"),
        )
        processor = WhisperProcessor.from_pretrained(
            model_name,
            cache_dir=str(MODELS_DIR / "transformers_cache"),
        )

        # Save locally
        model.save_pretrained(str(local_dir))
        processor.save_pretrained(str(local_dir))

        marker_file.touch()
        logging.info(f"Whisper model saved to: {local_dir}")
        return local_dir

    except Exception as e:
        logging.error(f"Failed to download Whisper: {e}")
        raise


# SPEECH ENHANCEMENT

def ensure_zipenhancer_downloaded() -> Path:
    """Download ZipEnhancer model from ModelScope if not present locally."""
    local_dir = MODELS_DIR / "zipenhancer"
    marker_file = local_dir / ".download_complete"

    if marker_file.exists():
        logging.info(f"ZipEnhancer found locally: {local_dir}")
        return local_dir

    logging.info("Downloading ZipEnhancer model from ModelScope...")
    try:
        from modelscope import snapshot_download

        local_dir.mkdir(parents=True, exist_ok=True)
        downloaded_path = snapshot_download(
            'iic/speech_zipenhancer_ans_multiloss_16k_base',
            local_dir=str(local_dir),
        )

        marker_file.touch()
        logging.info(f"ZipEnhancer downloaded to: {local_dir}")
        return Path(downloaded_path)

    except Exception as e:
        logging.error(f"Failed to download ZipEnhancer: {e}")
        raise RuntimeError(f"Failed to download ZipEnhancer: {e}")


class ZipEnhancerProcessor:
    """Wrapper for ZipEnhancer speech enhancement model."""

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()

    def _load_model(self):
        try:
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks

            self.pipeline = pipeline(
                task=Tasks.acoustic_noise_suppression,
                model=str(self.model_dir),
                device=self.device,
            )
            logging.info(f"ZipEnhancer loaded on {self.device}")
        except Exception as e:
            logging.error(f"Failed to load ZipEnhancer: {e}")
            raise

    def enhance(self, audio_path: str, output_path: str) -> str:
        """Enhance audio file and save result."""
        try:
            result = self.pipeline(audio_path, output_path=output_path)
            logging.info(f"Audio enhanced: {output_path}")
            return output_path
        except Exception as e:
            logging.error(f"Enhancement failed: {e}")
            # Return original if enhancement fails
            return audio_path


# Chunking system
# Sentence splitting regex - matches sentence endings: . ! ? ... (with optional closing quotes/brackets)
_sentence_end_re = re.compile(r"[.!?]+['\"\)\]]*\s+|[.!?]+['\"\)\]]*$", re.MULTILINE)


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences, preserving sentence delimiters."""
    if not text.strip():
        return []
    parts = _sentence_end_re.split(text)
    sentences = [s.strip() for s in parts if s.strip()]
    return sentences


@dataclass
class ScriptChunk:
    """Represents a single chunk of text for generation."""
    text: str
    start_idx: int
    end_idx: int
    is_first: bool
    is_last: bool
    estimated_duration_sec: float = 0.0


class ScriptChunker:
    """Splits long text into adaptively sized chunks.

    Supports three chunking modes:
    - "lines": chunk by number of lines
    - "sentences": chunk by number of sentences (detects . ! ? ... endings)
    - "characters": chunk by character count
    """

    def __init__(
        self,
        chunking_mode: str = "lines",
        max_lines_per_chunk: int = 8,
        max_sentences_per_chunk: int = 5,
        max_chars_per_chunk: int = 800,
        max_tokens_estimate: int = 1500,
        chars_per_token: float = 3.5,
        avg_words_per_minute: float = 150.0,
    ):
        self.chunking_mode = chunking_mode
        self.max_lines = max_lines_per_chunk
        self.max_sentences = max_sentences_per_chunk
        self.max_chars = max_chars_per_chunk
        self.max_tokens = max_tokens_estimate
        self.chars_per_token = chars_per_token
        self.wpm = avg_words_per_minute

    def _estimate_tokens(self, text: str) -> int:
        return int(len(text) / self.chars_per_token)

    def _estimate_duration(self, text: str) -> float:
        words = len(text.split())
        return (words / self.wpm) * 60.0

    def _chunk_by_lines(self, lines: List[str]) -> List[ScriptChunk]:
        """Original line-based chunking."""
        if not lines:
            return []

        chunks = []
        start_idx = 0

        while start_idx < len(lines):
            end_idx = start_idx + 1
            current_chars = len(lines[start_idx])
            current_tokens = self._estimate_tokens(lines[start_idx])

            while (end_idx < len(lines) and 
                   end_idx - start_idx < self.max_lines and
                   current_tokens < self.max_tokens and
                   current_chars < self.max_chars):

                next_line = lines[end_idx]
                next_tokens = self._estimate_tokens(next_line)

                if current_tokens + next_tokens > self.max_tokens * 1.1:
                    break

                current_chars += len(next_line)
                current_tokens += next_tokens
                end_idx += 1

            chunk_text = "\n".join(lines[start_idx:end_idx])
            est_duration = self._estimate_duration(chunk_text)

            chunks.append(ScriptChunk(
                text=chunk_text,
                start_idx=start_idx,
                end_idx=end_idx - 1,
                is_first=(start_idx == 0),
                is_last=(end_idx >= len(lines)),
                estimated_duration_sec=est_duration
            ))

            start_idx = end_idx

        return chunks

    def _chunk_by_sentences(self, text: str) -> List[ScriptChunk]:
        """Chunk by number of sentences."""
        sentences = split_into_sentences(text)
        if not sentences:
            return []

        chunks = []
        current_sentences = []
        current_chars = 0
        current_tokens = 0
        start_idx = 0
        sent_idx = 0

        for sentence in sentences:
            sent_chars = len(sentence)
            sent_tokens = self._estimate_tokens(sentence)

            would_exceed = (
                len(current_sentences) >= self.max_sentences or
                current_chars + sent_chars > self.max_chars or
                current_tokens + sent_tokens > self.max_tokens
            )

            if would_exceed and current_sentences:
                chunk_text = " ".join(current_sentences)
                chunks.append(ScriptChunk(
                    text=chunk_text,
                    start_idx=start_idx,
                    end_idx=sent_idx - 1,
                    is_first=(start_idx == 0),
                    is_last=False,
                    estimated_duration_sec=self._estimate_duration(chunk_text)
                ))
                start_idx = sent_idx
                current_sentences = [sentence]
                current_chars = sent_chars
                current_tokens = sent_tokens
            else:
                current_sentences.append(sentence)
                current_chars += sent_chars
                current_tokens += sent_tokens

            sent_idx += 1

        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append(ScriptChunk(
                text=chunk_text,
                start_idx=start_idx,
                end_idx=sent_idx - 1,
                is_first=(start_idx == 0),
                is_last=True,
                estimated_duration_sec=self._estimate_duration(chunk_text)
            ))

        return chunks

    def _chunk_by_characters(self, text: str) -> List[ScriptChunk]:
        """Chunk by total character count."""
        if not text:
            return []

        chunks = []
        current_text = ""
        current_chars = 0
        start_idx = 0
        char_idx = 0

        for char in text:
            char_idx += 1
            current_text += char
            current_chars += 1

            would_exceed = current_chars >= self.max_chars

            if would_exceed and current_text.strip():
                chunks.append(ScriptChunk(
                    text=current_text.strip(),
                    start_idx=start_idx,
                    end_idx=char_idx - 1,
                    is_first=(start_idx == 0),
                    is_last=False,
                    estimated_duration_sec=self._estimate_duration(current_text)
                ))
                start_idx = char_idx
                current_text = ""
                current_chars = 0

        if current_text.strip():
            chunks.append(ScriptChunk(
                text=current_text.strip(),
                start_idx=start_idx,
                end_idx=char_idx,
                is_first=(start_idx == 0),
                is_last=True,
                estimated_duration_sec=self._estimate_duration(current_text)
            ))

        return chunks

    def parse_text(self, text: str) -> List[ScriptChunk]:
        """Parse text into chunks using the configured chunking mode."""
        if not text or not text.strip():
            return []

        if self.chunking_mode == "sentences":
            return self._chunk_by_sentences(text)
        elif self.chunking_mode == "characters":
            return self._chunk_by_characters(text)
        else:  # "lines" (default)
            lines = [l for l in text.strip().split("\n") if l.strip()]
            return self._chunk_by_lines(lines)


class AudioCrossfader:
    """Smooth gluing of audio chunks with cosine crossfade."""

    def __init__(self, fade_duration_ms: float = 50.0, sample_rate: int = 24000):
        self.fade_samples = max(1, int(sample_rate * (fade_duration_ms / 1000.0)))
        self.sample_rate = sample_rate

    def apply_crossfade(self, chunk1: np.ndarray, chunk2: np.ndarray) -> np.ndarray:
        if len(chunk1) < self.fade_samples * 2 or len(chunk2) < self.fade_samples * 2:
            return np.concatenate([chunk1, chunk2])

        fade_out = chunk1[-self.fade_samples:]
        fade_in = chunk2[:self.fade_samples]

        t = np.linspace(0, 1, self.fade_samples)
        curve_out = np.cos(t * np.pi / 2)
        curve_in = np.sin(t * np.pi / 2)

        mixed = fade_out * curve_out + fade_in * curve_in

        result = np.concatenate([
            chunk1[:-self.fade_samples],
            mixed,
            chunk2[self.fade_samples:]
        ])

        return result

    def concatenate_chunks(self, chunks: List[np.ndarray]) -> np.ndarray:
        if not chunks:
            return np.array([])
        if len(chunks) == 1:
            return chunks[0]

        result = chunks[0]
        for next_chunk in chunks[1:]:
            result = self.apply_crossfade(result, next_chunk)

        return result


@dataclass
class GenerationCheckpoint:
    """Save point to resume generation."""
    session_id: str
    text_hash: str
    mode: str  # "clone" or "design"
    language: str
    num_step: int
    guidance_scale: float
    denoise: bool
    preprocess_prompt: bool
    postprocess_output: bool
    speed: float
    duration: float
    seed: int
    # state
    total_chunks: int
    completed_chunks: int
    chunk_audio_files: List[str]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "text_hash": self.text_hash,
            "mode": self.mode,
            "language": self.language,
            "num_step": self.num_step,
            "guidance_scale": self.guidance_scale,
            "denoise": self.denoise,
            "preprocess_prompt": self.preprocess_prompt,
            "postprocess_output": self.postprocess_output,
            "speed": self.speed,
            "duration": self.duration,
            "seed": self.seed,
            "total_chunks": self.total_chunks,
            "completed_chunks": self.completed_chunks,
            "chunk_audio_files": self.chunk_audio_files,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GenerationCheckpoint":
        return cls(**data)


class CheckpointManager:
    """Controls saving and loading checkpoints."""

    def __init__(self, checkpoint_dir: Path = CHECKPOINT_DIR):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def _get_checkpoint_path(self, session_id: str) -> Path:
        return self.checkpoint_dir / f"checkpoint_{session_id}.pkl"

    def save(self, checkpoint: GenerationCheckpoint):
        path = self._get_checkpoint_path(checkpoint.session_id)
        with open(path, 'wb') as f:
            pickle.dump(checkpoint.to_dict(), f)
        logging.info(f"[Checkpoint] Saved: {path} ({checkpoint.completed_chunks}/{checkpoint.total_chunks} chunks)")

    def load(self, session_id: str) -> Optional[GenerationCheckpoint]:
        path = self._get_checkpoint_path(session_id)
        if not path.exists():
            return None
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            ckpt = GenerationCheckpoint.from_dict(data)
            logging.info(f"[Checkpoint] Loaded: {session_id} ({ckpt.completed_chunks}/{ckpt.total_chunks} chunks)")
            return ckpt
        except Exception as e:
            logging.error(f"[Checkpoint] Failed to load: {e}")
            return None

    def delete(self, session_id: str):
        path = self._get_checkpoint_path(session_id)
        if path.exists():
            os.remove(path)
            logging.info(f"[Checkpoint] Deleted: {session_id}")

    def list_available(self) -> List[Tuple[str, int, int, float]]:
        """Returns a list (session_id, completed, total, timestamp)."""
        available = []
        for fname in os.listdir(self.checkpoint_dir):
            if fname.startswith("checkpoint_") and fname.endswith(".pkl"):
                try:
                    with open(self.checkpoint_dir / fname, 'rb') as f:
                        data = pickle.load(f)
                    sid = data["session_id"]
                    available.append((sid, data["completed_chunks"], data["total_chunks"], data["timestamp"]))
                except:
                    continue
        return sorted(available, key=lambda x: x[3], reverse=True)

    def generate_session_id(self, text: str, mode: str, seed: int) -> str:
        """Generates a unique session ID based on parameters."""
        effective_seed = seed if seed >= 0 else 0
        content = f"{text}:{mode}:{effective_seed}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class ProgressTracker:
    """Tracks progress and calculates ETA."""

    def __init__(self, total_chunks: int):
        self.total = total_chunks
        self.completed = 0
        self.chunk_times: List[float] = []
        self.start_time = time.time()
        self.current_chunk_start = 0.0

    def start_chunk(self):
        self.current_chunk_start = time.time()

    def finish_chunk(self):
        elapsed = time.time() - self.current_chunk_start
        self.chunk_times.append(elapsed)
        self.completed += 1

    def get_eta_seconds(self) -> float:
        if not self.chunk_times:
            return 0.0
        avg_time = sum(self.chunk_times) / len(self.chunk_times)
        remaining = self.total - self.completed
        return avg_time * remaining

    def get_progress_percent(self) -> float:
        if self.total == 0:
            return 100.0
        return (self.completed / self.total) * 100.0

    def get_stats(self) -> str:
        elapsed = time.time() - self.start_time
        eta = self.get_eta_seconds()
        pct = self.get_progress_percent()

        elapsed_str = self._format_time(elapsed)
        eta_str = self._format_time(eta)

        if self.chunk_times:
            avg_chunk = sum(self.chunk_times) / len(self.chunk_times)
            avg_str = f"{avg_chunk:.1f}s/chunk"
        else:
            avg_str = "calculating..."

        bar_len = 20
        filled = int(bar_len * pct / 100)
        bar = "█" * filled + "░" * (bar_len - filled)

        return f"[{bar}] {pct:.0f}% | {self.completed}/{self.total} chunks | Elapsed: {elapsed_str} | ETA: {eta_str} | {avg_str}"

    def _format_time(self, seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"


def generate_progress_html(progress_tracker: ProgressTracker, current_chunk: int = None) -> str:
    """Generates HTML progress bar in unified dark-purple theme."""
    pct = progress_tracker.get_progress_percent()
    stats = progress_tracker.get_stats()

    gradient = "linear-gradient(90deg, #667eea 0%, #764ba2 100%)"
    bg_dark = "#0f172a"
    bg_card = "#1e293b"
    border_color = "#334155"
    text_primary = "#e2e8f0"
    text_secondary = "#94a3b8"
    accent = "#667eea"

    if pct >= 100:
        status_color = "#10b981"
        status_icon = "✅"
    elif pct > 0:
        status_color = accent
        status_icon = "🔄"
    else:
        status_color = "#64748b"
        status_icon = "⏳"

    chunk_indicator = ""
    if current_chunk is not None and progress_tracker.total > 1:
        chunk_indicator = f'<span style="color:{accent}; font-size:13px; font-weight:600;">{status_icon} Chunk {current_chunk}/{progress_tracker.total}</span>'

    html = f"""
    <div style="width:100%; background:{bg_dark}; border-radius:12px; padding:16px; margin:8px 0; 
                border:1px solid {border_color}; font-family:'Segoe UI',system-ui,sans-serif;
                box-shadow:0 4px 6px rgba(0,0,0,0.3);">
        <div style="display:flex; justify-content:space-between; margin-bottom:10px; align-items:center;">
            <span style="color:{text_primary}; font-size:14px; font-weight:600;">🎙️ Generation Progress</span>
            <div style="display:flex; gap:12px; align-items:center;">
                {chunk_indicator}
                <span style="color:{text_secondary}; font-size:13px;">{pct:.0f}% complete</span>
            </div>
        </div>
        <div style="width:100%; height:24px; background:{bg_card}; border-radius:12px; overflow:hidden;
                    box-shadow:inset 0 2px 4px rgba(0,0,0,0.3);">
            <div style="width:{pct}%; height:100%; background:{gradient}; 
                        transition:width 0.5s cubic-bezier(0.4, 0, 0.2, 1);
                        border-radius:12px; position:relative;">
                <div style="position:absolute; right:0; top:0; bottom:0; width:30px; 
                            background:linear-gradient(90deg, transparent, rgba(255,255,255,0.3));"></div>
            </div>
        </div>
        <div style="display:flex; justify-content:space-between; margin-top:8px; flex-wrap:wrap; gap:8px;">
            <span style="color:{text_secondary}; font-size:12px; font-family:monospace;">{stats}</span>
            <span style="color:{status_color}; font-size:12px; font-weight:600;">
                {status_icon} {"Complete" if pct >= 100 else "In Progress" if pct > 0 else "Waiting"}
            </span>
        </div>
    </div>
    """
    return html


def generate_resume_html(checkpoints: List[Tuple[str, int, int, float]]) -> str:
    """Generates HTML list of available checkpoints in unified theme."""
    if not checkpoints:
        return "<div style='color:#64748b; padding:8px;'>No saved checkpoints found</div>"

    bg_card = "#1e293b"
    border_color = "#334155"
    text_primary = "#e2e8f0"
    text_secondary = "#94a3b8"
    accent = "#667eea"

    html = f"<div style='max-height:200px; overflow-y:auto;'>"
    for sid, completed, total, ts in checkpoints:
        pct = (completed / total * 100) if total > 0 else 0
        date = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
        progress_gradient = f"linear-gradient(90deg, rgba(102,126,234,0.3) {pct}%, transparent {pct}%)"

        html += f"""
        <div style="display:flex; justify-content:space-between; align-items:center; 
                    padding:8px 12px; margin:4px 0; background:{progress_gradient}, {bg_card}; 
                    border-radius:8px; border:1px solid {border_color}; cursor:pointer;
                    transition:all 0.2s ease;" 
             onmouseover="this.style.borderColor='{accent}'; this.style.transform='translateX(4px)'" 
             onmouseout="this.style.borderColor='{border_color}'; this.style.transform='translateX(0)'">
            <div>
                <div style="color:{text_primary}; font-size:13px;">Session {sid[:8]}...</div>
                <div style="color:{text_secondary}; font-size:11px;">{date} | {completed}/{total} chunks ({pct:.0f}%)</div>
            </div>
            <div style="color:{accent}; font-size:12px; font-weight:600;">RESUME →</div>
        </div>
        """
    html += "</div>"
    return html


def cleanup_checkpoints_folder():
    """Remove all files from checkpoints folder."""
    try:
        if not os.path.exists(CHECKPOINT_DIR):
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            return 0
        removed = 0
        for item in os.listdir(CHECKPOINT_DIR):
            item_path = os.path.join(CHECKPOINT_DIR, item)
            try:
                if os.path.isfile(item_path):
                    os.remove(item_path)
                    removed += 1
                elif os.path.isdir(item_path):
                    import shutil
                    shutil.rmtree(item_path)
                    removed += 1
            except Exception as e:
                logging.warning(f"[CheckpointCleanup] Failed to delete {item}: {e}")
        logging.info(f"[CheckpointCleanup] Cleared {removed} items from {CHECKPOINT_DIR}")
        return removed
    except Exception as e:
        logging.error(f"[CheckpointCleanup] Error: {e}")
        return 0


# SETTINGS MANAGEMENT

def load_settings() -> Dict[str, Any]:
    """Load settings from settings.txt."""
    defaults = {
        "normalize": True,
        "normalize_level": -20,
        "use_zipenhancer": True,
        "speed": 1.0,
        "num_step": 16,
        "guidance_scale": 2.0,
        "denoise": True,
        "preprocess_prompt": True,
        "postprocess_output": True,
        "output_format": "wav",
        "output_sample_rate": 48000,
        "bitrate": 320,
        "language": "Auto",
        "seed": -1,
        "random_seed": True,
        # Chunking settings
        "enable_chunking": True,
        "chunking_mode": "lines",
        "max_lines_per_chunk": 8,
        "max_sentences_per_chunk": 5,
        "max_chars_per_chunk": 800,
        "crossfade_ms": 50,
    }

    if not SETTINGS_FILE.exists():
        return defaults

    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            saved = json.load(f)
        # Merge with defaults
        for key, val in defaults.items():
            if key not in saved:
                saved[key] = val
        return saved
    except Exception as e:
        logging.warning(f"Failed to load settings: {e}")
        return defaults


def save_settings(settings: Dict[str, Any]):
    """Save settings to settings.txt."""
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
        logging.info(f"Settings saved to {SETTINGS_FILE}")
    except Exception as e:
        logging.error(f"Failed to save settings: {e}")


# AUDIO NORMALIZATION

def normalize_audio(waveform: np.ndarray, target_level_db: float) -> np.ndarray:
    """
    Normalize audio to target RMS level in dB.
    target_level_db: -5 to +5 dB relative to full scale.
    """
    if waveform.size == 0:
        return waveform

    # Convert to float
    if waveform.dtype == np.int16:
        waveform_float = waveform.astype(np.float32) / 32768.0
    else:
        waveform_float = waveform.astype(np.float32)

    # Calculate current RMS
    rms = np.sqrt(np.mean(waveform_float ** 2))
    if rms < 1e-10:
        return waveform

    # Target RMS from dB
    target_rms = 10 ** (target_level_db / 20.0)

    # Gain factor
    gain = target_rms / rms

    # Apply gain
    normalized = waveform_float * gain

    # Clip to prevent overflow
    normalized = np.clip(normalized, -1.0, 1.0)

    # Convert back to int16
    if waveform.dtype == np.int16:
        return (normalized * 32767).astype(np.int16)

    return normalized


class FullOffloadOmniVoice:
    def __init__(self, checkpoint: str):
        self.checkpoint = checkpoint
        self.dtype = torch.float16
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()

    def _load_model(self):
        logging.info("Loading model to CPU...")

        # Defining the path to the model
        if self.checkpoint.startswith("LeeAeron/") or "/" in self.checkpoint:
            # HuggingFace repo_id - use a local copy
            local_model_path = ensure_model_downloaded(self.checkpoint)
            load_path = str(local_model_path)
            logging.info(f"Using local model: {load_path}")
        else:
            # local path
            load_path = self.checkpoint
            logging.info(f"Using provided local path: {load_path}")

        # load from local_files_only
        self.model = OmniVoice.from_pretrained(
            load_path,
            device_map="cpu",
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            load_asr=True,
            local_files_only=True,
        )

        self.sampling_rate = self.model.sampling_rate
        logging.info("Model loaded to CPU")

        # After downloading, enable offline mode completely.
        os.environ["HF_HUB_OFFLINE"] = "1"
        logging.info("HF Hub offline mode on")

    def _to_gpu(self):
        if torch.cuda.is_available():
            self.model.to(self.device)
            torch.cuda.synchronize()

    def _to_cpu(self):
        self.model.to('cpu')
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()

    def create_voice_clone_prompt(self, ref_audio, ref_text=None):
        if isinstance(ref_audio, tuple):
            sr, data = ref_audio
            if data.dtype != np.float32:
                data = data.astype(np.float32)
            if data.max() > 1.0 or data.min() < -1.0:
                data = data / 32768.0 if data.dtype == np.int16 else data / np.abs(data).max()
            ref_audio = (sr, data)

        try:
            self._to_gpu()
            result = self.model.create_voice_clone_prompt(ref_audio, ref_text)
        except torch.cuda.OutOfMemoryError:
            logging.warning("High VRAM usage, moving to offload mode...")
            self._to_cpu()
            result = self.model.create_voice_clone_prompt(ref_audio, ref_text)
        finally:
            self._to_cpu()

        return result

    def generate(self, **kwargs):
        if not torch.cuda.is_available():
            return self.model.generate(**kwargs)

        try:
            self._to_gpu()
            result = self.model.generate(**kwargs)
        finally:
            self._to_cpu()

        return result


def restart_engine():
    """Complete shell restart without opening a new tab."""
    logging.info("Restarting application...")

    def delayed_restart():
        time.sleep(1)
        script_path = Path(__file__).absolute()
        python = sys.executable

        # Form arguments: remove --no-browser if present, add --no-restart-browser
        args = [str(script_path)]
        skip_next = False
        for i, arg in enumerate(sys.argv[1:]):
            if skip_next:
                skip_next = False
                continue
            if arg == "--no-browser":
                continue  # Remove --no-browser to prevent a new browser from opening upon restart.
            if arg.startswith("--no-browser="):
                continue
            args.append(arg)

        # Add a flag that this is a restart (do not open the browser)
        args.append("--internal-restart-flag")

        os.execl(python, python, *args)

    threading.Thread(target=delayed_restart, daemon=True).start()
    return "Restarting the server..."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="omnivoice-demo-full-offload")
    parser.add_argument("--model", default="LeeAeron/OmniVoice")
    parser.add_argument("--device", default=None)
    parser.add_argument("--ip", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--find-port", action="store_true", default=False,
                        help="Auto-find available port if default is taken")
    parser.add_argument("--root-path", default=None)
    parser.add_argument("--share", action="store_true", default=False)
    parser.add_argument("--no-browser", action="store_true", default=False,
                        help="Don't open the browser automatically")
    # Internal flag for restart
    parser.add_argument("--internal-restart-flag", action="store_true", default=False,
                        help=argparse.SUPPRESS)
    return parser


def build_demo(model: FullOffloadOmniVoice, checkpoint: str) -> gr.Blocks:
    sampling_rate = model.sampling_rate
    settings = load_settings()

    # Initialize ZipEnhancer if needed (lazy load on first use)
    zipenhancer = None

    # Initialize checkpoint manager
    checkpoint_mgr = CheckpointManager()

    def _set_seed(seed_value: int):
        """Set PyTorch random seed for reproducible generation."""
        if seed_value is not None and int(seed_value) >= 0:
            seed = int(seed_value)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)
            logging.info(f"Seed set to: {seed}")
            return seed
        else:
            # Random seed
            seed = random.randint(0, 2**32 - 1)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)
            logging.info(f"Random seed used: {seed}")
            return seed

    def _generate_single_chunk(
        model_instance: FullOffloadOmniVoice,
        text: str,
        language: str,
        ref_audio,
        instruct: str,
        num_step: int,
        guidance_scale: float,
        denoise: bool,
        speed: float,
        duration: float,
        preprocess_prompt: bool,
        postprocess_output: bool,
        mode: str,
        ref_text: str = None,
    ) -> np.ndarray:
        """Generate audio for a single text chunk."""

        effective_steps = min(int(num_step or 12), 25)

        gen_config = OmniVoiceGenerationConfig(
            num_step=effective_steps,
            guidance_scale=float(guidance_scale) if guidance_scale is not None else 3.0,
            denoise=bool(denoise) if denoise is not None else True,
            preprocess_prompt=bool(preprocess_prompt),
            postprocess_output=bool(postprocess_output),
        )

        lang = language if (language and language != "Auto") else None

        kw: Dict[str, Any] = dict(
            text=text.strip(), language=lang, generation_config=gen_config
        )

        if speed is not None and float(speed) != 1.0:
            kw["speed"] = float(speed)
        if duration is not None and float(duration) > 0:
            kw["duration"] = float(duration)

        if mode == "clone":
            if ref_audio:
                kw["voice_clone_prompt"] = model_instance.create_voice_clone_prompt(
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                )

        if mode == "design":
            if instruct and instruct.strip():
                kw["instruct"] = instruct.strip()

        try:
            audio = model_instance.generate(**kw)
            waveform = audio[0].squeeze(0).cpu().numpy().astype(np.float32)
            return waveform
        except Exception as e:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise e

    def _postprocess_audio(
        waveform: np.ndarray,
        current_sr: int,
        normalize: bool,
        normalize_level: float,
        use_zipenhancer: bool,
        target_sample_rate: int,
        bitrate: int,
        output_format: str,
    ) -> Tuple[np.ndarray, int, str, str]:
        """Apply post-processing: ZipEnhancer, normalization, save to file."""
        nonlocal zipenhancer

        # Apply ZipEnhancer if enabled (at 16kHz, BEFORE upsampling)
        if use_zipenhancer:
            if zipenhancer is None:
                zipenhancer_dir = ensure_zipenhancer_downloaded()
                zipenhancer = ZipEnhancerProcessor(zipenhancer_dir)

            temp_path = OUTPUTS_DIR / f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            sf.write(temp_path, waveform, current_sr)

            enhanced_path = str(temp_path.with_suffix('.enhanced.wav'))
            try:
                enhanced_path = zipenhancer.enhance(str(temp_path), enhanced_path)
                waveform, current_sr = sf.read(enhanced_path)

                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                if os.path.exists(enhanced_path):
                    os.unlink(enhanced_path)

                logging.info(f"ZipEnhancer output: {current_sr}Hz")
            except Exception as e:
                logging.error(f"ZipEnhancer failed, keeping original: {e}")
                if temp_path and os.path.exists(temp_path):
                    os.unlink(temp_path)

        # Apply normalization
        if normalize:
            waveform = normalize_audio(waveform, float(normalize_level))

        # Convert to int16 for ffmpeg
        if waveform.dtype != np.int16:
            waveform = (waveform * 32767).astype(np.int16)

        # Save via ffmpeg
        saved_path = save_audio_with_ffmpeg(
            waveform, 
            current_sr,
            output_format, 
            "omnivoice",
            target_sr=target_sample_rate,
            bitrate=bitrate
        )

        # Generate spectrogram
        try:
            spectrogram_path = save_spectrogram(waveform, current_sr)
        except Exception as e:
            logging.warning(f"Failed to generate spectrogram: {e}")
            spectrogram_path = None

        return waveform, current_sr, saved_path, spectrogram_path

    def _gen_core_chunked(
        text, language, ref_audio, instruct, num_step, guidance_scale, 
        denoise, speed, duration, preprocess_prompt, postprocess_output, 
        output_format, mode, ref_text=None,
        normalize=True, normalize_level=-20, use_zipenhancer=True,
        target_sample_rate=48000, bitrate=320, seed=-1, random_seed=True,
        # Chunking parameters
        enable_chunking=True,
        chunking_mode="lines",
        max_lines_per_chunk=8,
        max_sentences_per_chunk=5,
        max_chars_per_chunk=800,
        crossfade_ms=50,
    ) -> Iterator[tuple]:
        """Chunked generation with progress tracking and resume support."""

        if not text or not text.strip():
            yield None, "Enter text to be synthesized", None, None, ""
            return

        # Set seed before generation
        if random_seed:
            actual_seed = _set_seed(-1)
        else:
            actual_seed = _set_seed(seed)

        # Check if chunking should be used
        should_chunk = False
        if enable_chunking:
            if chunking_mode == "lines":
                total_lines = len([l for l in text.strip().split("\n") if l.strip()])
                should_chunk = total_lines > max_lines_per_chunk
            elif chunking_mode == "sentences":
                total_sentences = len(split_into_sentences(text))
                should_chunk = total_sentences > max_sentences_per_chunk
            elif chunking_mode == "characters":
                total_chars = len(text.strip())
                should_chunk = total_chars > max_chars_per_chunk

        # Single-pass generation (no chunking needed)
        if not should_chunk:
            try:
                waveform = _generate_single_chunk(
                    model, text, language, ref_audio, instruct,
                    num_step, guidance_scale, denoise, speed, duration,
                    preprocess_prompt, postprocess_output, mode, ref_text
                )

                current_sr = sampling_rate  # Model native rate

                # Post-process
                waveform, current_sr, saved_path, spectrogram_path = _postprocess_audio(
                    waveform, current_sr, normalize, normalize_level,
                    use_zipenhancer, target_sample_rate, bitrate, output_format
                )

                status_msg = f"Done! Saved in: {Path(saved_path).name} | Seed: {actual_seed}"
                yield (current_sr, waveform), status_msg, saved_path, spectrogram_path, ""

            except Exception as e:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                yield None, f"Error: {type(e).__name__}: {str(e)[:100]}", None, None, ""
            return

        # CHUNKED GENERATION
        chunker = ScriptChunker(
            chunking_mode=chunking_mode,
            max_lines_per_chunk=max_lines_per_chunk,
            max_sentences_per_chunk=max_sentences_per_chunk,
            max_chars_per_chunk=max_chars_per_chunk,
        )
        crossfader = AudioCrossfader(fade_duration_ms=crossfade_ms, sample_rate=sampling_rate)

        chunks = chunker.parse_text(text)
        total_chunks = len(chunks)

        if total_chunks == 0:
            yield None, "Error: Could not parse text into chunks", None, None, ""
            return

        # Initialize checkpoint for this session (internal, not exposed in UI)
        session_id = checkpoint_mgr.generate_session_id(text, mode, actual_seed)
        ckpt = GenerationCheckpoint(
            session_id=session_id,
            text_hash=hashlib.md5(text.encode()).hexdigest(),
            mode=mode,
            language=language,
            num_step=num_step,
            guidance_scale=guidance_scale,
            denoise=denoise,
            preprocess_prompt=preprocess_prompt,
            postprocess_output=postprocess_output,
            speed=speed,
            duration=duration,
            seed=actual_seed,
            total_chunks=total_chunks,
            completed_chunks=0,
            chunk_audio_files=[],
        )

        start_idx = 0
        all_chunk_audios = []

        # Progress tracker
        progress = ProgressTracker(total_chunks)

        status_msg = f"📦 Chunked generation: {total_chunks} chunks"
        progress_html = generate_progress_html(progress)
        yield None, status_msg, None, None, progress_html

        # Generate all chunks
        accumulated_audio = None

        for i, chunk in enumerate(chunks):
            chunk_idx = i
            progress.start_chunk()

            # Show "processing" status with progress BEFORE blocking call
            chunk_status = f"🔄 Processing chunk {chunk_idx + 1}/{total_chunks}..."
            progress_html = generate_progress_html(progress, chunk_idx + 1)
            yield None, chunk_status, None, None, progress_html
            time.sleep(0.05)  # Let Gradio render the update

            try:
                chunk_waveform = _generate_single_chunk(
                    model, chunk.text, language, ref_audio, instruct,
                    num_step, guidance_scale, denoise, speed, duration,
                    preprocess_prompt, postprocess_output, mode, ref_text
                )
            except Exception as e:
                logging.error(f"[Chunk {chunk_idx + 1}] Error: {e}")
                progress.finish_chunk()  # Count as completed (failed)
                chunk_status = f"❌ Chunk {chunk_idx + 1} failed: {e}"
                progress_html = generate_progress_html(progress, chunk_idx + 1)
                yield None, chunk_status, None, None, progress_html
                continue

            if chunk_waveform is None or len(chunk_waveform) == 0:
                progress.finish_chunk()  # Count as completed (empty)
                chunk_status = f"⚠️ Chunk {chunk_idx + 1} produced no audio"
                progress_html = generate_progress_html(progress, chunk_idx + 1)
                yield None, chunk_status, None, None, progress_html
                continue

            # Save chunk for resume
            chunk_filename = f"chunk_{session_id}_{chunk_idx:04d}.wav"
            chunk_path = CHECKPOINT_DIR / chunk_filename
            sf.write(chunk_path, chunk_waveform, sampling_rate, subtype='PCM_16')

            all_chunk_audios.append(chunk_waveform)
            ckpt.chunk_audio_files.append(str(chunk_path))
            ckpt.completed_chunks = len(all_chunk_audios)
            checkpoint_mgr.save(ckpt)

            progress.finish_chunk()

            # Incremental accumulation with crossfade
            chunk_float = chunk_waveform.astype(np.float32) if chunk_waveform.dtype != np.float32 else chunk_waveform
            if accumulated_audio is None:
                accumulated_audio = chunk_float
            else:
                accumulated_audio = crossfader.apply_crossfade(accumulated_audio, chunk_float)

            progress_html = generate_progress_html(progress, chunk_idx + 1)
            chunk_status = f"✅ Chunk {chunk_idx + 1}/{total_chunks}: {len(chunk_waveform)/sampling_rate:.1f}s | Total: {len(accumulated_audio)/sampling_rate:.1f}s"

            # Stream intermediate result
            audio_16bit = (np.clip(accumulated_audio, -1.0, 1.0) * 32767).astype(np.int16)
            yield (sampling_rate, audio_16bit), chunk_status, None, None, progress_html
            time.sleep(0.05)  # Let Gradio render the update

        # Final assembly
        if not all_chunk_audios or accumulated_audio is None or len(accumulated_audio) == 0:
            yield None, "❌ No audio generated", None, None, ""
            return

        final_duration = len(accumulated_audio) / sampling_rate

        # Session complete - checkpoints kept for reference, cleaned up periodically

        # Calculate timing stats
        total_generation_time = time.time() - progress.start_time
        realtime_factor = final_duration / total_generation_time if total_generation_time > 0 else 0
        avg_chunk_time = total_generation_time / total_chunks if total_chunks > 0 else 0

        # Final post-processing
        current_sr = sampling_rate
        if accumulated_audio.dtype != np.float32:
            accumulated_audio = accumulated_audio.astype(np.float32)

        # Apply ZipEnhancer and normalization to final accumulated audio
        final_waveform, current_sr, saved_path, spectrogram_path = _postprocess_audio(
            accumulated_audio, current_sr, normalize, normalize_level,
            use_zipenhancer, target_sample_rate, bitrate, output_format
        )

        final_status = f"🎉 Generation complete!\n"
        final_status += f"📦 Total chunks: {total_chunks}\n"
        final_status += f"⏱️ Audio duration: {final_duration:.1f}s\n"
        final_status += f"⏱️ Generation time: {total_generation_time:.1f}s\n"
        final_status += f"⚡ Real-time factor: {realtime_factor:.2f}x\n"
        final_status += f"🎲 Seed: {actual_seed}\n"
        final_status += f"💾 Saved: {Path(saved_path).name}"

        yield (current_sr, final_waveform), final_status, saved_path, spectrogram_path, ""

    def _gen_core(text, language, ref_audio, instruct, num_step, guidance_scale, 
                  denoise, speed, duration, preprocess_prompt, postprocess_output, 
                  output_format, mode, ref_text=None,
                  normalize=True, normalize_level=-20, use_zipenhancer=True,
                  target_sample_rate=48000, bitrate=320, seed=-1, random_seed=True,
                  enable_chunking=True, chunking_mode="lines",
                  max_lines_per_chunk=8, max_sentences_per_chunk=5,
                  max_chars_per_chunk=800, crossfade_ms=50):
        """Generator that yields updates for Gradio streaming."""

        for audio, status, saved_file, spectrogram, progress_html in _gen_core_chunked(
            text, language, ref_audio, instruct, num_step, guidance_scale,
            denoise, speed, duration, preprocess_prompt, postprocess_output,
            output_format, mode, ref_text,
            normalize, normalize_level, use_zipenhancer,
            target_sample_rate, bitrate, seed, random_seed,
            enable_chunking, chunking_mode,
            max_lines_per_chunk, max_sentences_per_chunk,
            max_chars_per_chunk, crossfade_ms
        ):
            # progress_html is either HTML content or empty string
            # Just pass the HTML string directly - component is always visible now
            yield audio, status, saved_file, spectrogram, progress_html

    

    theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.Color(
            name="indigo",
            c50="#eef2ff",
            c100="#e0e7ff",
            c200="#c7d2fe",
            c300="#a5b4fc",
            c400="#818cf8",
            c500="#667eea",
            c600="#5b6fd6",
            c700="#4f5fbf",
            c800="#444fa8",
            c900="#3a3f91",
            c950="#2d2f6e",
        ),
        secondary_hue=gr.themes.colors.Color(
            name="purple",
            c50="#faf5ff",
            c100="#f3e8ff",
            c200="#e9d5ff",
            c300="#d8b4fe",
            c400="#c084fc",
            c500="#a855f7",
            c600="#9333ea",
            c700="#7e22ce",
            c800="#6b21a8",
            c900="#581c87",
            c950="#3b0764",
        ),
        neutral_hue=gr.themes.colors.Color(
            name="slate",
            c50="#f8fafc",
            c100="#f1f5f9",
            c200="#e2e8f0",
            c300="#cbd5e1",
            c400="#94a3b8",
            c500="#64748b",
            c600="#475569",
            c700="#334155",
            c800="#1e293b",
            c900="#0f172a",
            c950="#020617",
        ),
        font=["Inter", "Arial", "sans-serif"],
        font_mono=["ui-monospace", "Consolas", "monospace"],
    )

    css = """
    /* === LAYOUT UTILITIES === */
    .square-btn {width: 40px !important; min-width: 40px !important; padding: 0 !important; height: 40px !important;}
    .voice-controls-row {align-items: center !important;}
    .voice-controls-row button {height: 40px !important; margin-top: 24px !important;}
    .generate-btn-row {margin-bottom: 10px !important;}
    .seed-row {align-items: center !important; gap: 8px !important;}
    .seed-row button {height: 40px !important; margin-top: 24px !important; min-width: 40px !important;}
    .seed-row .form {margin-bottom: 0 !important;}
    .chunking-row {align-items: center !important; gap: 8px !important;}
    .chunking-row .form {margin-bottom: 0 !important;}

    /* === SCROLLBAR === */
    ::-webkit-scrollbar {width: 8px; height: 8px;}
    ::-webkit-scrollbar-track {background: #0f172a;}
    ::-webkit-scrollbar-thumb {background: linear-gradient(#667eea, #764ba2); border-radius: 4px;}
    ::-webkit-scrollbar-thumb:hover {background: #667eea;}
    """

    # JavaScript to reload the current tab after server restart
    js_reload_on_ready = """
    () => {
        const statusText = document.querySelector('textarea[data-testid="textbox"]');
        if (statusText) {
            statusText.value = "⏳ Waiting for server restart...";
            statusText.dispatchEvent(new Event('input'));
        }

        const checkServer = async () => {
            try {
                const resp = await fetch(window.location.origin, {cache: "no-store"});
                if (resp.ok) {
                    window.location.reload();
                } else {
                    setTimeout(checkServer, 800);
                }
            } catch (e) {
                setTimeout(checkServer, 800);
            }
        };
        setTimeout(checkServer, 1500);
    }
    """

    def _lang_dropdown(label="Language", value="Auto"):
        return gr.Dropdown(label=label, choices=_ALL_LANGUAGES, value=value, allow_custom_value=False, interactive=True)

    def _gen_settings():
        with gr.Accordion("⚙️ Settings", open=True):
            sp = gr.Slider(0.5, 1.5, value=settings.get("speed", 1.0), step=0.05, label="Speed")
            du = gr.Number(value=None, label="Duration (seconds)", info="Leave blank to use speed settings")
            ns = gr.Slider(4, 25, value=settings.get("num_step", 16), step=1, label="Inference Steps", info="12-16 optimal")
            gs = gr.Slider(1.0, 4.0, value=settings.get("guidance_scale", 2.0), step=0.1, label="Guidance Scale (CFG)")
            dn = gr.Checkbox(label="Denoise", value=settings.get("denoise", True))
            pp = gr.Checkbox(label="Preprocess Prompt", value=settings.get("preprocess_prompt", True))
            po = gr.Checkbox(label="Postprocess Output", value=settings.get("postprocess_output", True))

            # Seed controls
            with gr.Row(elem_classes="seed-row"):
                seed_input = gr.Number(
                    value=settings.get("seed", -1),
                    label="Seed",
                    info="-1 = random, 0+ = fixed",
                    precision=0,
                    scale=2
                )
                random_seed_cb = gr.Checkbox(
                    label="Random Seed",
                    value=settings.get("random_seed", True),
                    scale=1
                )
                random_seed_btn = gr.Button(
                    "🎲",
                    elem_classes="square-btn",
                    scale=0,
                    min_width=40
                )

            with gr.Row():
                fmt = gr.Dropdown(
                    choices=AUDIO_FORMATS,
                    value=settings.get("output_format", "wav"),
                    label="Output Format",
                    scale=1
                )
                sr = gr.Dropdown(
                    choices=OUTPUT_SAMPLE_RATES,
                    value=settings.get("output_sample_rate", 48000),
                    label="Output Sample Rate (Hz)",
                    scale=1
                )

            # Bitrate only shown for lossy formats
            br = gr.Dropdown(
                choices=BITRATE_OPTIONS,
                value=settings.get("bitrate", 320),
                label="Bitrate (kbps) - for MP3/OGG/AAC",
                visible=True
            )

        return ns, gs, sp, du, dn, pp, po, fmt, sr, br, seed_input, random_seed_cb, random_seed_btn

    def _chunking_settings():
        """Chunking settings UI - ported from VibeVoice."""
        with gr.Accordion("📦 Smart Chunking", open=True):
            enable_chunking = gr.Checkbox(
                label="Enable Smart Chunking",
                value=settings.get("enable_chunking", True),
                info="Split long text into chunks to prevent quality degradation. REQUIRED for long texts."
            )

            chunking_mode = gr.Dropdown(
                choices=[
                    ("Lines", "lines"),
                    ("Sentences (. ! ? …)", "sentences"),
                    ("Characters", "characters")
                ],
                value=settings.get("chunking_mode", "lines"),
                label="Chunking Mode",
                info="How to split the text into chunks"
            )

            # Use Tabs for reliable rendering of mode-specific sliders
            with gr.Tabs() as chunking_tabs:
                with gr.TabItem("Lines", id="lines"):
                    max_lines = gr.Slider(
                        minimum=1, maximum=20, value=settings.get("max_lines_per_chunk", 8), step=1,
                        label="Max Lines Per Chunk",
                        info="Lower = more stable voice, but more chunks"
                    )

                with gr.TabItem("Sentences", id="sentences"):
                    max_sentences = gr.Slider(
                        minimum=1, maximum=20, value=settings.get("max_sentences_per_chunk", 5), step=1,
                        label="Max Sentences Per Chunk",
                        info="Sentences end with . ! ? …"
                    )

                with gr.TabItem("Characters", id="characters"):
                    max_chars = gr.Slider(
                        minimum=100, maximum=3000, value=settings.get("max_chars_per_chunk", 800), step=50,
                        label="Max Characters Per Chunk",
                        info="Approximate character limit per chunk"
                    )

            crossfade_ms = gr.Slider(
                minimum=50, maximum=200, value=settings.get("crossfade_ms", 50), step=10,
                label="Crossfade Duration (ms)",
                info="Smooth transition between chunks. 50ms is good for most cases."
            )



        return (enable_chunking, chunking_mode, chunking_tabs, max_lines, max_sentences, max_chars,
                crossfade_ms)

    # Dark theme style injection via HTML
    dark_theme_css = """
    <style id="omnivoice-dark-theme">
    /* === DARK THEME OVERRIDES === */
    /* These styles are injected into the page and override Gradio's defaults */

    html, body {
        background: #0f172a !important;
        color: #e2e8f0 !important;
    }

    /* Main app container */
    #app, .wrap, .gradio-container, [data-testid="block"] {
        background: #0f172a !important;
    }

    /* All block/panel containers */
    .block, .panel, .form, .component-wrapper, .container,
    [class*="svelte-"][class*="block"],
    [class*="svelte-"][class*="panel"] {
        background: #1e293b !important;
        border-color: #334155 !important;
    }

    /* Input elements */
    input, textarea, select,
    [class*="svelte-"][class*="input"],
    [class*="svelte-"][class*="textbox"],
    [class*="svelte-"][class*="number"],
    [class*="svelte-"][class*="dropdown"] {
        background: #1e293b !important;
        border: 1px solid #334155 !important;
        color: #e2e8f0 !important;
        caret-color: #667eea !important;
    }

    input::placeholder, textarea::placeholder {
        color: #64748b !important;
    }

    input:focus, textarea:focus, select:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 2px rgba(102,126,234,0.2) !important;
    }

    /* Buttons - primary gradient */
    button[class*="primary"], [class*="svelte-"][class*="primary"] {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
    }

    button[class*="primary"]:hover {
        filter: brightness(1.1) !important;
        box-shadow: 0 4px 12px rgba(102,126,234,0.4) !important;
    }

    /* Buttons - secondary */
    button[class*="secondary"], [class*="svelte-"][class*="secondary"] {
        background: #1e293b !important;
        border: 1px solid #334155 !important;
        color: #94a3b8 !important;
    }

    button[class*="secondary"]:hover {
        background: #334155 !important;
        color: #e2e8f0 !important;
    }

    /* Tabs */
    [class*="tab-nav"], [class*="tabs"] {
        background: #1e293b !important;
        border-bottom: 1px solid #334155 !important;
    }

    [role="tab"], button[role="tab"], [class*="tab"] {
        color: #94a3b8 !important;
        background: transparent !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
    }

    [role="tab"][aria-selected="true"], [class*="tab"][class*="selected"] {
        color: #667eea !important;
        border-bottom: 2px solid #667eea !important;
        background: rgba(102,126,234,0.1) !important;
    }

    [class*="tabitem"], [role="tabpanel"] {
        background: #0f172a !important;
    }

    /* Accordions */
    [class*="accordion"], details, summary {
        background: #1e293b !important;
        border-color: #334155 !important;
    }

    [class*="accordion-header"], summary {
        color: #e2e8f0 !important;
    }

    /* Sliders */
    input[type="range"] {
        accent-color: #667eea !important;
    }

    /* Checkboxes and radios */
    input[type="checkbox"], input[type="radio"] {
        accent-color: #667eea !important;
    }

    /* Labels */
    label, [class*="label"], [class*="title"] {
        color: #94a3b8 !important;
    }

    [class*="block-title"], [class*="main-title"] {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
    }

    /* Audio player */
    audio {
        filter: hue-rotate(220deg) saturate(1.5) !important;
    }

    /* File upload */
    [class*="upload"], [class*="dropzone"] {
        background: #1e293b !important;
        border: 2px dashed #334155 !important;
        color: #94a3b8 !important;
    }

    [class*="upload"]:hover, [class*="dropzone"]:hover {
        border-color: #667eea !important;
    }

    /* Dropdown options */
    [class*="options"], [class*="dropdown-options"], [role="listbox"] {
        background: #1e293b !important;
        border: 1px solid #334155 !important;
    }

    [class*="options"] li, [role="option"] {
        color: #e2e8f0 !important;
    }

    [class*="options"] li:hover, [role="option"]:hover {
        background: rgba(102,126,234,0.2) !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {width: 8px; height: 8px;}
    ::-webkit-scrollbar-track {background: #0f172a;}
    ::-webkit-scrollbar-thumb {background: linear-gradient(#667eea, #764ba2); border-radius: 4px;}
    ::-webkit-scrollbar-thumb:hover {background: #667eea;}

    /* Remove any orange/yellow artifacts */
    [style*="orange"], [style*="#ff8c00"], [style*="#ffa500"] {
        background: #1e293b !important;
        border-color: #334155 !important;
        color: #e2e8f0 !important;
    }
    </style>
    """

    with gr.Blocks(title="OmniVoice Portable", css=css, theme=theme) as demo:
        gr.Markdown("<div align='center'><h1>OmniVoice Portable</h1></div>")

        if torch.cuda.is_available():
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            gr.Markdown(f'<div style="padding:10px;border-radius:5px;">🟢 GPU: {torch.cuda.get_device_name(0)} VRAM: {vram_total:.1f}GB</div>')

        with gr.Tabs():
            # VOICE CLONE
            with gr.TabItem("Voice Clone"):
                with gr.Row():
                    with gr.Column(scale=1):
                        vc_text = gr.Textbox(label="Text", lines=5, placeholder="Enter text for voiceover...")

                        # Paste, Clear, Copy buttons
                        with gr.Row():
                            vc_clear_btn = gr.Button("🗑️ Clear", scale=1)
                            vc_paste_btn = gr.Button("📋 Paste", scale=1)
                            vc_copy_btn = gr.Button("📄 Copy", scale=1)

                        with gr.Row(elem_classes="voice-controls-row"):
                            vc_voice_selector = gr.Dropdown(
                                choices=get_voice_choices(),
                                label="Pre-defined voices (reference_audio folder)",
                                value="-NONE-",
                                scale=1
                            )

                        with gr.Row(elem_classes="voice-controls-row"):                        
                            vc_refresh_btn = gr.Button("🔄 Refresh voices", elem_classes="square-btn", scale=1)
                            vc_download_btn = gr.Button("⬇️ Download voices", scale=1)

                        vc_ref_audio = gr.Audio(label="Reference Audio", type="filepath")
                        vc_ref_text = gr.Textbox(label="Reference Text", lines=1, 
                                                  placeholder="Leave empty for automatic transcription")
                        vc_lang = _lang_dropdown(value=settings.get("language", "Auto"))
                        vc_ns, vc_gs, vc_sp, vc_du, vc_dn, vc_pp, vc_po, vc_fmt, vc_sr, vc_br, vc_seed, vc_random_seed, vc_random_btn = _gen_settings()

                        # Chunking settings
                        (vc_enable_chunking, vc_chunking_mode, vc_chunking_tabs, vc_max_lines, vc_max_sentences, 
                         vc_max_chars, vc_crossfade_ms) = _chunking_settings()

                        # Audio Processing controls
                        with gr.Accordion("🔊 Audio Processing", open=True):
                            vc_normalize = gr.Checkbox(
                                label="Normalize Audio", 
                                value=settings.get("normalize", False)
                            )
                            vc_norm_level = gr.Dropdown(
                                label="Normalization Level (dB)",
                                choices=NORMALIZATION_LEVELS,
                                value=settings.get("normalize_level", -15),
                                interactive=True
                            )
                            vc_use_zipenhancer = gr.Checkbox(
                                label="Use ZipEnhancer (Speech Restoration)", 
                                value=settings.get("use_zipenhancer", True),
                                info="Enhance output audio quality using AI"
                            )

                    with gr.Column(scale=1):
                        with gr.Row(elem_classes="generate-btn-row"):
                            vc_btn = gr.Button("🚀 Generate", variant="primary", scale=3)
                            vc_restart_btn = gr.Button("🔄 Restart UI", variant="secondary", scale=1)

                        # Progress bar
                        vc_progress = gr.HTML(value="", visible=True)

                        vc_audio = gr.Audio(label="Output", type="numpy")
                        vc_spectrogram = gr.Image(label="Spectrogram", type="filepath")
                        vc_status = gr.Textbox(label="Status", lines=2)
                        vc_saved_file = gr.File(label="Saved File", visible=True)

                # Button click handlers
                vc_clear_btn.click(lambda: "", inputs=None, outputs=vc_text)

                vc_paste_btn.click(
                    None, inputs=None, outputs=vc_text,
                    js="""
                    async () => {
                        try {
                            const text = await navigator.clipboard.readText();
                            return text;
                        } catch (err) {
                            alert("Error accessing clipboard: " + err);
                            return "";
                        }
                    }
                    """
                )

                vc_copy_btn.click(
                    None, inputs=[vc_text], outputs=None,
                    js="(text) => navigator.clipboard.writeText(text)"
                )

                vc_refresh_btn.click(
                    lambda: gr.update(choices=get_voice_choices(), value="-NONE-"),
                    inputs=None, outputs=vc_voice_selector,
                )

                vc_download_btn.click(
                    lambda: download_reference_voices(),
                    inputs=None, outputs=vc_voice_selector,
                )

                vc_voice_selector.change(
                    lambda: (None, ""),
                    inputs=None, outputs=[vc_ref_audio, vc_ref_text],
                ).then(
                    set_voice_file,
                    inputs=[vc_voice_selector],
                    outputs=[vc_ref_audio],
                )

                # Random seed button handler
                def _random_seed():
                    new_seed = random.randint(0, 2**32 - 1)
                    return gr.update(value=new_seed)

                vc_random_btn.click(_random_seed, inputs=None, outputs=vc_seed)

                # Chunking mode -> switch tabs
                vc_chunking_mode.change(
                    fn=lambda mode: gr.update(selected=mode),
                    inputs=vc_chunking_mode,
                    outputs=vc_chunking_tabs
                )

                def _clone_fn(text, lang, ref_aud, ref_text, ns, gs, dn, sp, du, pp, po, fmt, sr, br, seed, random_seed,
                             normalize, norm_level, use_zipenhancer,
                             enable_chunking, chunking_mode, max_lines, max_sentences, max_chars,
                             crossfade_ms):
                    # Save settings before generation
                    current_settings = {
                        "speed": sp,
                        "num_step": ns,
                        "guidance_scale": gs,
                        "denoise": dn,
                        "preprocess_prompt": pp,
                        "postprocess_output": po,
                        "output_format": fmt,
                        "output_sample_rate": sr,
                        "bitrate": br,
                        "language": lang,
                        "seed": seed,
                        "random_seed": random_seed,
                        "normalize": normalize,
                        "normalize_level": norm_level,
                        "use_zipenhancer": use_zipenhancer,
                        "enable_chunking": enable_chunking,
                        "chunking_mode": chunking_mode,
                        "max_lines_per_chunk": max_lines,
                        "max_sentences_per_chunk": max_sentences,
                        "max_chars_per_chunk": max_chars,
                        "crossfade_ms": crossfade_ms,
                    }
                    save_settings(current_settings)

                    yield from _gen_core(text, lang, ref_aud, None, ns, gs, dn, sp, du, pp, po, fmt,
                                   mode="clone", ref_text=ref_text or None,
                                   normalize=normalize, normalize_level=norm_level,
                                   use_zipenhancer=use_zipenhancer,
                                   target_sample_rate=sr, bitrate=br,
                                   seed=seed, random_seed=random_seed,
                                   enable_chunking=enable_chunking,
                                   chunking_mode=chunking_mode,
                                   max_lines_per_chunk=max_lines,
                                   max_sentences_per_chunk=max_sentences,
                                   max_chars_per_chunk=max_chars,
                                   crossfade_ms=crossfade_ms)

                vc_btn.click(_clone_fn,
                    inputs=[vc_text, vc_lang, vc_ref_audio, vc_ref_text, vc_ns, vc_gs, vc_dn, vc_sp, vc_du, vc_pp, vc_po, vc_fmt, vc_sr, vc_br, vc_seed, vc_random_seed,
                           vc_normalize, vc_norm_level, vc_use_zipenhancer,
                           vc_enable_chunking, vc_chunking_mode, vc_max_lines, vc_max_sentences, vc_max_chars,
                           vc_crossfade_ms],
                    outputs=[vc_audio, vc_status, vc_saved_file, vc_spectrogram, vc_progress])

                vc_restart_btn.click(
                    restart_engine,
                    inputs=None,
                    outputs=vc_status,
                    js=js_reload_on_ready
                )

            # VOICE DESIGN
            with gr.TabItem("Voice Design"):
                with gr.Row():
                    with gr.Column(scale=1):
                        vd_text = gr.Textbox(label="Text", lines=5, placeholder="Enter text for voiceover...")

                        # Paste, Clear, Copy buttons
                        with gr.Row():
                            vd_clear_btn = gr.Button("🗑️ Clear", scale=1)
                            vd_paste_btn = gr.Button("📋 Paste", scale=1)
                            vd_copy_btn = gr.Button("📄 Copy", scale=1)

                        vd_lang = _lang_dropdown(value=settings.get("language", "Auto"))

                        with gr.Row(elem_classes="voice-controls-row"):
                            vd_voice_selector = gr.Dropdown(
                                choices=get_voice_choices(),
                                label="Pre-defined voices (reference_audio folder)",
                                value="-NONE-",
                                scale=1
                            )

                        with gr.Row(elem_classes="voice-controls-row"):
                            vd_refresh_btn = gr.Button("🔄 Refresh voices", elem_classes="square-btn", scale=1)
                            vd_download_btn = gr.Button("⬇️  Download voices", scale=1)

                        vd_ref_audio = gr.Audio(label="Reference Audio (optional)", type="filepath")

                        vd_groups = []
                        for _cat, _choices in _CATEGORIES.items():
                            vd_groups.append(gr.Dropdown(label=_cat, choices=["Auto"] + _choices, value="Auto"))

                        vd_ns, vd_gs, vd_sp, vd_du, vd_dn, vd_pp, vd_po, vd_fmt, vd_sr, vd_br, vd_seed, vd_random_seed, vd_random_btn = _gen_settings()

                        # Chunking settings
                        (vd_enable_chunking, vd_chunking_mode, vd_chunking_tabs, vd_max_lines, vd_max_sentences, 
                         vd_max_chars, vd_crossfade_ms) = _chunking_settings()

                        # Audio Processing controls
                        with gr.Accordion("🔊 Audio Processing", open=True):
                            vd_normalize = gr.Checkbox(
                                label="Normalize Audio", 
                                value=settings.get("normalize", True)
                            )
                            vd_norm_level = gr.Dropdown(
                                label="Normalization Level (dB)",
                                choices=NORMALIZATION_LEVELS,
                                value=settings.get("normalize_level", -15),
                                interactive=True
                            )
                            vd_use_zipenhancer = gr.Checkbox(
                                label="Use ZipEnhancer (Speech Restoration)", 
                                value=settings.get("use_zipenhancer", True),
                                info="Enhance output audio quality using AI"
                            )

                    with gr.Column(scale=1):
                        with gr.Row(elem_classes="generate-btn-row"):
                            vd_btn = gr.Button("🚀 Generate", variant="primary", scale=3)
                            vd_restart_btn = gr.Button("🔄 Restart UI", variant="secondary", scale=1)

                        # Progress bar
                        vd_progress = gr.HTML(value="", visible=True)

                        vd_audio = gr.Audio(label="Output", type="numpy")
                        vd_spectrogram = gr.Image(label="Spectrogram", type="filepath")
                        vd_status = gr.Textbox(label="Status", lines=2)
                        vd_saved_file = gr.File(label="Saved File", visible=True)

                # Button click handlers
                vd_clear_btn.click(lambda: "", inputs=None, outputs=vd_text)

                vd_paste_btn.click(
                    None, inputs=None, outputs=vd_text,
                    js="""
                    async () => {
                        try {
                            const text = await navigator.clipboard.readText();
                            return text;
                        } catch (err) {
                            alert("Error accessing clipboard: " + err);
                            return "";
                        }
                    }
                    """
                )

                vd_copy_btn.click(
                    None, inputs=[vd_text], outputs=None,
                    js="(text) => navigator.clipboard.writeText(text)"
                )

                vd_refresh_btn.click(
                    lambda: gr.update(choices=get_voice_choices(), value="-NONE-"),
                    inputs=None, outputs=vd_voice_selector,
                )

                vd_download_btn.click(
                    lambda: download_reference_voices(),
                    inputs=None, outputs=vd_voice_selector,
                )

                vd_voice_selector.change(
                    lambda: None,
                    inputs=None, outputs=[vd_ref_audio],
                ).then(
                    set_voice_file,
                    inputs=[vd_voice_selector],
                    outputs=[vd_ref_audio],
                )

                # Random seed button handler
                vd_random_btn.click(_random_seed, inputs=None, outputs=vd_seed)

                # Chunking mode -> switch tabs
                vd_chunking_mode.change(
                    fn=lambda mode: gr.update(selected=mode),
                    inputs=vd_chunking_mode,
                    outputs=vd_chunking_tabs
                )

                def _build_instruct(groups):
                    selected = [g for g in groups if g and g != "Auto"]
                    if not selected:
                        return None
                    parts = []
                    for v in selected:
                        if " / " in v:
                            ru_part = v.split(" / ", 1)[0].strip()
                            parts.append(ru_part)
                        else:
                            parts.append(v)
                    return ", ".join(parts)

                def _design_fn(text, lang, ref_aud, ns, gs, dn, sp, du, pp, po, fmt, sr, br, seed, random_seed, *args):
                    # Separate groups from audio processing and chunking args
                    num_groups = len(vd_groups)
                    groups = args[:num_groups]
                    normalize = args[num_groups]
                    norm_level = args[num_groups + 1]
                    use_zipenhancer = args[num_groups + 2]
                    enable_chunking = args[num_groups + 3]
                    chunking_mode = args[num_groups + 4]
                    max_lines = args[num_groups + 5]
                    max_sentences = args[num_groups + 6]
                    max_chars = args[num_groups + 7]
                    crossfade_ms = args[num_groups + 8]

                    # Save settings before generation
                    current_settings = {
                        "speed": sp,
                        "num_step": ns,
                        "guidance_scale": gs,
                        "denoise": dn,
                        "preprocess_prompt": pp,
                        "postprocess_output": po,
                        "output_format": fmt,
                        "output_sample_rate": sr,
                        "bitrate": br,
                        "language": lang,
                        "seed": seed,
                        "random_seed": random_seed,
                        "normalize": normalize,
                        "normalize_level": norm_level,
                        "use_zipenhancer": use_zipenhancer,
                        "enable_chunking": enable_chunking,
                        "chunking_mode": chunking_mode,
                        "max_lines_per_chunk": max_lines,
                        "max_sentences_per_chunk": max_sentences,
                        "max_chars_per_chunk": max_chars,
                        "crossfade_ms": crossfade_ms,
                    }
                    save_settings(current_settings)

                    instruct = _build_instruct(groups)

                    if ref_aud:
                        yield from _gen_core(text, lang, ref_aud, None, ns, gs, dn, sp, du, pp, po, fmt,
                                       mode="clone", ref_text=None,
                                       normalize=normalize, normalize_level=norm_level,
                                       use_zipenhancer=use_zipenhancer,
                                       target_sample_rate=sr, bitrate=br,
                                       seed=seed, random_seed=random_seed,
                                       enable_chunking=enable_chunking,
                                       chunking_mode=chunking_mode,
                                       max_lines_per_chunk=max_lines,
                                       max_sentences_per_chunk=max_sentences,
                                       max_chars_per_chunk=max_chars,
                                       crossfade_ms=crossfade_ms)
                    else:
                        yield from _gen_core(text, lang, None, instruct, ns, gs, dn, sp, du, pp, po, fmt, 
                                       mode="design",
                                       normalize=normalize, normalize_level=norm_level,
                                       use_zipenhancer=use_zipenhancer,
                                       target_sample_rate=sr, bitrate=br,
                                       seed=seed, random_seed=random_seed,
                                       enable_chunking=enable_chunking,
                                       chunking_mode=chunking_mode,
                                       max_lines_per_chunk=max_lines,
                                       max_sentences_per_chunk=max_sentences,
                                       max_chars_per_chunk=max_chars,
                                       crossfade_ms=crossfade_ms)

                vd_btn.click(_design_fn,
                    inputs=[vd_text, vd_lang, vd_ref_audio, vd_ns, vd_gs, vd_dn, vd_sp, vd_du, vd_pp, vd_po, vd_fmt, vd_sr, vd_br, vd_seed, vd_random_seed] + 
                           vd_groups + [vd_normalize, vd_norm_level, vd_use_zipenhancer,
                           vd_enable_chunking, vd_chunking_mode, vd_max_lines, vd_max_sentences, vd_max_chars,
                           vd_crossfade_ms],
                    outputs=[vd_audio, vd_status, vd_saved_file, vd_spectrogram, vd_progress])

                vd_restart_btn.click(
                    restart_engine,
                    inputs=None,
                    outputs=vd_status,
                    js=js_reload_on_ready
                )

        gr.Markdown(f"""
                💾 Saving in: `{OUTPUTS_DIR}`
                🎤 References in: `{REFERENCE_AUDIO_DIR}`
                💾 Checkpoints in: `{CHECKPOINT_DIR}`
                """)

        gr.Markdown(
            """
            <div align='center'>
                <a href="https://github.com/LeeAeron/OmniVoice" 
                   target="_blank" 
                   style="font-size:14px;">
                   GitHub: OmniVoice
                </a>
            </div>
            """
        )

    return demo


def find_free_port(start_port: int = 7860, max_attempts: int = 20) -> int:
    """Find first available port starting from start_port."""
    import socket
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free ports found in range {start_port}-{start_port + max_attempts}")


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    parser = build_parser()
    args = parser.parse_args(argv)

    device = args.device or get_best_device()

    # Checking the restart flag (hidden from the user)
    is_restart = args.internal_restart_flag

    logging.info("=" * 60)
    logging.info(f"OmniVoice Demo Starting...")
    logging.info(f"Current Working Directory: {CWD}")
    logging.info(f"Script location: {Path(__file__).absolute()}")
    logging.info(f"Outputs will be saved to: {OUTPUTS_DIR}")
    logging.info(f"Reference audio folder: {REFERENCE_AUDIO_DIR}")
    logging.info(f"Models folder: {MODELS_DIR}")
    logging.info(f"Checkpoints folder: {CHECKPOINT_DIR}")
    logging.info(f"Model: {args.model}")
    logging.info(f"Target sample rate: {TARGET_SAMPLE_RATE}Hz")
    logging.info(f"Is restart: {is_restart}")
    logging.info(f"Auto-open browser: {not args.no_browser}")
    logging.info("=" * 60)

    # Checkpoints are managed internally, no UI exposure for resume/cleanup

    model = FullOffloadOmniVoice(checkpoint=args.model)

    demo = build_demo(model, args.model)

    # Open the browser only if:
    # 1. --no-browser not specified
    # 2. This is not a restart (is_restart=False)
    should_open_browser = (not args.no_browser) and (not is_restart)

    logging.info(f"Opening browser: {should_open_browser}")

    # Auto-find port if requested or if default is taken
    launch_port = args.port
    if args.find_port:
        try:
            launch_port = find_free_port(args.port)
            if launch_port != args.port:
                logging.info(f"Port {args.port} busy, using {launch_port}")
        except RuntimeError as e:
            logging.error(str(e))
            return 1

    demo.queue().launch(
        server_name=args.ip, 
        server_port=launch_port, 
        share=args.share, 
        root_path=args.root_path,
        inbrowser=should_open_browser
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())