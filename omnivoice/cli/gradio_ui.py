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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


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

OUTPUTS_DIR.mkdir(exist_ok=True)
REFERENCE_AUDIO_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

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
            cmd.extend(["-c:a", "libvorbis", "-q:a", "4"])  # Vorbis uses quality, not bitrate directly
            # Alternative: use opus for better quality control
            # cmd.extend(["-c:a", "libopus", "-b:a", f"{bitrate}k"])
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

    def _gen_core(text, language, ref_audio, instruct, num_step, guidance_scale, 
                  denoise, speed, duration, preprocess_prompt, postprocess_output, 
                  output_format, mode, ref_text=None,
                  normalize=True, normalize_level=-20, use_zipenhancer=True,
                  target_sample_rate=48000, bitrate=320, seed=-1, random_seed=True):

        if not text or not text.strip():
            return None, "Enter text to be synthesized", None, None

        # Set seed before generation
        if random_seed:
            actual_seed = _set_seed(-1)
        else:
            actual_seed = _set_seed(seed)

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
            if not ref_audio:
                return None, "Upload reference audio", None, None
            kw["voice_clone_prompt"] = model.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text,
            )

        if mode == "design":
            if instruct and instruct.strip():
                kw["instruct"] = instruct.strip()

        try:
            audio = model.generate(**kw)
        except Exception as e:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None, f"Error: {type(e).__name__}: {str(e)[:100]}", None, None

        # Start with float32 for processing chain
        waveform = audio[0].squeeze(0).cpu().numpy().astype(np.float32)

        # Track the actual sample rate of the audio data
        current_sr = sampling_rate  # Starts at model's native rate (16kHz)

        # Apply ZipEnhancer if enabled (at 16kHz, BEFORE upsampling to 48kHz)
        if use_zipenhancer:
            nonlocal zipenhancer
            if zipenhancer is None:
                zipenhancer_dir = ensure_zipenhancer_downloaded()
                zipenhancer = ZipEnhancerProcessor(zipenhancer_dir)

            # Save float32 to temp file for ZipEnhancer
            temp_path = OUTPUTS_DIR / f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            sf.write(temp_path, waveform, current_sr)

            enhanced_path = str(temp_path.with_suffix('.enhanced.wav'))
            try:
                enhanced_path = zipenhancer.enhance(str(temp_path), enhanced_path)

                # Read enhanced result AND its actual sample rate
                waveform, current_sr = sf.read(enhanced_path)

                # Cleanup
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                if os.path.exists(enhanced_path):
                    os.unlink(enhanced_path)

                logging.info(f"ZipEnhancer output: {current_sr}Hz")

            except Exception as e:
                logging.error(f"ZipEnhancer failed, keeping original: {e}")
                if temp_path and os.path.exists(temp_path):
                    os.unlink(temp_path)
                # waveform stays as original float32

        # Apply normalization to float32 data (after ZipEnhancer if used, before int16 conversion)
        if normalize:
            waveform = normalize_audio(waveform, float(normalize_level))

        # Convert to int16 ONLY at the very end, right before ffmpeg
        if waveform.dtype != np.int16:
            waveform = (waveform * 32767).astype(np.int16)

        # Pass int16, actual sample rate, target sample rate and bitrate to ffmpeg
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

        status_msg = f"Done! Saved in: {Path(saved_path).name} | Seed: {actual_seed}"
        return (current_sr, waveform), status_msg, saved_path, spectrogram_path

    theme = gr.themes.Soft(font=["Inter", "Arial", "sans-serif"])

    css = """
    .gradio-container {max-width: 100% !important;}
    .square-btn {width: 40px !important; min-width: 40px !important; padding: 0 !important; height: 40px !important;}
    .voice-controls-row {align-items: center !important;}
    .voice-controls-row button {height: 40px !important; margin-top: 24px !important;}
    .generate-btn-row {margin-bottom: 10px !important;}
    .seed-row {align-items: center !important; gap: 8px !important;}
    .seed-row button {height: 40px !important; margin-top: 24px !important; min-width: 40px !important;}
    .seed-row .form {margin-bottom: 0 !important;}
    """

    # JavaScript to reload the current tab after server restart
    js_reload_on_ready = """
    () => {
        // Show a restart message
        const statusText = document.querySelector('textarea[data-testid="textbox"]');
        if (statusText) {
            statusText.value = "⏳ Waiting for server restart...";
            statusText.dispatchEvent(new Event('input'));
        }

        const checkServer = async () => {
            try {
                const resp = await fetch(window.location.origin, {cache: "no-store"});
                if (resp.ok) {
                    // The server is available - reload the current tab
                    window.location.reload();
                } else {
                    setTimeout(checkServer, 800);
                }
            } catch (e) {
                // The server hasn't started yet - we're waiting.
                setTimeout(checkServer, 800);
            }
        };
        // We start checking in 1.5 seconds
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

    with gr.Blocks(title="OmniVoice Portable") as demo:
        gr.Markdown("<div align='center'><h1>OmniVoice Portable</h1></div>")

        if torch.cuda.is_available():
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            gr.Markdown(f'<div style="padding:10px;border-radius:5px;">🟢 GPU: {torch.cuda.get_device_name(0)} VRAM: {vram_total:.1f}GB</div>')

        with gr.Tabs():
            # VOICE CLONE
            with gr.TabItem("Voice Clone"):
                with gr.Row():
                    with gr.Column(scale=1):
                        vc_text = gr.Textbox(label="Text", lines=3, placeholder="Enter text for voiceover...")

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

                        vc_audio = gr.Audio(label="Output", type="numpy")
                        # Spectrogram output
                        vc_spectrogram = gr.Image(label="Spectrogram", type="filepath")
                        vc_status = gr.Textbox(label="Status", lines=2)
                        vc_saved_file = gr.File(label="Saved File", visible=True)

                # Button click handlers
                vc_clear_btn.click(
                    lambda: "",
                    inputs=None,
                    outputs=vc_text,
                )

                vc_paste_btn.click(
                    None,
                    inputs=None,
                    outputs=vc_text,
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
                    None,
                    inputs=[vc_text],
                    outputs=None,
                    js="(text) => navigator.clipboard.writeText(text)"
                )

                vc_refresh_btn.click(
                    lambda: gr.update(choices=get_voice_choices(), value="-NONE-"),
                    inputs=None,
                    outputs=vc_voice_selector,
                )

                vc_download_btn.click(
                    lambda: download_reference_voices(),
                    inputs=None,
                    outputs=vc_voice_selector,
                )

                vc_voice_selector.change(
                    lambda: (None, ""),
                    inputs=None,
                    outputs=[vc_ref_audio, vc_ref_text],
                ).then(
                    set_voice_file,
                    inputs=[vc_voice_selector],
                    outputs=[vc_ref_audio],
                )

                # Random seed button handler
                def _random_seed():
                    new_seed = random.randint(0, 2**32 - 1)
                    return gr.update(value=new_seed)

                vc_random_btn.click(
                    _random_seed,
                    inputs=None,
                    outputs=vc_seed,
                )

                def _clone_fn(text, lang, ref_aud, ref_text, ns, gs, dn, sp, du, pp, po, fmt, sr, br, seed, random_seed,
                             normalize, norm_level, use_zipenhancer):
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
                    }
                    save_settings(current_settings)

                    return _gen_core(text, lang, ref_aud, None, ns, gs, dn, sp, du, pp, po, fmt,
                                   mode="clone", ref_text=ref_text or None,
                                   normalize=normalize, normalize_level=norm_level,
                                   use_zipenhancer=use_zipenhancer,
                                   target_sample_rate=sr, bitrate=br,
                                   seed=seed, random_seed=random_seed)

                vc_btn.click(_clone_fn,
                    inputs=[vc_text, vc_lang, vc_ref_audio, vc_ref_text, vc_ns, vc_gs, vc_dn, vc_sp, vc_du, vc_pp, vc_po, vc_fmt, vc_sr, vc_br, vc_seed, vc_random_seed,
                           vc_normalize, vc_norm_level, vc_use_zipenhancer],
                    outputs=[vc_audio, vc_status, vc_saved_file, vc_spectrogram])

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
                        vd_text = gr.Textbox(label="Text", lines=3, placeholder="Enter text for voiceover...")

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

                        vd_audio = gr.Audio(label="Output", type="numpy")
                        # Spectrogram output
                        vd_spectrogram = gr.Image(label="Spectrogram", type="filepath")
                        vd_status = gr.Textbox(label="Status", lines=2)
                        vd_saved_file = gr.File(label="Saved File", visible=True)

                # Button click handlers
                vd_clear_btn.click(
                    lambda: "",
                    inputs=None,
                    outputs=vd_text,
                )

                vd_paste_btn.click(
                    None,
                    inputs=None,
                    outputs=vd_text,
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
                    None,
                    inputs=[vd_text],
                    outputs=None,
                    js="(text) => navigator.clipboard.writeText(text)"
                )

                vd_refresh_btn.click(
                    lambda: gr.update(choices=get_voice_choices(), value="-NONE-"),
                    inputs=None,
                    outputs=vd_voice_selector,
                )

                vd_download_btn.click(
                    lambda: download_reference_voices(),
                    inputs=None,
                    outputs=vd_voice_selector,
                )

                vd_voice_selector.change(
                    lambda: None,
                    inputs=None,
                    outputs=[vd_ref_audio],
                ).then(
                    set_voice_file,
                    inputs=[vd_voice_selector],
                    outputs=[vd_ref_audio],
                )

                # Random seed button handler
                vd_random_btn.click(
                    _random_seed,
                    inputs=None,
                    outputs=vd_seed,
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
                    # Separate groups from audio processing args
                    num_groups = len(vd_groups)
                    groups = args[:num_groups]
                    normalize = args[num_groups]
                    norm_level = args[num_groups + 1]
                    use_zipenhancer = args[num_groups + 2]

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
                    }
                    save_settings(current_settings)

                    if ref_aud:
                        return _gen_core(text, lang, ref_aud, None, ns, gs, dn, sp, du, pp, po, fmt,
                                       mode="clone", ref_text=None,
                                       normalize=normalize, normalize_level=norm_level,
                                       use_zipenhancer=use_zipenhancer,
                                       target_sample_rate=sr, bitrate=br,
                                       seed=seed, random_seed=random_seed)
                    else:
                        return _gen_core(text, lang, None, _build_instruct(groups), ns, gs, dn, sp, du, pp, po, fmt, 
                                       mode="design",
                                       normalize=normalize, normalize_level=norm_level,
                                       use_zipenhancer=use_zipenhancer,
                                       target_sample_rate=sr, bitrate=br,
                                       seed=seed, random_seed=random_seed)

                vd_btn.click(_design_fn,
                    inputs=[vd_text, vd_lang, vd_ref_audio, vd_ns, vd_gs, vd_dn, vd_sp, vd_du, vd_pp, vd_po, vd_fmt, vd_sr, vd_br, vd_seed, vd_random_seed] + vd_groups + [vd_normalize, vd_norm_level, vd_use_zipenhancer],
                    outputs=[vd_audio, vd_status, vd_saved_file, vd_spectrogram])

                vd_restart_btn.click(
                    restart_engine,
                    inputs=None,
                    outputs=vd_status,
                    js=js_reload_on_ready
                )

        gr.Markdown(f"""
                💾 Saving in: `{OUTPUTS_DIR}`
                🎤 References in: `{REFERENCE_AUDIO_DIR}`
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
    logging.info(f"Model: {args.model}")
    logging.info(f"Target sample rate: {TARGET_SAMPLE_RATE}Hz")
    logging.info(f"Is restart: {is_restart}")
    logging.info(f"Auto-open browser: {not args.no_browser}")
    logging.info("=" * 60)

    model = FullOffloadOmniVoice(checkpoint=args.model)

    demo = build_demo(model, args.model)

    # Open the browser only if:
    # 1. --no-browser not specified
    # 2. This is not a restart (is_restart=False)
    should_open_browser = (not args.no_browser) and (not is_restart)

    logging.info(f"Opening browser: {should_open_browser}")

    demo.queue().launch(
        server_name=args.ip, 
        server_port=args.port, 
        share=args.share, 
        root_path=args.root_path,
        inbrowser=should_open_browser
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
