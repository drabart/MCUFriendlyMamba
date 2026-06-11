import ctypes

import pandas as pd
import torch
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio import transforms
import torch.nn.functional as F
from torch.utils.data import TensorDataset, random_split
from torchvision import datasets, transforms
import os
import hashlib
import numpy as np
import tensorflow as tf
from ai_edge_litert.interpreter import Interpreter

from tflite_micro.python.tflite_micro import runtime


def load_mnist_data(data_dir):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_ds, val_ds = random_split(
        datasets.MNIST(data_dir, train=True, download=True, transform=transform),
        [0.8, 0.2],
    )
    test_ds = datasets.MNIST(data_dir, train=False, transform=transform)
    return train_ds, val_ds, test_ds


def load_har_data(data_dir):
    def load_txt(file_path):
        return pd.read_csv(file_path, sep=r"\s+", header=None).values

    har_data_dir = os.path.join(data_dir)
    X_train = load_txt(os.path.join(har_data_dir, "train", "X_train.txt"))
    y_train = load_txt(os.path.join(har_data_dir, "train", "y_train.txt")).squeeze() - 1
    X_test = load_txt(os.path.join(har_data_dir, "test", "X_test.txt"))
    y_test = load_txt(os.path.join(har_data_dir, "test", "y_test.txt")).squeeze() - 1

    def prepare(X):
        X = F.pad(torch.tensor(X, dtype=torch.float32), (0, 570 - 561))
        return X.view(-1, 10, 57)

    X_train = prepare(X_train)
    X_test = prepare(X_test)

    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_ds, val_ds = random_split(TensorDataset(X_train, y_train), [0.8, 0.2], generator=torch.Generator().manual_seed(42))
    test_ds = TensorDataset(X_test, y_test)
    return train_ds, val_ds, test_ds

# The 35 classes in the full Speech Commands v2 dataset
CLASSES = [
    "backward", "bed", "bird", "cat", "dog", "down", "eight", "five", "follow",
    "forward", "four", "go", "happy", "house", "learn", "left", "marvin", "nine",
    "no", "off", "on", "one", "right", "seven", "sheila", "six", "stop", "three",
    "tree", "two", "up", "visual", "wow", "yes", "zero",
]
LABEL_TO_IDX = {label: idx for idx, label in enumerate(CLASSES)}

TARGET_SAMPLE_RATE = 16_000
SAMPLE_LENGTH = 16_000  # 1 second at 16 kHz


def _preprocessing_hash(preprocessor_path):
    """Stable hash based on the preprocessor model file's modification time."""
    mtime = os.path.getmtime(preprocessor_path)
    key = f"tflite_preprocessor_{mtime}"
    return hashlib.md5(key.encode()).hexdigest()[:10]


def load_speechcommands_data(data_dir, preprocessor_tflite_path, cache_dir=None):
    """
    Load Speech Commands v2 using Google's micro_speech LiteRT preprocessor model.
    Each sample is returned as X ∈ ℝ^(49 × 40) in INT8 format, perfectly matching
    the microcontroller's on-device tensor inputs.

    Args:
        data_dir:                 Root directory where the dataset is stored.
        preprocessor_tflite_path: Path to the generated 'audio_preprocessor_int8.tflite'.
        cache_dir:                Root directory for the feature cache.
    Returns:
        train_ds, val_ds, test_ds — wrapped datasets ready for DataLoader.
    """
    if not os.path.exists(preprocessor_tflite_path):
        raise FileNotFoundError(
            f"Preprocessor LiteRT file not found at: {preprocessor_tflite_path}. "
            "Please run the micro_speech audio_preprocessor script first to generate it."
        )

    if cache_dir is None:
        cache_dir = os.path.join(data_dir, ".micro_speech_cache")

    param_tag = _preprocessing_hash(preprocessor_tflite_path)
    versioned_cache = os.path.join(cache_dir, param_tag)

    # 1. Initialize the modern CompiledModel (Defaults to CPU for data preprocessing)
    interpreter = runtime.Interpreter.from_file(
        preprocessor_tflite_path,
    )

    def preprocess(waveform, sample_rate):
        # Handle Resampling
        if sample_rate != TARGET_SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(
                sample_rate, TARGET_SAMPLE_RATE
            )(waveform)
            
        # Pad or truncate to exactly 1 second (16000 samples)
        length = waveform.shape[-1]
        if length < SAMPLE_LENGTH:
            waveform = torch.nn.functional.pad(waveform, (0, SAMPLE_LENGTH - length))
        else:
            waveform = waveform[..., :SAMPLE_LENGTH]
            
        # Convert Float32 audio (-1.0 to 1.0) into INT16 samples (-32768 to 32767)
        audio_np = waveform.squeeze(0).numpy()
        audio_int16 = (audio_np * 32768.0).astype(np.int16)

        window_size = 480
        hop_length = 320
        features = []

        for i in range(0, len(audio_int16) - window_size + 1, hop_length):
            frame = audio_int16[i : i + window_size]
            frame = np.reshape(frame, (1, window_size)) # (1, 480)
            
            # 2. Use the legacy set/get methods
            interpreter.set_input(frame, 0)
            interpreter.invoke()
            
            feature_frame = interpreter.get_output(0)
            features.append(feature_frame.flatten())

        interpreter.reset()

        return torch.tensor(np.array(features), dtype=torch.float32)

    class SpeechCommandsWrapper(torch.utils.data.Dataset):
        def __init__(self, subset):
            self._ds        = SPEECHCOMMANDS(data_dir, download=True, subset=subset)
            self._cache_dir = os.path.join(versioned_cache, subset)
            self._mem: dict[int, tuple] = {}
            os.makedirs(self._cache_dir, exist_ok=True)

        def __len__(self):
            return len(self._ds)

        def _cache_path(self, idx):
            audio_path = self._ds._walker[idx]
            rel = os.path.relpath(audio_path, data_dir)
            safe_name = rel.replace(os.sep, "__")
            return os.path.join(self._cache_dir, safe_name + ".pt")

        def __getitem__(self, idx):
            if idx in self._mem:
                return self._mem[idx]

            path = self._cache_path(idx)
            if os.path.exists(path):
                item = torch.load(path, weights_only=True)
            else:
                waveform, sample_rate, label, *_ = self._ds[idx]
                features = preprocess(waveform, sample_rate)   # Shape: (49, 40)
                target   = LABEL_TO_IDX[label]
                item     = (features, target)

                tmp_path = path + ".tmp"
                torch.save((features, target), tmp_path)
                os.replace(tmp_path, path)

            self._mem[idx] = item
            return item

    train_ds = SpeechCommandsWrapper("training")
    val_ds   = SpeechCommandsWrapper("validation")
    test_ds  = SpeechCommandsWrapper("testing")
    return train_ds, val_ds, test_ds


def get_data_input_size(dataset):
    if dataset == "mnist":
        input_dim = 28
    elif dataset == "har":
        input_dim = 57
    elif dataset == "kws":
        input_dim = 40
    else:
        raise ValueError("Unknown dataset type", dataset)
    return input_dim


def get_data_output_size(dataset):
    if dataset == "mnist":
        output_dim = 10
    elif dataset == "har":
        output_dim = 6
    elif dataset == "kws":
        output_dim = 35
    else:
        raise ValueError("Unknown dataset type", dataset)
    return output_dim