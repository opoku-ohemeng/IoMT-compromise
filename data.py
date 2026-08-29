import numpy as np
import wfdb


def load_real_mitbih_dataset(
    record_ids=("100", "101", "102", "103"),
    segments_per_rec=25,
    segment_len=720,
):
    """Streams multi-subject telemetry from PhysioNet and segments into windows."""
    signals, labels = [], []
    for idx, rec in enumerate(record_ids):
        record = wfdb.rdrecord(
            rec, pn_dir="mitdb", sampto=segments_per_rec * segment_len
        )
        sig = record.p_signal[:, 0]
        for s in range(segments_per_rec):
            segment = sig[s * segment_len : (s + 1) * segment_len]
            # Zero-mean unit-variance normalization
            segment = (segment - np.mean(segment)) / (np.std(segment) + 1e-8)
            signals.append(segment)
            labels.append(idx % 2)
    return np.array(signals), np.array(labels)


def calculate_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Calculates Peak Signal-to-Noise Ratio (PSNR)."""
    mse = np.mean((original - reconstructed) ** 2)
    if mse < 1e-10:
        return 100.0
    max_pixel = np.max(original) - np.min(original)
    return 20 * np.log10(max_pixel / np.sqrt(mse))
