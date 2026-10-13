"""Data pipeline for GTZAN (4 genres) exactly as described in Sawhney et al. (2018).

Steps (paper Sec. 3 + authors' public code):
  1. Load each .wav with librosa at its native 22,050 Hz.
  2. Keep the first 600,000 samples and average-pool with a window of 40
     -> 15,000 samples (an effective ~551 Hz signal).
  3. Cut the first 20 consecutive windows of 500 samples (each window covers
     20,000 raw samples, i.e. ~0.9 s of audio) -> 20 examples per song.
  4. 4 genres x 100 songs x 20 segments = 8,000 examples in R^500.
  5. Random 6000 / 1000 / 1000 split into train / dev / test.

The paper (and the authors' code) shuffles *segments*, so segments from the same
song can land in both train and test. We reproduce that split ("segment") and
also provide a stricter song-level split ("song") as a supplementary check.
"""
import os
import numpy as np

GENRES = ["classical", "jazz", "metal", "pop"]
N_SEGMENTS = 20
SEG_LEN = 500
POOL = 40
KEEP = 600_000

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "genres")
CACHE = os.path.join(ROOT, "data", "processed_gtzan4.npz")


def build_dataset(raw_dir=RAW_DIR, cache=CACHE):
    """Return X (8000, 500), y (8000,), song_id (8000,). Cached after first run."""
    if os.path.exists(cache):
        d = np.load(cache)
        return d["X"], d["y"], d["song"]
    import librosa

    X, y, song = [], [], []
    skipped = []
    sid = 0
    for label, g in enumerate(GENRES):
        files = sorted(f for f in os.listdir(os.path.join(raw_dir, g)) if f.endswith(".wav"))
        for f in files:
            # Data cleaning: some GTZAN copies contain an unreadable file (the Kaggle
            # version's jazz.00054.wav is corrupted). Skip anything that fails to decode
            # or is nearly empty instead of crashing.
            try:
                audio, sr = librosa.load(os.path.join(raw_dir, g, f), sr=22050)
            except Exception as e:
                skipped.append(f"{g}/{f}")
                print(f"  skipping unreadable file {g}/{f} ({type(e).__name__})")
                continue
            if len(audio) < KEEP * 0.9:
                skipped.append(f"{g}/{f}")
                print(f"  skipping too-short file {g}/{f} ({len(audio)} samples)")
                continue
            audio = audio[:KEEP]
            if len(audio) < KEEP:  # safety for slightly short clips
                audio = np.pad(audio, (0, KEEP - len(audio)))
            pooled = audio.reshape(-1, POOL).mean(axis=1)  # (15000,)
            for j in range(N_SEGMENTS):
                X.append(pooled[j * SEG_LEN:(j + 1) * SEG_LEN])
                y.append(label)
                song.append(sid)
            sid += 1
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    song = np.asarray(song, dtype=np.int64)
    print(f"Loaded {len(np.unique(song))} songs -> {len(X)} segments; skipped {len(skipped)} file(s)"
          + (f": {', '.join(skipped)}" if skipped else ""))
    np.savez_compressed(cache, X=X, y=y, song=song)
    return X, y, song


def make_split(y, song, kind="segment", seed=0):
    """Indices for train/dev/test with 6000/1000/1000 examples."""
    rng = np.random.RandomState(seed)
    if kind == "segment":  # as in the paper
        idx = rng.permutation(len(y))
        return idx[:6000], idx[6000:7000], idx[7000:]
    if kind == "song":  # stricter: whole songs in one split, stratified by genre
        tr, dv, te = [], [], []
        for g in np.unique(y):
            songs = rng.permutation(np.unique(song[y == g]))  # 100 songs per genre
            for s_list, out in ((songs[:75], tr), (songs[75:87], dv), (songs[87:], te)):
                out.extend(np.where(np.isin(song, s_list))[0])
        # 75/12/13 songs per genre -> 6000/960/1040 segments
        return rng.permutation(tr), rng.permutation(dv), rng.permutation(te)
    raise ValueError(kind)


if __name__ == "__main__":
    X, y, song = build_dataset()
    print("X", X.shape, "y counts", np.bincount(y), "songs", len(np.unique(song)))
