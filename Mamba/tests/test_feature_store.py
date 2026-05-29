import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

H5_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "hd5f" / "amr_features.h5"
PLOT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "hd5f"


def test_file_exists():
    assert H5_PATH.exists(), f"Feature store not found at {H5_PATH}"
    print(f"[PASS] File exists: {H5_PATH}")


def test_datasets():
    f = h5py.File(H5_PATH, "r")
    expected = {"X", "Y", "read_id", "strain_id", "gene_target"}
    assert expected.issubset(f.keys()), f"Missing datasets: {expected - set(f.keys())}"
    print(f"[PASS] All datasets present: {sorted(f.keys())}")

    assert f["X"].shape[0] == f["Y"].shape[0], "X and Y length mismatch"
    assert f["X"].dtype == np.float32, f"X dtype is {f['X'].dtype}, expected float32"
    assert f["Y"].dtype == np.int8, f"Y dtype is {f['Y'].dtype}, expected int8"
    print(f"[PASS] X shape: {f['X'].shape}, Y shape: {f['Y'].shape}")
    f.close()


def test_class_balance():
    f = h5py.File(H5_PATH, "r")
    y = f["Y"][:]
    pos = np.sum(y == 1)
    neg = np.sum(y == 0)
    total = len(y)
    print(f"[INFO] Positives: {pos} ({pos/total*100:.1f}%), Negatives: {neg} ({neg/total*100:.1f}%)")
    assert pos > 0 and neg > 0, "Both classes must be present"
    assert pos / total > 0.05, f"Too few positives ({pos/total*100:.1f}%)"
    print("[PASS] Both classes present with sufficient representation")
    f.close()


def test_strain_coverage():
    f = h5py.File(H5_PATH, "r")
    strains = f["strain_id"][:]
    unique = set(s.decode() for s in np.unique(strains))
    print(f"[INFO] Strains found: {sorted(unique)}")
    assert len(unique) > 0, "No strains found"
    for s in unique:
        count = np.sum(strains == s.encode())
        print(f"       {s}: {count} reads")
        assert count > 0, f"Strain {s} has 0 reads"
    print("[PASS] All strains have reads")
    f.close()


def test_signal_quality():
    f = h5py.File(H5_PATH, "r")
    x = f["X"][:]
    y = f["Y"][:]

    print(f"[INFO] Signal shape: {x.shape}")
    print(f"[INFO] Global stats — min: {x.min():.4f}, max: {x.max():.4f}, mean: {x.mean():.4f}, std: {x.std():.4f}")

    assert x.min() > -100, f"Signal min too low: {x.min()}"
    assert x.max() < 100, f"Signal max too high: {x.max()}"
    assert abs(x.mean()) < 1, f"Signal mean too far from 0: {x.mean()}"
    assert 0.5 < x.std() < 2, f"Signal std outside expected range: {x.std()}"

    extreme = np.sum(np.any(np.abs(x) > 50, axis=1))
    assert extreme == 0, f"{extreme} samples have extreme values (>50)"

    all_zero = np.sum(np.all(np.abs(x) < 1e-6, axis=1))
    print(f"[INFO] All-zero windows: {all_zero}")
    assert all_zero < len(x) * 0.01, f"Too many all-zero windows: {all_zero}/{len(x)}"

    print("[PASS] Signal stats within expected ranges")
    f.close()


def test_signal_quality_per_class():
    f = h5py.File(H5_PATH, "r")
    x = f["X"][:]
    y = f["Y"][:]
    gene = f["gene_target"][:]

    for cls_name, mask in [
        ("AMR", y == 1),
        ("MLST", (y == 0) & (gene != b"background")),
        ("background", gene == b"background"),
    ]:
        if np.sum(mask) == 0:
            print(f"[WARN] {cls_name}: 0 samples, skipping")
            continue
        sub = x[mask]
        print(f"[INFO] {cls_name} ({len(sub)} samples) — min: {sub.min():.4f}, max: {sub.max():.4f}, "
              f"mean: {sub.mean():.4f}, std: {sub.std():.4f}")

        extreme = np.sum(np.any(np.abs(sub) > 50, axis=1))
        assert extreme == 0, f"{cls_name}: {extreme} extreme samples"

    print("[PASS] Per-class signal quality OK")
    f.close()


def test_gene_distribution():
    f = h5py.File(H5_PATH, "r")
    gene = f["gene_target"][:]
    unique, counts = np.unique(gene, return_counts=True)
    print("[INFO] Gene distribution:")
    for g, c in zip(unique, counts):
        print(f"       {g.decode()}: {c}")
    assert len(unique) >= 3, "Fewer than 3 gene categories"
    print("[PASS] Gene distribution covers all expected categories")
    f.close()


def plot_sample_signals():
    f = h5py.File(H5_PATH, "r")
    x = f["X"][:]
    y = f["Y"][:]
    gene = f["gene_target"][:]

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(14, 8))

    categories = [
        ("AMR Positive", y == 1, "#e74c3c"),
        ("MLST Hard Negative", (y == 0) & (gene != b"background"), "#3498db"),
        ("Background", gene == b"background", "#95a5a6"),
    ]

    for ax, (title, mask, color) in zip(axes, categories):
        indices = np.where(mask)[0]
        if len(indices) == 0:
            ax.text(0.5, 0.5, "No samples", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            continue

        samples = np.random.choice(indices, min(5, len(indices)), replace=False)
        offset = 0
        for i, idx in enumerate(samples):
            signal = x[idx]
            ax.plot(signal + offset, color=color, alpha=0.8, linewidth=0.5)
            offset += 4

        ax.set_ylabel("Signal (MAD units)")
        ax.set_title(f"{title} ({len(indices)} reads)")
        ax.set_yticks([])

    axes[-1].set_xlabel("Sample index")
    plt.tight_layout()

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = PLOT_DIR / "signal_quality_check.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"[PASS] Plot saved to {plot_path}")
    f.close()


if __name__ == "__main__":
    print("=" * 60)
    print("Feature Store Quality Tests")
    print("=" * 60)

    test_file_exists()
    print()
    test_datasets()
    print()
    test_class_balance()
    print()
    test_strain_coverage()
    print()
    test_gene_distribution()
    print()
    test_signal_quality()
    print()
    test_signal_quality_per_class()
    print()
    plot_sample_signals()
    print()
    print("=" * 60)
    print("All tests passed.")
    print("=" * 60)
