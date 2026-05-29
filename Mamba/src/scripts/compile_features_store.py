import numpy as np
import h5py
import pod5
import pysam
import argparse
from pathlib import Path
from collections import defaultdict


class FeatureStoreCompiler:
    def __init__(self, target_window=30000, coverage_threshold=0.95, max_background_per_strain=500):
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.raw_dir = self.project_root / "data" / "raw"
        self.alignments_dir = self.project_root / "data" / "processed" / "alignments"
        self.hdf5_dir = self.project_root / "data" / "processed" / "hd5f"
        self.index_file = self.project_root / "data" / "data_index" / "data_ids.txt"
        self.h5_path = self.hdf5_dir / "amr_features.h5"
        self.error_log_path = self.hdf5_dir / "compilation_errors.log"
        self.target_window = target_window
        self.coverage_threshold = coverage_threshold
        self.max_background_per_strain = max_background_per_strain

        self.pos_genes = [
            "ENA|HEE1644226|HEE1644226.1",
            "ENA|MH733892|MH733892.1",
            "ENA|MZ092836|MZ092836.1"
        ]
        self.neg_genes = [
            "rpoB_1_Klebsiella_pneumoniae_Pasteur",
            "gapA_1_Klebsiella_pneumoniae_Pasteur",
            "mdh_1_Klebsiella_pneumoniae_Pasteur"
        ]

    def _mad_normalize(self, signal):
        median = np.median(signal)
        mad = np.median(np.abs(signal - median))
        if mad == 0:
            mad = 1e-6
        return (signal - median) / mad

    def _extract_spatial_window(self, read, pod5_record):
        tags = dict(read.tags)
        if "mv" not in tags or "ts" not in tags:
            return None

        stride = tags["mv"][0]
        moves = np.array(tags["mv"][1:], dtype=np.int32)
        trim_start = tags["ts"]

        cum_moves = np.cumsum(moves)
        moves_to_start = np.searchsorted(cum_moves, read.query_alignment_start, side="left") + 1
        moves_to_end = np.searchsorted(cum_moves, read.query_alignment_end, side="left") + 1

        raw_start = trim_start + (moves_to_start * stride)
        raw_end = trim_start + (moves_to_end * stride)

        signal_midpoint = raw_start + ((raw_end - raw_start) // 2)
        tensor_start = signal_midpoint - (self.target_window // 2)
        tensor_end = signal_midpoint + (self.target_window // 2)

        raw_signal = self._mad_normalize(pod5_record.signal)

        pad_left = max(0, -tensor_start)
        pad_right = max(0, tensor_end - len(raw_signal))

        valid_start = max(0, tensor_start)
        valid_end = min(len(raw_signal), tensor_end)

        sliced_signal = raw_signal[valid_start:valid_end]

        if pad_left > 0 or pad_right > 0:
            sliced_signal = np.pad(
                sliced_signal, (pad_left, pad_right),
                mode="constant", constant_values=0
            )

        return sliced_signal

    def _extract_background_window(self, pod5_record):
        raw_signal = self._mad_normalize(pod5_record.signal)
        if len(raw_signal) < self.target_window:
            pad_total = self.target_window - len(raw_signal)
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            signal = np.pad(raw_signal, (pad_left, pad_right), mode="constant", constant_values=0)
        else:
            center = len(raw_signal) // 2
            start = center - self.target_window // 2
            signal = raw_signal[start:start + self.target_window]
        return signal

    def _collect_reads_for_strain(self, strain_id):
        bam_path = self.alignments_dir / f"{strain_id}.bam"
        if not bam_path.exists():
            print(f"[!] {strain_id}: BAM not found at {bam_path}")
            return []

        samfile = pysam.AlignmentFile(bam_path, "rb")
        reads = []

        for gene, label in (
            [(g, 1) for g in self.pos_genes]
            + [(g, 0) for g in self.neg_genes]
        ):
            ref_len = None
            try:
                ref_len = samfile.get_reference_length(gene)
            except ValueError:
                print(f"[!] {strain_id}: Gene '{gene}' not in BAM header, skipping")
                continue
            if ref_len is None or ref_len == 0:
                print(f"[!] {strain_id}: Gene '{gene}' has zero-length reference, skipping")
                continue

            gene_count = 0
            for aln in samfile.fetch(reference=gene):
                if not aln.is_mapped:
                    continue
                rlen = aln.reference_length
                if rlen is None or rlen == 0:
                    continue
                if rlen / ref_len >= self.coverage_threshold:
                    reads.append({
                        "read_id": aln.query_name,
                        "label": label,
                        "strain": strain_id,
                        "gene": gene,
                        "sam_read": aln,
                    })
                    gene_count += 1
            print(f"[*] {strain_id}: {gene_count} reads for {'pos' if label == 1 else 'neg'} target '{gene.split('|')[-1] if '|' in gene else gene}'")

        # Collect background (unmapped) reads up to the limit
        background_count = 0
        for aln in samfile.fetch(until_eof=True):
            if aln.is_unmapped:
                reads.append({
                    "read_id": aln.query_name,
                    "label": 0,
                    "strain": strain_id,
                    "gene": "background",
                    "sam_read": aln,
                    "type": "background",
                })
                background_count += 1
                if background_count >= self.max_background_per_strain:
                    break

        samfile.close()
        return reads

    def compile(self, strain_ids=None, append=False):
        if strain_ids is None:
            if not self.index_file.exists():
                print(f"[-] FATAL: Index file not found at {self.index_file}")
                return
            with open(self.index_file) as f:
                strain_ids = [l.strip().rstrip('\r\n') for l in f if l.strip()]

        print(f"[*] Initiating Topological Projection. Strains: {strain_ids}")

        # --- Pass 1: BAM Scan ---
        all_reads = []
        for sid in strain_ids:
            reads = self._collect_reads_for_strain(sid)
            print(f"[+] {sid}: {len(reads)} valid reads")
            all_reads.extend(reads)

        N = len(all_reads)
        if N == 0:
            print("[-] No reads to process. Exiting.")
            return

        print(f"[+] Total valid reads: {N}")

        # --- Pass 2: Signal Extraction + HDF5 Write ---
        self.hdf5_dir.mkdir(parents=True, exist_ok=True)
        dt_str = h5py.string_dtype(encoding="utf-8")

        by_strain = defaultdict(list)
        for r in all_reads:
            by_strain[r["strain"]].append(r)

        errors = []

        file_mode = "r+" if (append and self.h5_path.exists()) else "w"
        with h5py.File(self.h5_path, file_mode) as h5f:
            if file_mode == "r+":
                start = h5f["X"].shape[0]
                total = start + N
                for key in ["X", "Y", "read_id", "strain_id", "gene_target"]:
                    ds = h5f[key]
                    if ds.ndim == 2:
                        ds.resize((total, ds.shape[1]))
                    else:
                        ds.resize((total,))
                X = h5f["X"]; Y = h5f["Y"]
                M_read = h5f["read_id"]; M_strain = h5f["strain_id"]; M_gene = h5f["gene_target"]
                print(f"[+] Appending {N} reads starting at index {start}")
            else:
                start = 0
                X = h5f.create_dataset(
                    "X", shape=(N, self.target_window),
                    maxshape=(None, self.target_window), dtype=np.float32
                )
                Y = h5f.create_dataset("Y", shape=(N,), maxshape=(None,), dtype=np.int8)
                M_read = h5f.create_dataset("read_id", shape=(N,), maxshape=(None,), dtype=dt_str)
                M_strain = h5f.create_dataset("strain_id", shape=(N,), maxshape=(None,), dtype=dt_str)
                M_gene = h5f.create_dataset("gene_target", shape=(N,), maxshape=(None,), dtype=dt_str)

            idx = start
            success_by_strain = defaultdict(int)
            for strain_id, strain_reads in by_strain.items():
                pod5_dir = self.raw_dir / strain_id
                pod5_files = sorted(pod5_dir.glob("*.pod5"))
                if not pod5_files:
                    print(f"[-] {strain_id}: No POD5 files found")
                    for r in strain_reads:
                        errors.append((r["read_id"], strain_id, "No POD5 files"))
                    continue

                pending = {r["read_id"]: r for r in strain_reads}

                for pod5_file in pod5_files:
                    if not pending:
                        break

                    remaining = list(pending.keys())
                    with pod5.Reader(pod5_file) as reader:
                        for record in reader.reads(remaining, missing_ok=True):
                            rid = str(record.read_id)
                            meta = pending.pop(rid, None)
                            if meta is None:
                                continue

                            try:
                                if meta.get("type") == "background":
                                    signal = self._extract_background_window(record)
                                else:
                                    signal = self._extract_spatial_window(
                                        meta["sam_read"], record
                                    )
                                if signal is None:
                                    raise ValueError("Read missing mv or ts tags")

                                X[idx] = signal
                                Y[idx] = meta["label"]
                                M_read[idx] = rid
                                M_strain[idx] = strain_id
                                M_gene[idx] = meta["gene"]
                                idx += 1
                                success_by_strain[strain_id] += 1

                                if idx % 100 == 0:
                                    print(f"    -> Projected {idx} tensors...")
                            except Exception as e:
                                errors.append((rid, strain_id, str(e)))

                for rid, meta in pending.items():
                    errors.append((rid, strain_id, "Read not found in any POD5 file"))

            actual = idx
            if file_mode == "w" and actual < start + N:
                remaining = start + N - actual
                total_expected = start + N
                X.resize((actual, self.target_window))
                Y.resize((actual,))
                M_read.resize((actual,))
                M_strain.resize((actual,))
                M_gene.resize((actual,))

        if errors:
            with open(self.error_log_path, "a" if append else "w") as f:
                if file_mode == "w":
                    f.write("read_id,strain_id,error\n")
                for rid, sid, err in errors:
                    f.write(f"{rid},{sid},{err}\n")
            print(f"[!] {len(errors)} errors logged to {self.error_log_path}")

        written = actual - start
        print(f"[+] Feature Store saved to {self.h5_path}")
        print(f"[+] Tensors written this run: {written}")
        if written < N:
            print(f"[!] Failed / skipped {N - written} of {N} reads ({((N - written) / N) * 100:.1f}%)")

        for sid in sorted(success_by_strain):
            n_written = success_by_strain[sid]
            n_requested = len(by_strain[sid])
            if n_written < n_requested:
                print(f"[*] {sid}: {n_written}/{n_requested} tensors written ({n_requested - n_written} failed)")
            else:
                print(f"[*] {sid}: {n_written} tensors written")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3: Compile HDF5 feature store from BAM + POD5")
    parser.add_argument("--window-size", type=int, default=30000, help="Signal window size in samples (default: 30000)")
    parser.add_argument("--coverage", type=float, default=0.95, help="Min alignment coverage for positive reads (default: 0.95)")
    parser.add_argument("--max-background", type=int, default=500, help="Max background reads per strain (default: 500)")
    parser.add_argument("--index-file", type=str, default=None, help="Path to strain ID index file")
    parser.add_argument("--strain", type=str, default=None, help="Process a single strain ID only")
    parser.add_argument("--append", action="store_true", help="Append to existing HDF5 instead of overwriting")
    args = parser.parse_args()

    compiler = FeatureStoreCompiler(
        target_window=args.window_size,
        coverage_threshold=args.coverage,
        max_background_per_strain=args.max_background,
    )
    if args.index_file:
        compiler.index_file = Path(args.index_file)
    strain_ids = [args.strain] if args.strain else None
    compiler.compile(strain_ids=strain_ids, append=args.append)
