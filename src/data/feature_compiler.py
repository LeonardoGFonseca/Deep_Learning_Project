import numpy as np
import h5py
import pod5
import pysam
from pathlib import Path
from collections import defaultdict
import os

class FeatureStoreAppender:
    def __init__(self, target_window=30000, coverage_threshold=0.95, max_background_per_strain=500, h5_path="amr_features_master.h5"):
        self.target_window = target_window
        self.coverage_threshold = coverage_threshold
        self.max_background_per_strain = max_background_per_strain
        self.h5_path = Path(h5_path)
        
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
            sliced_signal = np.pad(sliced_signal, (pad_left, pad_right), mode="constant", constant_values=0)
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

    def _collect_reads(self, bam_path, strain_id):
        if not Path(bam_path).exists():
            print(f"[!] BAM not found at {bam_path}")
            return []

        samfile = pysam.AlignmentFile(bam_path, "rb")
        reads = []

        for gene, label in ([(g, 1) for g in self.pos_genes] + [(g, 0) for g in self.neg_genes]):
            try:
                ref_len = samfile.get_reference_length(gene)
                for aln in samfile.fetch(reference=gene):
                    if aln.reference_length / ref_len >= self.coverage_threshold:
                        reads.append({
                            "read_id": aln.query_name, "label": label, 
                            "strain": strain_id, "gene": gene, "sam_read": aln
                        })
            except ValueError:
                pass

        background_count = 0
        for aln in samfile.fetch(until_eof=True):
            if aln.is_unmapped:
                reads.append({
                    "read_id": aln.query_name, "label": 0, "strain": strain_id,
                    "gene": "background", "sam_read": aln, "type": "background"
                })
                background_count += 1
                if background_count >= self.max_background_per_strain:
                    break
        samfile.close()
        return reads

    def compile_strain(self, pod5_dir, bam_path, strain_id):
        print(f"[*] Compiling tensors for {strain_id} into HDF5...")
        
        reads = self._collect_reads(bam_path, strain_id)
        print(f"[+] {strain_id}: Found {len(reads)} viable reads in BAM.")
        if len(reads) == 0:
            return

        pod5_files = sorted(Path(pod5_dir).glob("*.pod5"))
        if not pod5_files:
            print(f"[-] {strain_id}: No POD5 files found.")
            return

        pending = {r["read_id"]: r for r in reads}
        X_batch, Y_batch, M_rid, M_sid, M_gen = [], [], [], [], []

        for pod5_file in pod5_files:
            if not pending: break
            remaining = list(pending.keys())
            with pod5.Reader(pod5_file) as reader:
                for record in reader.reads(remaining, missing_ok=True):
                    rid = str(record.read_id)
                    meta = pending.pop(rid, None)
                    if meta is None: continue
                    try:
                        if meta.get("type") == "background":
                            signal = self._extract_background_window(record)
                        else:
                            signal = self._extract_spatial_window(meta["sam_read"], record)
                        
                        if signal is None: continue
                        
                        X_batch.append(signal)
                        Y_batch.append(meta["label"])
                        M_rid.append(rid)
                        M_sid.append(strain_id)
                        M_gen.append(meta["gene"])
                    except Exception as e:
                        pass
        
        n_new = len(X_batch)
        if n_new == 0:
            print("[-] No valid signals extracted.")
            return

        dt_str = h5py.string_dtype(encoding="utf-8")
        
        if not self.h5_path.exists():
            with h5py.File(self.h5_path, "w") as h5f:
                h5f.create_dataset("X", shape=(n_new, self.target_window), maxshape=(None, self.target_window), dtype=np.float32)
                h5f.create_dataset("Y", shape=(n_new,), maxshape=(None,), dtype=np.int8)
                h5f.create_dataset("read_id", shape=(n_new,), maxshape=(None,), dtype=dt_str)
                h5f.create_dataset("strain_id", shape=(n_new,), maxshape=(None,), dtype=dt_str)
                h5f.create_dataset("gene_target", shape=(n_new,), maxshape=(None,), dtype=dt_str)
                
                h5f["X"][:] = X_batch
                h5f["Y"][:] = Y_batch
                h5f["read_id"][:] = M_rid
                h5f["strain_id"][:] = M_sid
                h5f["gene_target"][:] = M_gen
        else:
            with h5py.File(self.h5_path, "a") as h5f:
                old_len = h5f["X"].shape[0]
                new_len = old_len + n_new
                
                h5f["X"].resize((new_len, self.target_window))
                h5f["Y"].resize((new_len,))
                h5f["read_id"].resize((new_len,))
                h5f["strain_id"].resize((new_len,))
                h5f["gene_target"].resize((new_len,))
                
                h5f["X"][old_len:new_len] = X_batch
                h5f["Y"][old_len:new_len] = Y_batch
                h5f["read_id"][old_len:new_len] = M_rid
                h5f["strain_id"][old_len:new_len] = M_sid
                h5f["gene_target"][old_len:new_len] = M_gen

        print(f"[+] Appended {n_new} tensors to {self.h5_path}. Current HDF5 size: {new_len if 'new_len' in locals() else n_new} tensors.")
