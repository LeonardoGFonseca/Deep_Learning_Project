import pysam
import json
import argparse
from pathlib import Path


class BAMAuditor:
    def __init__(self, bam_path: Path, coverage_threshold: float = 0.90):
        self.bam_path = bam_path
        self.coverage_threshold = coverage_threshold

        self.stats = {
            "total_read_processed": 0,
            "high_quality_positives": 0,
            "fragmented_positive_discarded": 0,
            "hard_negatives": 0,
            "background_unmapped": 0,
            "missing_move_tags": 0,
            "secondary_alignment_ignored": 0}
    def audit(self):
        print(f"[*] Auditing BAM: {self.bam_path.name}")
        # 'rb' indicates reading a binary file
        try:
            samfile = pysam.AlignmentFile(self.bam_path, "rb")
        except ValueError:
            print(f"[-] FATAL: Could not open BAM file {self.bam_path}. Ensure it is a valid BAM file.")
            return self.stats

        for read in samfile.fetch(until_eof=True):
            self.stats["total_read_processed"] += 1

            # GATE:1 primary alignment check. We only consider primary alignments for our analysis, as secondary and supplementary alignments can introduce noise and ambiguity in the context of targeted gene detection.
            if read.is_secondary or read.is_supplementary:
                self.stats["secondary_alignment_ignored"] += 1
                continue
            # GATE:2 Unmapped reads are categorized as background noise.
            if read.is_unmapped:
                self.stats["background_unmapped"] += 1
                continue
            # GATE:3 Move Table and Trim Start tags required for coordinate-to-signal mapping
            if not read.has_tag("mv") or not read.has_tag("ts"):
                self.stats["missing_move_tags"] += 1
                continue

            # --- THE MATHEMATICAL COVERAGE GATE ---
            # Extract the actual length of the reference sequence from the BAM header
            reference_name = read.reference_name
            l_ref = samfile.get_reference_length(reference_name)

            # Extract how many bases aligned to the reference.
            l_aligned = read.reference_length
            coverage_ratio = l_aligned / l_ref if l_ref > 0 else 0

            # House keeping genes add it in the fasta reference .
            is_housekeeping = any(hk in reference_name.lower() for hk in ['rpob', 'gapa', 'mdh'])

            # Check for high-quality positive alignments

            if coverage_ratio >= self.coverage_threshold:
                if is_housekeeping:
                    self.stats["hard_negatives"] += 1
                else:
                    self.stats["high_quality_positives"] += 1
            else:
                # Check if the read mapped to a resistance gene but did not meet the coverage threshold 
                # indicating a fragmented positive alignment that should be discarded.
                if not is_housekeeping:
                    self.stats["fragmented_positive_discarded"] += 1
        samfile.close()
        return self.stats

    def print_report(self):
        print("\n" + "="*50)
        print(f"🧬 AUDIT REPORT: {self.bam_path.name}")
        print("="*50)
        print(json.dumps(self.stats, indent=4))

        # Calculate Signal-to-Noise Purity
        usable_reads = (self.stats['high_quality_positives'] +
                        self.stats['hard_negatives'] +
                        self.stats['background_unmapped'])

        if self.stats['total_read_processed'] > 0:
            yield_pct = (usable_reads / self.stats['total_read_processed']) * 100
            print(f"\n📊 Topological Yield: {yield_pct:.2f}% of reads are Training-Ready.")
        print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit BAM quality for training readiness")
    parser.add_argument("--coverage-threshold", type=float, default=0.90, help="Min coverage ratio for high-quality alignments (default: 0.90)")
    parser.add_argument("--bam-dir", type=str, default=None, help="Directory with BAM files to audit")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    alignments_dir = Path(args.bam_dir) if args.bam_dir else project_root / "data" / "processed" / "alignments"

    if not alignments_dir.exists():
        print(f"[-] Alignment manifold missing at {alignments_dir}")
    else:
        for bam_file in sorted(alignments_dir.glob("*.bam")):
            auditor = BAMAuditor(bam_path=bam_file, coverage_threshold=args.coverage_threshold)
            auditor.audit()
            auditor.print_report()