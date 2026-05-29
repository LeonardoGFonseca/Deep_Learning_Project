# Implementation Plan: BAM Audit & Quality Filtration

## Objective
Quantify and categorize the aligned nanopore reads within the existing BAM manifold to ensure a high-signal-to-noise ratio for the MambaCNN training.

## Key Files & Context
- **Source Alignments:** data/processed/alignments/*.bam
- **Reference Manifest:** data/ref/resistance_genes.fasta
- **Audit Script:** src/scripts/audit_bam_quality.py (To be created)

## Implementation Steps

### 1. Classification Logic
The script will iterate through each BAM file and categorize reads based on their mapping target:
- **Positives:** Aligned to ENA|... (AMR resistance genes).
- **Hard Negatives:** Aligned to rpoB, gapA, or mdh (Housekeeping genes).
- **Background:** is_unmapped == True.

### 2. Quality Gates (The "Filters")
For a read to be considered "Training Ready," it must pass three gates:
- **Coverage Gate:** Aligned length must be >= 95% of the reference gene length.
- **Tag Gate:** Must contain both mv (Move Table) and ts (Trim Start) tags.
- **Primary Alignment:** Only primary alignments are considered (excludes secondary/supplementary).

### 3. Data Auditing & Telemetry
The script will generate a JSON or Markdown report containing:
- Total reads processed per strain.
- Count of high-quality positives vs. fragmented (discarded) positives.
- Count of hard negatives (MLST genes).
- Percentage of reads missing required Dorado tags.

## Verification & Testing
- **Execution:** Run the script against a small BAM (e.g., KP1779.bam).
- **Validation:** Ensure the total number of reads matches the samtools flagstat output.
- **Output Check:** Verify that the "Fragmented" count correctly identifies reads below the 95% threshold.