import subprocess
import os
import sys
import re
import argparse
import threading
from pathlib import Path

class DoradoRunner:
    """
    Process Manager for Executing dorado basecalling, avoiding Deadlock.
    Handles targeted alignment and sorting for NanoSquiggle-AMR.
    """

    def __init__(self):
        # Find and configure the project root directory
        self.script_dir = Path(__file__).resolve().parent
        self.project_root = self.script_dir.parent.parent

        self.raw_dir = self.project_root / "data" / "raw"
        self.processed_dir = self.project_root / "data" / "processed" / "alignments"

        # Path to Dorado binary and reference
        self.dorado_bin = self.project_root / "bin" / "dorado"
        self.reference_fasta = self.project_root / "data" / "ref" / "resistance_genes.fasta"

        self.processed_dir.mkdir(parents=True, exist_ok=True)

        print(f"[*] Geometry Engine initialized at: {self.project_root}")

        # Assert execution permissions for the binary
        try:
            self.dorado_bin.chmod(0o755)
            print(f"[+] Execution vector asserted for {self.dorado_bin.name}")
        except Exception as e:
            print(f"[-] Warning: Could not set execute permissions on {self.dorado_bin}: {e}")

        self._emit_moves_supported = self._check_emit_moves_support()
        if self._emit_moves_supported:
            print("[+] Dorado version supports --emit-moves")
        else:
            print("[*] Dorado version does not support --emit-moves (moves emitted by default)")

    def _get_env(self) -> dict:
        """
        Constructs the environment with the local Dorado library path.
        """
        env = os.environ.copy()
        local_lib = str(self.project_root / "bin" / "lib")
        current_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{local_lib}:{current_ld}" if current_ld else local_lib
        return env

    def _check_emit_moves_support(self) -> bool:
        try:
            result = subprocess.run(
                [str(self.dorado_bin), "basecaller", "--help"],
                capture_output=True, text=True,
                env=self._get_env()
            )
            return "--emit-moves" in result.stdout
        except Exception:
            return False

    def _monitor_telemetry(self, pipe, strain_id: str):
        """
        Reads progress from stderr to provide a real-time speed metric.
        """
        throughput_pattern = re.compile(r'([0-9.]+\s+[kM]?(?:samples|bp)/s)')

        for line in iter(pipe.readline, b''):
            decoded_line = line.decode('utf-8', errors='ignore').strip()

            match = throughput_pattern.search(decoded_line)
            if match:
                throughput = match.group(1)
                sys.stderr.write(f"\r[*] {strain_id} Speed: {throughput}       ")
                sys.stderr.flush()
            elif any(x in decoded_line.lower() for x in ["error", "warning", "fatal", "info"]):
                if "samples/s" not in decoded_line and "bp/s" not in decoded_line:
                    sys.stderr.write(f"\n[*] {strain_id} Status: {decoded_line}\n")

        sys.stderr.write("\n")

    def execute_pipeline(self, strain_id: str, pod5_dir: Path) -> bool:
        """
        Executes Dorado | Samtools sort -> BAM
        Implements the Physical Log Manifold and Atomic Commit.
        """
        final_bam = self.processed_dir / f"{strain_id}.bam"

        # 1. Immediate Existence Check (Pre-flight)
        if final_bam.exists():
            print(f"[SKIP] {strain_id}: Alignment artifact present.")
            return True

        tmp_bam = self.processed_dir / f"{strain_id}.tmp.bam"
        samtools_log = self.processed_dir / f"{strain_id}_samtools.log"

        # 2. Command Definitions
        dorado_cmd = [
            str(self.dorado_bin), "basecaller",
            "hac",
            str(pod5_dir),
            "--reference", str(self.reference_fasta),
            "--emit-sam"
        ]

        if self._emit_moves_supported:
            dorado_cmd.append("--emit-moves")

        # Use samtools sort with 2G buffer for random access readiness
        samtools_cmd = ["samtools", "sort", "-m", "2G", "-o", str(tmp_bam), "-"]

        print(f"[*] {strain_id}: Booting Compute Manifold...")

        try:
            # 3. The Physical Log Manifold context
            with samtools_log.open('w') as s_log:
                p_dorado = subprocess.Popen(
                    dorado_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=self._get_env()
                )

                p_samtools = subprocess.Popen(
                    samtools_cmd,
                    stdin=p_dorado.stdout,
                    stderr=s_log  # Kernel-level offloading of samtools stderr
                )

                # SIGPIPE safety: parent no longer needs stdout handle
                p_dorado.stdout.close()

                # Telemetry monitoring in a separate thread
                monitor_thread = threading.Thread(
                    target=self._monitor_telemetry,
                    args=(p_dorado.stderr, strain_id),
                    daemon=True
                )
                monitor_thread.start()

                # Wait for processes in producer-consumer order
                samtools_exit = p_samtools.wait()
                dorado_exit = p_dorado.wait()
                monitor_thread.join()

            # 4. Atomic Commit Logic
            if dorado_exit == 0 and samtools_exit == 0:
                tmp_bam.rename(final_bam)
                print(f"[+] {strain_id}: Atomic Commit successful. Generating index...")

                # Cleanup the manifold on success
                if samtools_log.exists():
                    samtools_log.unlink()

                try:
                    # Index for random access in Phase 2.2
                    subprocess.run(["samtools", "index", str(final_bam)], check=True)
                    return True
                except subprocess.CalledProcessError as e:
                    print(f"[-] {strain_id}: Indexing Failed: {e}.")
                    return False
            else:
                print(f"[-] {strain_id}: Pipeline Failed (Dorado: {dorado_exit}, Samtools: {samtools_exit})")
                print(f"    -> Check {samtools_log.name} for kernel errors.")
                if tmp_bam.exists():
                    tmp_bam.unlink()
                return False

        except Exception as e:
            print(f"[-] FATAL IPC ERROR: {e}")
            if tmp_bam.exists():
                tmp_bam.unlink()
            return False

    def run(self, strain_ids: list[str] | None = None):
        """
        Orchestrates processing across all valid strain directories.
        """
        if not self.raw_dir.exists():
            print(f"[-] No raw manifold found at {self.raw_dir}")
            return

        for strain_folder in self.raw_dir.iterdir():
            if not strain_folder.is_dir():
                continue
            if strain_ids is not None and strain_folder.name not in strain_ids:
                continue

            # Only process verified acquisitions from Phase 1
            sentinel = strain_folder / ".download_complete"
            if not sentinel.exists():
                print(f"[!] Skipping {strain_folder.name}: Sentinel missing.")
                continue

            self.execute_pipeline(strain_folder.name, strain_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: Dorado basecalling + alignment")
    parser.add_argument("--reference", type=str, default=None, help="Reference FASTA path")
    parser.add_argument("--raw-dir", type=str, default=None, help="Directory with raw POD5 strain folders")
    parser.add_argument("--processed-dir", type=str, default=None, help="Output directory for BAM files")
    parser.add_argument("--strain", type=str, default=None, help="Process a single strain ID only")
    args = parser.parse_args()

    runner = DoradoRunner()
    if args.reference:
        runner.reference_fasta = Path(args.reference)
    if args.raw_dir:
        runner.raw_dir = Path(args.raw_dir)
    if args.processed_dir:
        runner.processed_dir = Path(args.processed_dir)
    strain_ids = [args.strain] if args.strain else None
    runner.run(strain_ids=strain_ids)
