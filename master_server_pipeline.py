import os
import tarfile
import shutil
import requests
import subprocess
from pathlib import Path
from src.data.feature_compiler import FeatureStoreAppender

PROJECT_ROOT = Path(__file__).resolve().parent

# Configurações de Pastas
RAW_DIR = PROJECT_ROOT / "data" / "raw"
ALIGNMENTS_DIR = PROJECT_ROOT / "data" / "processed" / "alignments"
HDF5_DIR = PROJECT_ROOT / "data" / "processed" / "hd5f"
INDEX_FILE = PROJECT_ROOT / "data_index" / "data_ids.txt"
REF_FASTA = PROJECT_ROOT / "data" / "ref" / "resistance_genes.fasta"
MASTER_H5 = HDF5_DIR / "amr_features_master.h5"
DORADO_BIN = PROJECT_ROOT / "bin" / "dorado" / "bin" / "dorado"

# URL base do repositório
BASE_URL = "https://data.narodni-repozitar.cz/general/datasets/dj8ys-a4r49/files/"

def setup_directories():
    for d in [RAW_DIR, ALIGNMENTS_DIR, HDF5_DIR, REF_FASTA.parent, INDEX_FILE.parent]:
        d.mkdir(parents=True, exist_ok=True)
    if not INDEX_FILE.exists():
        print(f"[!] Please create {INDEX_FILE} with strain IDs before running.")
        exit(1)

def download_strain(strain_id: str, strain_dir: Path) -> bool:
    url = f"{BASE_URL}{strain_id}_pod5.tar.gz"
    tar_path = strain_dir / f"{strain_id}.tar.gz"
    
    print(f"[*] Downloading {strain_id} from {url}...")
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(tar_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return True
    except Exception as e:
        print(f"[-] Download failed for {strain_id}: {e}")
        if tar_path.exists(): tar_path.unlink()
        return False

def extract_pod5_pass(strain_id: str, strain_dir: Path) -> bool:
    tar_path = strain_dir / f"{strain_id}.tar.gz"
    temp_extract = strain_dir / "temp_extract"
    temp_extract.mkdir(exist_ok=True)
    
    print(f"[*] Extracting {strain_id}...")
    try:
        with tarfile.open(tar_path, 'r:*') as tar:
            tar.extractall(path=temp_extract)
        
        # Procurar por pod5_pass recursivamente
        found_pod5 = False
        for root, dirs, files in os.walk(temp_extract):
            if "pod5_pass" in root:
                for file in files:
                    if file.endswith(".pod5"):
                        shutil.move(os.path.join(root, file), str(strain_dir / file))
                        found_pod5 = True
        
        if not found_pod5:
            print(f"[-] No 'pod5_pass' files found in {strain_id}.")
            return False
            
        print(f"[+] Successfully extracted POD5 files for {strain_id}.")
        return True
    except Exception as e:
        print(f"[-] Extraction failed for {strain_id}: {e}")
        return False
    finally:
        if temp_extract.exists(): shutil.rmtree(temp_extract)
        if tar_path.exists(): tar_path.unlink()  # APAGAR TAR.GZ

def run_dorado(strain_id: str, pod5_dir: Path) -> Path:
    print(f"[*] Running Dorado Basecalling for {strain_id}...")
    tmp_bam = ALIGNMENTS_DIR / f"{strain_id}.tmp.bam"
    final_bam = ALIGNMENTS_DIR / f"{strain_id}.bam"
    
    if not DORADO_BIN.exists():
        print(f"[-] Dorado binary not found at {DORADO_BIN}.")
        raise FileNotFoundError("Dorado not found")
        
    DORADO_BIN.chmod(0o755)
    
    dorado_cmd = [
        str(DORADO_BIN), "basecaller", "hac", str(pod5_dir),
        "--reference", str(REF_FASTA), "--emit-moves", "--emit-sam"
    ]
    samtools_cmd = ["samtools", "sort", "-m", "2G", "-o", str(tmp_bam), "-"]
    
    try:
        p_dorado = subprocess.Popen(dorado_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        p_samtools = subprocess.Popen(samtools_cmd, stdin=p_dorado.stdout, stderr=subprocess.DEVNULL)
        p_dorado.stdout.close()
        
        samtools_exit = p_samtools.wait()
        dorado_exit = p_dorado.wait()
        
        if dorado_exit == 0 and samtools_exit == 0:
            tmp_bam.rename(final_bam)
            subprocess.run(["samtools", "index", str(final_bam)], check=True)
            print(f"[+] Dorado successful for {strain_id}.")
            return final_bam
        else:
            print(f"[-] Dorado/Samtools failed (Exit codes: Dorado {dorado_exit}, Samtools {samtools_exit})")
            if tmp_bam.exists(): tmp_bam.unlink()
            return None
    except Exception as e:
        print(f"[-] IPC Error during basecalling: {e}")
        return None

def main():
    print("=" * 60)
    print("🚀 SEQUENTIAL STREAMING PIPELINE INIT")
    print("=" * 60)
    setup_directories()
    
    with open(INDEX_FILE, 'r') as f:
        strain_ids = [line.strip() for line in f if line.strip()]
    
    appender = FeatureStoreAppender(h5_path=str(MASTER_H5))
    
    for sid in strain_ids:
        print(f"\n--- Processing {sid} ---")
        strain_dir = RAW_DIR / sid
        strain_dir.mkdir(exist_ok=True)
        
        # 1. Download
        if not download_strain(sid, strain_dir):
            continue
            
        # 2. Extract
        if not extract_pod5_pass(sid, strain_dir):
            shutil.rmtree(strain_dir)
            continue
            
        # 3. Basecall (Dorado)
        bam_path = run_dorado(sid, strain_dir)
        if not bam_path:
            shutil.rmtree(strain_dir)
            continue
            
        # 4. Compile Features directly to HDF5
        appender.compile_strain(pod5_dir=strain_dir, bam_path=bam_path, strain_id=sid)
        
        # 5. GARBAGE COLLECTION (Crucial for servers with low disk space)
        print(f"[*] Triggering Garbage Collection for {sid}...")
        shutil.rmtree(strain_dir)
        if bam_path.exists(): bam_path.unlink()
        bam_index = Path(str(bam_path) + ".bai")
        if bam_index.exists(): bam_index.unlink()
        print(f"[+] Memory purged for {sid}.\n")

    print("=" * 60)
    print("✅ DATA PIPELINE COMPLETE. TRIGGERING HYPERPARAMETER SWEEP.")
    print("=" * 60)

    # Triggering Sweep script
    sweep_script = PROJECT_ROOT / "src" / "scripts" / "sweep_h5.py"
    if sweep_script.exists():
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT)
        subprocess.run(["python", str(sweep_script), "--h5_path", str(MASTER_H5)], env=env)
    else:
        print("[-] Sweep script not found.")

if __name__ == "__main__":
    main()
