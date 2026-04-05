import os
from pathlib import Path
from collections import Counter
from Bio.PDB import MMCIFParser
from tqdm import tqdm  # Progress bar library

# Configuration
DATA_PATH = "raw_structures"
OUTPUT_FILE = "rna_atom_analysis_report.txt"
TARGET_RESIDUES = ['A', 'U', 'G', 'C']

def run_atom_analysis():
    parser = MMCIFParser(QUIET=True)
    # Dictionary to store atom counts: { 'A': Counter({22: 500, 23: 10}), ... }
    stats = {res: Counter() for res in TARGET_RESIDUES}
    
    data_dir = Path(DATA_PATH)
    if not data_dir.exists():
        print(f"Error: Directory '{DATA_PATH}' not found.")
        return

    cif_files = list(data_dir.glob("*.cif"))
    total_files = len(cif_files)
    print(f"Found {total_files} files. Starting analysis...")

    processed_count = 0
    error_count = 0

    # Wrap the loop with tqdm for a visual progress bar
    for cif_file in tqdm(cif_files, desc="Analyzing RNA Structures", unit="file"):
        try:
            structure = parser.get_structure(cif_file.stem, str(cif_file))
            for model in structure:
                for chain in model:
                    for residue in chain:
                        res_name = residue.get_resname().strip()
                        
                        if res_name in TARGET_RESIDUES:
                            # Count atoms excluding Hydrogens
                            atoms = [a for a in residue if a.element != 'H']
                            stats[res_name][len(atoms)] += 1
            processed_count += 1
        except Exception:
            error_count += 1
            continue

    # Write results to TXT
    print(f"\n Writing results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("MIND Project - RNA Atom Variance Analysis\n")
        f.write("=========================================\n\n")
        f.write(f"Total files processed: {processed_count}\n")
        f.write(f"Files with errors: {error_count}\n\n")

        for res in TARGET_RESIDUES:
            f.write(f"Residue: {res}\n")
            f.write("-" * 20 + "\n")
            
            res_counts = stats[res]
            total_instances = sum(res_counts.values())
            
            if total_instances == 0:
                f.write("No instances found.\n\n")
                continue

            # Sort by atom count for readability
            sorted_counts = sorted(res_counts.items())
            
            for atom_num, freq in sorted_counts:
                percentage = (freq / total_instances) * 100
                f.write(f"  {atom_num} atoms: {freq:>6} instances ({percentage:>6.2f}%)\n")
            
            # Highlight potential leakage/confusion points
            if len(res_counts) > 1:
                f.write(f"  [!] Variance detected: {len(res_counts)} different atom configurations.\n")
            
            f.write(f"  Total {res} samples: {total_instances}\n\n")

    print(f"Analysis complete! Check '{OUTPUT_FILE}' for details.")

if __name__ == "__main__":
    run_atom_analysis()