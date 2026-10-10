from rdkit import Chem
import csv

def sdf_field_smiles_to_csv(
    sdf_path: str,
    csv_path: str,
    field_name: str = "SMILES",
    smiles_column: str = "smiles"
):
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
    smiles_set = set()

    for mol in suppl:
        if mol is None:
            continue
        smi = mol.GetProp(field_name) if mol.HasProp(field_name) else None
        if smi:
            smiles_set.add(smi)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([smiles_column])
        for smi in sorted(smiles_set):
            writer.writerow([smi])

if __name__ == "__main__":
    input_sdf = "coconut_min10atoms_allowed_elements_selected_600000.sdf"
    output_csv = "coconut_min10atoms_allowed_elements_unique_smiles_600000.csv"
    sdf_field_smiles_to_csv(input_sdf, output_csv)
    print(f"Unique SMILES (from field) written to: {output_csv}")