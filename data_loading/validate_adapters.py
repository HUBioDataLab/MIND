#!/usr/bin/env python3
"""
Adapter Validation Tool

Comprehensive validation tool for testing data adapters and their universal representation compatibility.
Validates that adapters correctly convert raw data to universal format and ensures all components work together.

Features:
- Tests QM9, LBA, COCONUT, RNA and other dataset adapters
- Validates universal data type compatibility
- Checks entity indexing and block consistency
- Provides detailed statistics and error reporting

Usage:
    python validate_adapters.py                           # Test LBA adapter (default)
    python validate_adapters.py --adapter lba             # Test LBA adapter
    python validate_adapters.py --adapter qm9             # Test QM9 adapter
    python validate_adapters.py --adapter coconut         # Test COCONUT adapter
    python validate_adapters.py --adapter rna             # Test RNA adapter
    python validate_adapters.py --adapter all             # Test all available adapters
    python validate_adapters.py --data_path ./custom/path # Custom data path
    python validate_adapters.py --max_samples 5           # Test more samples
"""

import sys
import os
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

#from data_types import UniversalAtom, UniversalBlock, UniversalMolecule
from data_loading.data_types import UniversalAtom, UniversalBlock, UniversalMolecule

def validate_universal_compatibility(adapter, data_path, adapter_name="Adapter", max_samples=1):
    """Validate that any adapter produces compatible universal representations"""
    
    print(f"Adapter Validation Tool")
    print(f"Testing: {adapter_name}")
    print("=" * 60)
    
    try:
        # Load real data
        raw_data = adapter.load_raw_data(data_path, max_samples=max_samples)
        print(f"Loaded {len(raw_data)} real samples")
        
        # Test first sample
        sample = raw_data[0]
        print(f"Sample: {sample.get('id', 'unknown')}")
        
        # Convert to universal
        universal_mol = adapter.convert_to_universal(sample)
        
        # If testing multiple samples, validate all of them
        comprehensive_stats = None
        if max_samples > 1:
            print(f"\nValidating all {len(raw_data)} samples...")
            all_atom_counts = []
            all_block_counts = []
            all_elements = set()
            all_block_symbols = set()
            sample_details = []
            
            for i, sample in enumerate(raw_data):
                try:
                    mol = adapter.convert_to_universal(sample)
                    atoms = mol.get_all_atoms()
                    all_atom_counts.append(len(atoms))
                    all_block_counts.append(len(mol.blocks))
                    
                    # Collect detailed information
                    elements = [atom.element for atom in atoms]
                    all_elements.update(elements)
                    
                    block_symbols = [block.symbol for block in mol.blocks]
                    all_block_symbols.update(block_symbols)
                    
                    # Store detailed info for first few samples
                    if i < 3:  # Store details for first 3 samples
                        sample_details.append({
                            'id': mol.id,
                            'atoms': len(atoms),
                            'blocks': len(mol.blocks),
                            'elements': elements,
                            'block_symbols': block_symbols,
                            'entity_distribution': {entity: len(mol.get_atoms_by_entity(entity)) for entity in set(atom.entity_idx for atom in atoms)}
                        })
                    
                    if (i + 1) % 100 == 0:
                        print(f"  Processed {i + 1}/{len(raw_data)} samples...")
                        
                except Exception as e:
                    print(f"  Error processing sample {i}: {e}")
                    return False
            
            print(f"  All {len(raw_data)} samples processed successfully!")
            print(f"  Atom count range: {min(all_atom_counts)} - {max(all_atom_counts)}")
            print(f"  Block count range: {min(all_block_counts)} - {max(all_block_counts)}")
            print(f"  Average atoms per molecule: {sum(all_atom_counts) / len(all_atom_counts):.1f}")
            print(f"  Average blocks per molecule: {sum(all_block_counts) / len(all_block_counts):.1f}")
            print(f"  Unique elements found: {sorted(all_elements)}")
            print(f"  Unique block symbols: {sorted(all_block_symbols)}")
            
            # Show detailed information for first few samples
            print(f"\n  Detailed sample information:")
            for i, detail in enumerate(sample_details):
                print(f"    Sample {i+1} ({detail['id']}):")
                print(f"      Atoms: {detail['atoms']}, Blocks: {detail['blocks']}")
                print(f"      Elements: {detail['elements']}")
                print(f"      Block symbols: {detail['block_symbols']}")
                print(f"      Entity distribution: {detail['entity_distribution']}")
            
            # Store comprehensive stats for final summary
            comprehensive_stats = {
                'total_samples': len(raw_data),
                'atom_counts': all_atom_counts,
                'block_counts': all_block_counts,
                'min_atoms': min(all_atom_counts),
                'max_atoms': max(all_atom_counts),
                'avg_atoms': sum(all_atom_counts) / len(all_atom_counts),
                'min_blocks': min(all_block_counts),
                'max_blocks': max(all_block_counts),
                'avg_blocks': sum(all_block_counts) / len(all_block_counts),
                'all_elements': sorted(all_elements),
                'all_block_symbols': sorted(all_block_symbols),
                'sample_details': sample_details
            }
        
        # Test UniversalMolecule structure
        print(f"\nValidating UniversalMolecule structure:")
        print(f"  ID: {universal_mol.id}")
        print(f"  Dataset type: {universal_mol.dataset_type}")
        print(f"  Number of blocks: {len(universal_mol.blocks)}")
        print(f"  Properties: {list(universal_mol.properties.keys())}")
        
        # Validate required fields
        assert isinstance(universal_mol.id, str), "ID must be string"
        assert isinstance(universal_mol.dataset_type, str), "Dataset type must be string"
        assert isinstance(universal_mol.blocks, list), "Blocks must be list"
        assert isinstance(universal_mol.properties, dict), "Properties must be dict"
        print("  UniversalMolecule structure valid")
        
        # Test UniversalBlock structure
        print(f"\nValidating UniversalBlock structure:")
        print(f"  Total blocks: {len(universal_mol.blocks)}")
        for i, block in enumerate(universal_mol.blocks[:5]):  # Show first 5 blocks
            print(f"  Block {i}:")
            print(f"    Symbol: {block.symbol}")
            print(f"    Number of atoms: {len(block.atoms)}")
            print(f"    Block length method: {len(block)}")
            if len(block.atoms) > 0:
                print(f"    First atom: {block.atoms[0].element} at {block.atoms[0].position}")
                print(f"    Last atom: {block.atoms[-1].element} at {block.atoms[-1].position}")
            
            # Validate block structure
            assert isinstance(block.symbol, str), f"Block {i} symbol must be string"
            assert isinstance(block.atoms, list), f"Block {i} atoms must be list"
            assert len(block) == len(block.atoms), f"Block {i} length method mismatch"
        
        if len(universal_mol.blocks) > 5:
            print(f"  ... and {len(universal_mol.blocks) - 5} more blocks")
        print("  UniversalBlock structure valid")
        
        # Test UniversalAtom structure
        print(f"\nValidating UniversalAtom structure:")
        all_atoms = universal_mol.get_all_atoms()
        print(f"  Total atoms: {len(all_atoms)}")
        
        if all_atoms:
            # Show first few atoms in detail
            for i, atom in enumerate(all_atoms[:3]):
                print(f"  Atom {i}:")
                print(f"    Element: {atom.element}")
                print(f"    Position: {atom.position}")
                print(f"    Pos code: {atom.pos_code}")
                print(f"    Block idx: {atom.block_idx}")
                print(f"    Atom idx in block: {atom.atom_idx_in_block}")
                print(f"    Entity idx: {atom.entity_idx}")
                
                # Validate atom structure
                assert isinstance(atom.element, str), "Element must be string"
                assert isinstance(atom.position, tuple), "Position must be tuple"
                assert len(atom.position) == 3, "Position must have 3 coordinates"
                assert isinstance(atom.pos_code, str), "Pos code must be string"
                assert isinstance(atom.block_idx, int), "Block idx must be int"
                assert isinstance(atom.atom_idx_in_block, int), "Atom idx in block must be int"
                assert isinstance(atom.entity_idx, int), "Entity idx must be int"
            
            if len(all_atoms) > 3:
                print(f"  ... and {len(all_atoms) - 3} more atoms")
                
            # Show element distribution
            element_counts = {}
            for atom in all_atoms:
                element_counts[atom.element] = element_counts.get(atom.element, 0) + 1
            print(f"  Element distribution: {element_counts}")
            
        print("  UniversalAtom structure valid")
        
        # Test entity indexing
        print(f"\nValidating entity indexing:")
        entity_counts = {}
        for atom in all_atoms:
            entity_counts[atom.entity_idx] = entity_counts.get(atom.entity_idx, 0) + 1
        print(f"  Entity distribution: {entity_counts}")
        
        # Test entity separation methods
        protein_atoms = universal_mol.get_atoms_by_entity(0)
        ligand_atoms = universal_mol.get_atoms_by_entity(1)
        print(f"  Entity 0 atoms: {len(protein_atoms)}")
        print(f"  Entity 1 atoms: {len(ligand_atoms)}")
        
        # Test block separation by entity
        protein_blocks = universal_mol.get_blocks_by_entity(0)
        ligand_blocks = universal_mol.get_blocks_by_entity(1)
        print(f"  Entity 0 blocks: {len(protein_blocks)}")
        print(f"  Entity 1 blocks: {len(ligand_blocks)}")
        
        # Validate entity indexing (flexible for different datasets)
        if len(all_atoms) == 0:
            print("  Warning: Empty molecule detected (no atoms)")
            print("  This may indicate processing issues with the molecule")
            print("  Entity indexing skipped for empty molecule")
        else:
            assert len(protein_atoms) > 0, "Should have entity 0 atoms"
            if len(ligand_atoms) > 0:  # Some datasets might only have one entity
                print("  Multi-entity dataset detected")
            else:
                print("  Single-entity dataset detected")
            print("  Entity indexing valid")
        
        # Test block index consistency
        print(f"\nValidating block index consistency:")
        if len(universal_mol.blocks) > 0:
            for block_idx, block in enumerate(universal_mol.blocks):
                for atom in block.atoms:
                    assert atom.block_idx == block_idx, f"Block index mismatch: {atom.block_idx} != {block_idx}"
            print("  Block index consistency valid")
        else:
            print("  No blocks to validate")
        
        # Test atom index consistency within blocks
        print(f"\nValidating atom index consistency within blocks:")
        if len(universal_mol.blocks) > 0:
            for block_idx, block in enumerate(universal_mol.blocks):
                for atom_idx, atom in enumerate(block.atoms):
                    assert atom.atom_idx_in_block == atom_idx, f"Atom index mismatch in block {block_idx}: {atom.atom_idx_in_block} != {atom_idx}"
            print("  Atom index consistency valid")
        else:
            print("  No blocks to validate")
        
        # Final validation
        print(f"\nFinal Validation:")
        assert isinstance(universal_mol, UniversalMolecule), "Should be UniversalMolecule"
        if len(universal_mol.blocks) > 0:
            assert isinstance(universal_mol.blocks[0], UniversalBlock), "Should be UniversalBlock"
        if len(all_atoms) > 0:
            assert isinstance(all_atoms[0], UniversalAtom), "Should be UniversalAtom"
        
        print("ALL UNIVERSAL REPRESENTATION COMPATIBILITY TESTS PASSED!")
        print(f"Summary:")
        
        if comprehensive_stats:
            # Show comprehensive statistics for multiple samples
            print(f"  - Total samples tested: {comprehensive_stats['total_samples']}")
            print(f"  - Atom count range: {comprehensive_stats['min_atoms']} - {comprehensive_stats['max_atoms']}")
            print(f"  - Block count range: {comprehensive_stats['min_blocks']} - {comprehensive_stats['max_blocks']}")
            print(f"  - Average atoms per molecule: {comprehensive_stats['avg_atoms']:.1f}")
            print(f"  - Average blocks per molecule: {comprehensive_stats['avg_blocks']:.1f}")
            print(f"  - All unique elements: {comprehensive_stats['all_elements']}")
            print(f"  - All unique block symbols: {comprehensive_stats['all_block_symbols']}")
            print(f"  - First sample details:")
            print(f"    * Atoms: {len(all_atoms)}")
            print(f"    * Blocks: {len(universal_mol.blocks)}")
            print(f"    * Entity 0 atoms: {len(protein_atoms)}")
            print(f"    * Entity 1 atoms: {len(ligand_atoms)}")
            print(f"    * Entity 0 blocks: {len(protein_blocks)}")
            print(f"    * Entity 1 blocks: {len(ligand_blocks)}")
        else:
            # Show single sample statistics
            print(f"  - Total atoms: {len(all_atoms)}")
            print(f"  - Total blocks: {len(universal_mol.blocks)}")
            print(f"  - Entity 0 atoms: {len(protein_atoms)}")
            print(f"  - Entity 1 atoms: {len(ligand_atoms)}")
            print(f"  - Entity 0 blocks: {len(protein_blocks)}")
            print(f"  - Entity 1 blocks: {len(ligand_blocks)}")
        
        return True
        
    except Exception as e:
        print(f"Compatibility validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def get_adapter(adapter_name):
    """Get adapter instance by name"""
    if adapter_name.lower() == 'lba':
        from adapters.lba_adapter import LBAAdapter
        return LBAAdapter()
    elif adapter_name.lower() == 'qm9':
        from adapters.qm9_adapter import QM9Adapter
        return QM9Adapter()
    elif adapter_name.lower() == 'coconut':
        from adapters.coconut_adapter import COCONUTAdapter
        return COCONUTAdapter()
    elif adapter_name.lower() == 'pdb':
        # TODO: Add PDB adapter when available
        raise NotImplementedError("PDB adapter not yet implemented")
    elif adapter_name.lower() == 'rna':  
        from adapters.rna_adapter import RNAAdapter  
        return RNAAdapter() 
    else:
        raise ValueError(f"Unknown adapter: {adapter_name}")

def get_default_data_path(adapter_name):
    """Get default data path for adapter"""
    if adapter_name.lower() == 'lba':
        return "./data/LBA"
    elif adapter_name.lower() == 'qm9':
        return "./data/qm9"
    elif adapter_name.lower() == 'coconut':
        return "./data"
    elif adapter_name.lower() == 'pdb':
        return "./data/pdb"
    elif adapter_name.lower() == 'rna':  
        return "./data/filtered_rna_cifs"
    else:
        raise ValueError(f"No default path for adapter: {adapter_name}")

def validate_adapter(adapter_name, data_path=None, max_samples=1):
    """Validate a specific adapter"""
    try:
        print(f"Validating {adapter_name.upper()} Adapter...")
        
        # Get adapter instance
        adapter = get_adapter(adapter_name)
        
        # Use default data path if not provided
        if data_path is None:
            data_path = get_default_data_path(adapter_name)
        
        # Run validation
        success = validate_universal_compatibility(
            adapter, 
            data_path, 
            f"{adapter_name.upper()} Adapter",
            max_samples
        )
        
        if success:
            print(f"\n{adapter_name.upper()} adapter validation PASSED!")
            print("Adapter is fully compatible with universal representation!")
            return True
        else:
            print(f"\n{adapter_name.upper()} adapter validation FAILED!")
            return False
            
    except Exception as e:
        print(f"Validation error for {adapter_name}: {e}")
        return False

def main():
    """Main validation function"""
    parser = argparse.ArgumentParser(description='Validate data adapters and universal representation compatibility')
    parser.add_argument('--adapter', type=str, default='lba',
                       choices=['lba', 'qm9', 'coconut', 'pdb', 'rna', 'all'],
                       help='Adapter to test (default: lba)')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to dataset data (uses default if not provided)')
    parser.add_argument('--max_samples', type=int, default=1,
                       help='Maximum number of samples to test (default: 1)')
    
    args = parser.parse_args()
    
    print("Running Universal Representation Validation...")
    
    if args.adapter == 'all':
        # Test all available adapters
        adapters = ['lba', 'qm9', 'coconut', 'rna']  # Add more as they become available
        all_passed = True
        
        for adapter_name in adapters:
            print(f"\n{'='*80}")
            success = validate_adapter(adapter_name, args.data_path, args.max_samples)
            if not success:
                all_passed = False
        
        if all_passed:
            print(f"\n{'='*80}")
            print("ALL ADAPTERS VALIDATION PASSED!")
            return 0
        else:
            print(f"\n{'='*80}")
            print("SOME ADAPTERS VALIDATION FAILED!")
        return 1
    else:
        # Test single adapter
        success = validate_adapter(args.adapter, args.data_path, args.max_samples)
        return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)