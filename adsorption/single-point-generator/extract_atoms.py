#!/usr/bin/env python3
import sys
import os
from pymatgen.io.vasp import Poscar

def main():
    if len(sys.argv) != 4:
        print("Usage: python extract_atoms.py <input_POSCAR> <atom_indices_file> <output_POSCAR>")
        sys.exit(1)
    
    input_poscar = sys.argv[1]
    indices_file = sys.argv[2]
    output_poscar = sys.argv[3]
    
    # Check input files exist
    if not os.path.exists(input_poscar):
        print(f"Error: Input POSCAR {input_poscar} not found!")
        sys.exit(1)
    
    if not os.path.exists(indices_file):
        print(f"Error: Indices file {indices_file} not found!")
        sys.exit(1)
    
    # Read atom indices
    with open(indices_file, 'r') as f:
        lines = f.readlines()
    
    if not lines:
        print(f"Warning: No atoms specified in {indices_file}")
        # Create empty POSCAR or copy original structure without atoms
        poscar = Poscar.from_file(input_poscar)
        # Create structure with no atoms
        from pymatgen.core.structure import Structure
        empty_structure = Structure(
            lattice=poscar.structure.lattice,
            species=[],
            coords=[],
            coords_are_cartesian=True
        )
        empty_poscar = Poscar(empty_structure)
        empty_poscar.write_file(output_poscar)
        return
    
    atom_indices = []
    for line in lines:
        line = line.strip()
        if line:
            try:
                atom_indices.append(int(line))
            except ValueError:
                print(f"Warning: Invalid atom index '{line}' - skipping")
    
    if not atom_indices:
        print("Error: No valid atom indices found!")
        sys.exit(1)
    
    # Read original POSCAR
    poscar = Poscar.from_file(input_poscar)
    structure = poscar.structure
    
    # Validate indices
    max_index = len(structure) - 1
    valid_indices = []
    for idx in atom_indices:
        if 0 <= idx <= max_index:
            valid_indices.append(idx)
        else:
            print(f"Warning: Atom index {idx} out of range (0-{max_index}) - skipping")
    
    if not valid_indices:
        print("Error: No valid atom indices after filtering!")
        sys.exit(1)
    
    # Extract selected sites
    selected_sites = [structure[i] for i in valid_indices]
    
    # Create new structure with selected atoms
    from pymatgen.core.structure import Structure
    new_structure = Structure(
        lattice=structure.lattice,
        species=[site.specie for site in selected_sites],
        coords=[site.coords for site in selected_sites],
        coords_are_cartesian=True
    )
    
    # Create new POSCAR
    new_poscar = Poscar(new_structure)
    
    # Preserve selective dynamics if present
    if hasattr(poscar, 'selective_dynamics') and poscar.selective_dynamics is not None:
        new_selective_dynamics = [poscar.selective_dynamics[i] for i in valid_indices]
        new_poscar.selective_dynamics = new_selective_dynamics
    
    # Write output POSCAR
    new_poscar.write_file(output_poscar)
    
    print(f"Created {output_poscar} with {len(valid_indices)} atoms")

if __name__ == "__main__":
    main()
