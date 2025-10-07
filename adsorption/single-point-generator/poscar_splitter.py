#!/usr/bin/env python3
import sys
import os
import numpy as np
from pymatgen.io.vasp import Poscar
import importlib.util

def main():
    if len(sys.argv) != 2:
        print("Usage: python poscar_splitter.py <POSCAR_file>")
        sys.exit(1)
    
    poscar_path = sys.argv[1]
    if not os.path.exists(poscar_path):
        print(f"Error: File {poscar_path} not found!")
        sys.exit(1)
    
    # Get the directory of this script to find z_cut.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    z_cut_path = None
    
    # Look for z_cut.py in multiple locations - updated paths
    possible_locations = [
        os.path.join(script_dir, "z_cut.py"),
        os.path.join(script_dir, "..", "threshold_edit", "z_cut.py"),
        os.path.join(script_dir, "..", "bader", "z_cut.py")
    ]
    
    for location in possible_locations:
        normalized_path = os.path.normpath(location)
        if os.path.exists(normalized_path):
            z_cut_path = normalized_path
            break
    
    if not z_cut_path:
        print("Error: z_cut.py not found in expected locations!")
        print("Looked in:")
        for loc in possible_locations:
            print(f"  - {os.path.normpath(loc)}")
        sys.exit(1)
    
    print(f"Using z_cut.py from: {z_cut_path}")
    
    try:
        # Save current directory and change to POSCAR directory
        current_dir = os.getcwd()
        poscar_dir = os.path.dirname(os.path.abspath(poscar_path))
        if poscar_dir:
            os.chdir(poscar_dir)
        
        # Load z_cut.py as a module
        spec = importlib.util.spec_from_file_location("z_cut", z_cut_path)
        z_cut_module = importlib.util.module_from_spec(spec)
        sys.modules["z_cut"] = z_cut_module
        
        # Set the POSCAR file
        z_cut_module.poscar_file = os.path.basename(poscar_path)
        
        # Execute the module to set up the visualization
        spec.loader.exec_module(z_cut_module)
        
        # Show the interactive plot
        z_cut_module.plt.show()
        
        # Get results after user interaction
        z_cutoff = z_cut_module.z_slider.val
        excluded_atoms = z_cut_module.excluded_atoms.copy()
        
        # Get structure info
        structure = z_cut_module.structure
        coords = np.array([site.coords for site in structure])
        
        # Determine surface and molecule atoms
        surf_atoms = []  # Below z-cutoff
        mol_atoms = []   # Above z-cutoff
        
        for i in range(len(coords)):
            if i in excluded_atoms:
                # Excluded atoms go to the opposite side of where they would normally be
                if coords[i, 2] < z_cutoff:
                    # Atom is below z-cutoff, so move it to molecule (above)
                    mol_atoms.append(i)
                else:
                    # Atom is above z-cutoff, so move it to surface (below)
                    surf_atoms.append(i)
            else:
                # Normal assignment based on z-cutoff
                if coords[i, 2] < z_cutoff:
                    surf_atoms.append(i)
                else:
                    mol_atoms.append(i)
        
        # Restore original directory
        os.chdir(current_dir)
        
        # Write results to temporary files
        with open("z_cutoff.tmp", "w") as f:
            f.write(f"{z_cutoff:.6f}")
        
        with open("surf_atoms.tmp", "w") as f:
            for atom in surf_atoms:
                f.write(f"{atom}\n")
        
        with open("mol_atoms.tmp", "w") as f:
            for atom in mol_atoms:
                f.write(f"{atom}\n")
        
        print(f"\nResults:")
        print(f"Z-cutoff: {z_cutoff:.6f} Å")
        print(f"Surface atoms (below z-cutoff): {len(surf_atoms)}")
        print(f"Molecule atoms (above z-cutoff): {len(mol_atoms)}")
        print(f"Excluded atoms: {len(excluded_atoms)}")
        
    except Exception as e:
        print(f"Error during z-cut determination: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
