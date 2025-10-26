#!/usr/bin/env python3
import os
import sys
import importlib.util
import readline
import glob

#------------ Dependency check ----------
import importlib
import subprocess

def check_dependency(dependency):
    try:
        importlib.import_module(dependency)
    except ImportError:
        print(f"{dependency} is not installed. Installing...")
        install_dependency(dependency)

def install_dependency(dependency):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", dependency])
        print(f"{dependency} installed successfully.")
    except subprocess.CalledProcessError:
        print(f"Failed to install {dependency}. Please install it manually.")

# Auto-install required libraries
for pkg in ["matplotlib", "numpy", "pymatgen"]:
    check_dependency(pkg)

# Now import third-party libraries after confirming they're installed
import numpy as np
import matplotlib.pyplot as plt
from pymatgen.io.vasp import Poscar

def input_with_completion(prompt):
    def complete(text, state):
        options = [path for path in glob.glob(text + '*')]
        if state < len(options):
            path = options[state]
            if os.path.isdir(path):
                return path + os.sep  # Add directory separator (/ on Unix, \ on Windows)
            else:
                return path + ' '  # Add space for files
        else:
            return None

    readline.set_completer_delims('\t')
    readline.parse_and_bind("tab: complete")
    readline.set_completer(complete)

    return input(prompt)

def plot_structure(poscar_path):
    """Plot 3D atomic structure with interactive z-cut slider using z_cut.py"""
    # Get the shared z_cut.py location (parent directory of current script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)  # Go up one level from threshold_edit/
    
    # Full path to shared z_cut.py
    z_cut_path = os.path.join(parent_dir, "z_cut.py")
    
    if not os.path.exists(z_cut_path):
        print(f"Warning: z_cut.py not found at {z_cut_path}")
        print("Falling back to basic visualization...")
        
        # Basic visualization as fallback
        pos = Poscar.from_file(poscar_path)
        structure = pos.structure
        coords = np.array([s.coords for s in structure])
        elems = [str(s.specie) for s in structure]

        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        unique_elems = sorted(set(elems))
        colours = plt.cm.tab10(np.linspace(0, 1, len(unique_elems)))
        colour_map = {el: colours[i] for i, el in enumerate(unique_elems)}

        for el in unique_elems:
            idx = [i for i, e in enumerate(elems) if e == el]
            ax.scatter(coords[idx, 0], coords[idx, 1], coords[idx, 2],
                    label=el, s=70, alpha=0.8, color=colour_map[el])
        ax.set_xlabel("x (Å)")
        ax.set_ylabel("y (Å)")
        ax.set_zlabel("z (Å)")
        ax.legend()
        ax.set_title("Atomic positions (rotate to see layers)")
        plt.tight_layout()
        plt.show()
        
        return None, set()
    
    # Load z_cut.py as a module
    try:
        # Save the current working directory
        current_dir = os.getcwd()
        
        # Temporarily change working directory to where the POSCAR is
        poscar_dir = os.path.dirname(os.path.abspath(poscar_path))
        os.chdir(poscar_dir)
        
        # Load z_cut.py as a module
        spec = importlib.util.spec_from_file_location("z_cut", z_cut_path)
        z_cut_module = importlib.util.module_from_spec(spec)
        sys.modules["z_cut"] = z_cut_module
        spec.loader.exec_module(z_cut_module)
        
        # Modify z_cut module's variables
        z_cut_module.poscar_file = os.path.basename(poscar_path)
        
        # Set terminology for threshold editing context
        z_cut_module.action_verb = "remove"
        z_cut_module.action_past_tense = "removed"
        z_cut_module.undo_description = "removal"
        
        # Run the visualization
        z_cut_module.update_plot(z_cut_module.z_cut_guess, True, True)
        plt.show()
        
        # Get the z_cut value and excluded atoms from the slider
        z_cut = z_cut_module.z_slider.val
        excluded_atoms = z_cut_module.excluded_atoms.copy()
        
        # Restore original directory
        os.chdir(current_dir)
        
        return z_cut, excluded_atoms
        
    except Exception as e:
        print(f"Error loading z_cut.py: {e}")
        print("Falling back to basic visualization...")
        # Use the fallback code (same as above)
        pos = Poscar.from_file(poscar_path)
        structure = pos.structure
        coords = np.array([s.coords for s in structure])
        elems = [str(s.specie) for s in structure]

        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        unique_elems = sorted(set(elems))
        colours = plt.cm.tab10(np.linspace(0, 1, len(unique_elems)))
        colour_map = {el: colours[i] for i, el in enumerate(unique_elems)}

        for el in unique_elems:
            idx = [i for i, e in enumerate(elems) if e == el]
            ax.scatter(coords[idx, 0], coords[idx, 1], coords[idx, 2],
                    label=el, s=70, alpha=0.8, color=colour_map[el])
        ax.set_xlabel("x (Å)")
        ax.set_ylabel("y (Å)")
        ax.set_zlabel("z (Å)")
        ax.legend()
        ax.set_title("Atomic positions (rotate to see layers)")
        plt.tight_layout()
        plt.show()
        
        return None, set()

def freeze_custom_region():
    # New: ask operation first, using tab completion for file path
    file_path = input_with_completion("Enter POSCAR file path: ").strip()
    if not os.path.exists(file_path):
        print("File not found.")
        return

    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Detect coordinate mode first and inform user
    coord_mode = None
    coord_mode_index = None
    for i, line in enumerate(lines):
        lower = line.strip().lower()
        if lower == 'direct' or lower == 'cartesian':
            coord_mode = line.strip().lower()  # Lowercase for consistency
            coord_mode_index = i
            print(f"Coordinate mode detected: {line.strip()}")
            break
    
    if coord_mode is None:
        print("Could not find coordinate type ('Direct' or 'Cartesian') in POSCAR.")
        return
        
    # Parse lattice vectors for coordinate conversion if needed
    lattice_vectors = []
    scale_factor = float(lines[1].strip())
    for i in range(2, 5):
        vector = [float(x) * scale_factor for x in lines[i].strip().split()]
        lattice_vectors.append(vector)
    lattice_matrix = np.array(lattice_vectors)

    print("Choose operation:")
    print("1. Fix (freeze) atoms based on threshold")
    print("2. Delete atoms based on threshold (includes manual selection)")
    op_choice = input("Enter 1 or 2: ").strip()
    if op_choice not in ['1', '2']:
        print("Invalid operation choice.")
        return

    # Initialize variables for manual selection
    manually_excluded_atoms = set()
    threshold = None

    # Offer z_cut.py visualization option
    print("Choose threshold determination method:")
    print("1. Enter threshold value manually")
    print("2. Use interactive visualization")
    if op_choice == '2':
        print("   Note: In visualization mode, you can also manually select atoms to delete")
        print("   Instructions: Click atoms to select, Backspace to exclude, Ctrl+Z to undo")
    
    threshold_method = input("Enter 1 or 2: ").strip()
    
    if threshold_method == '2':
        print("Launching interactive visualization...")
        if op_choice == '2':
            print("You can manually exclude atoms in addition to the threshold-based selection")
            print("Excluded atoms will be deleted regardless of their z-position")
        
        result = plot_structure(file_path)
        if result[0] is None:
            print("Interactive threshold determination failed. Falling back to manual entry.")
            threshold_method = '1'
        else:
            threshold, manually_excluded_atoms = result
            print(f"Z-cut value from visualization: {threshold:.4f} Å")
            if manually_excluded_atoms and op_choice == '2':
                print(f"Manually excluded atoms: {len(manually_excluded_atoms)} atoms")
                print(f"Excluded atom indices: {sorted(list(manually_excluded_atoms))}")
    
    if threshold_method == '1':
        try:
            threshold = float(input("Enter threshold value for z in Cartesian coordinates (Å): ").strip())
        except ValueError:
            print("Invalid threshold value.")
            return

    # Get operation name for clearer prompts
    operation_name = "Freeze" if op_choice == '1' else "Delete"
    print("------------------------------------")
    print(f"Choose region criterion: {operation_name}")
    print("1. Atoms with z > threshold")
    print("2. Atoms with z < threshold")
    if op_choice == '2' and manually_excluded_atoms:
        print(f"   Plus {len(manually_excluded_atoms)} manually excluded atoms")
    
    region_choice = input("Enter 1 or 2: ").strip()
    if region_choice not in ['1', '2']:
        print("Invalid region choice.")
        return

    # Detect existing Selective dynamics
    has_selective_dynamics = any(line.strip().lower() == "selective dynamics" for line in lines)

    coord_start = None  # first coordinate line index inside lines

    # Set coordinate start and optionally insert Selective dynamics only for fix op
    if op_choice == '1' and not has_selective_dynamics:  # only insert for fix mode
        lines.insert(coord_mode_index, "Selective dynamics\n")
        coord_start = coord_mode_index + 2
        has_selective_dynamics = True
    else:
        coord_start = coord_mode_index + 1

    # Locate counts line (the last line of integers before coord_mode or Selective dynamics)
    counts_index = None
    counts = []
    for j in range(coord_mode_index - 1, -1, -1):
        tokens = lines[j].split()
        if tokens and all(t.isdigit() for t in tokens):
            counts_index = j
            counts = list(map(int, tokens))
            break

    if op_choice == '2' and counts_index is None:
        print("Could not locate atomic counts line; deletion would corrupt POSCAR. Aborting.")
        return

    # Prepare for processing coordinates
    atom_coords = lines[coord_start:]
    
    # Function to convert Direct to Cartesian coordinates if needed
    def get_cartesian_z(coords, is_direct=False):
        if is_direct:
            # Convert fractional to Cartesian
            # For just the z component, we need the dot product with the third row
            x, y, z = float(coords[0]), float(coords[1]), float(coords[2])
            cart_z = x * lattice_vectors[0][2] + y * lattice_vectors[1][2] + z * lattice_vectors[2][2]
            return cart_z
        else:
            # Already Cartesian
            return float(coords[2])

    if op_choice == '1':
        # Fix (freeze) mode: add T/F flags
        modified_coords = []
        for line in atom_coords:
            parts = line.split()
            if len(parts) < 3:  # blank or non-coordinate
                modified_coords.append(line)
                continue
            try:
                # Get Cartesian z coordinate (converting if needed)
                is_direct = (coord_mode == 'direct')
                z = get_cartesian_z(parts[:3], is_direct)
            except ValueError:
                modified_coords.append(line)
                continue
            condition = (z > threshold) if region_choice == '1' else (z < threshold)
            flags = "F F F" if condition else "T T T"
            
            # Keep original line format but append/replace flags
            if len(parts) >= 6 and parts[3] in ['T', 'F'] and parts[4] in ['T', 'F'] and parts[5] in ['T', 'F']:
                # Already has T/F flags, replace them
                parts[3:6] = flags.split()
                modified_coords.append(" ".join(parts) + "\n")
            else:
                # No T/F flags, add them
                modified_coords.append(f"{parts[0]:>20} {parts[1]:>20} {parts[2]:>20}   {flags}\n")
                
        output_path = file_path + "_fixed.vasp"
        with open(output_path, 'w') as f:
            f.writelines(lines[:coord_start] + modified_coords)
        print(f"Modified POSCAR written to {output_path}")
    else:
        # Delete mode: remove atoms satisfying condition OR manually excluded, update counts
        total_atoms = sum(counts)
        species_boundaries = []
        running = 0
        for c in counts:
            running += c
            species_boundaries.append(running)  # cumulative counts

        kept_coords = []
        removed_per_species = [0] * len(counts)
        atom_index = 0  # index among atoms (excluding blank/comment lines)
        manually_deleted_count = 0
        threshold_deleted_count = 0

        for line in atom_coords:
            parts = line.split()
            if len(parts) < 3:
                # Non-coordinate line (rare after coords start); write as-is then break
                kept_coords.append(line)
                continue
            # Stop if we've already processed declared number of atoms
            if atom_index >= total_atoms:
                kept_coords.append(line)
                continue
            try:
                # Get Cartesian z coordinate (converting if needed)
                is_direct = (coord_mode == 'direct')
                z = get_cartesian_z(parts[:3], is_direct)
            except ValueError:
                kept_coords.append(line)
                continue
                
            # Determine species index from atom_index
            species_idx = 0
            while species_idx < len(species_boundaries) - 1 and atom_index >= species_boundaries[species_idx]:
                species_idx += 1
                
            # Check if atom should be deleted (manually excluded OR threshold condition)
            manually_excluded = atom_index in manually_excluded_atoms
            threshold_condition = (z > threshold) if region_choice == '1' else (z < threshold)
            
            if manually_excluded or threshold_condition:
                removed_per_species[species_idx] += 1
                if manually_excluded:
                    manually_deleted_count += 1
                if threshold_condition:
                    threshold_deleted_count += 1
                # skip adding this coordinate line (deleting atom)
            else:
                kept_coords.append(line)
            atom_index += 1

        # Update counts
        new_counts = [c - r for c, r in zip(counts, removed_per_species)]
        if any(c < 0 for c in new_counts):
            print("Error computing new counts; aborting.")
            return
        lines[counts_index] = " ".join(str(c) for c in new_counts) + "\n"

        output_path = file_path + "_deleted.vasp"
        with open(output_path, 'w') as f:
            f.writelines(lines[:coord_start] + kept_coords)
        
        removed_total = sum(removed_per_species)
        print(f"Deleted {removed_total} atoms total:")
        if manually_excluded_atoms:
            print(f"  - {manually_deleted_count} manually excluded atoms")
            # Account for overlap between manual and threshold deletion
            overlap = manually_deleted_count + threshold_deleted_count - removed_total
            if overlap > 0:
                print(f"  - {threshold_deleted_count - overlap} atoms by threshold (z-cut)")
                print(f"  - {overlap} atoms met both criteria")
            else:
                print(f"  - {threshold_deleted_count} atoms by threshold (z-cut)")
        else:
            print(f"  - All {threshold_deleted_count} atoms deleted by threshold (z-cut)")
        print(f"Modified POSCAR written to {output_path}")

freeze_custom_region()

