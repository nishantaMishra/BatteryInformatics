#!/usr/bin/env python3
import os
import re
import sys
import readline
import glob
import importlib
import subprocess

# ANSI color formatting for terminal output
RED = '\033[91m'
BOLD_RED = '\033[1;91m'
RESET = '\033[0m'

#------------ Dependency check ----------
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
import importlib.util

# Function to enable tab-completion for file paths (tested only in linux)
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

# ---------- Helper functions ----------

def run_cmd(cmd, workdir):
    """Run a shell command in given directory, returns True if successful."""
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, cwd=workdir, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Command failed: {e}")
        return False

def file_exists(path):
    return os.path.isfile(path)

def check_required_files(calc_dir):
    """Check if POTCAR, POSCAR, ACF.dat exist. If not, try to create ACF.dat."""
    potcar = os.path.join(calc_dir, "POTCAR")
    poscar = os.path.join(calc_dir, "POSCAR")
    acf = os.path.join(calc_dir, "ACF.dat")
    chgsum = os.path.join(os.path.dirname(__file__), "chgsum.pl")
    bader_exec = os.path.join(os.path.dirname(__file__), "bader")
    
    # First check for all needed input files
    aeccar0 = os.path.join(calc_dir, "AECCAR0")
    aeccar2 = os.path.join(calc_dir, "AECCAR2")
    chgcar = os.path.join(calc_dir, "CHGCAR")
    
    missing_files = []
    for file_path in [potcar, poscar]:
        if not file_exists(file_path):
            missing_files.append(os.path.basename(file_path))
    
    # Required for Bader analysis generation
    if not file_exists(acf):
        aeccar_available = file_exists(aeccar0) and file_exists(aeccar2)
        chgcar_available = file_exists(chgcar)
        
        if not aeccar_available and not chgcar_available:
            missing_charge_files = []
            for charge_file in [chgcar, aeccar0, aeccar2]:
                if not file_exists(charge_file):
                    missing_charge_files.append(os.path.basename(charge_file))
            print(f"{RED}Cannot generate Bader charge analysis: Missing required charge files: {BOLD_RED}{', '.join(missing_charge_files)}{RESET}")
            if missing_files:
                print(f"{RED}Also missing other required files: {BOLD_RED}{', '.join(missing_files)}{RESET}")
            print(f"{RED}Please make sure all required files are in the calculation directory.{RESET}")
            return None, None, None
            
        # Try to generate the needed files
        success = True
        
        # Try sophisticated method first if AECCAR files are available
        if aeccar_available:
            print("Generating Bader analysis using sophisticated method with AECCAR files...")
            if not file_exists(os.path.join(calc_dir, "CHGCAR_sum")):
                success = run_cmd(["perl", chgsum, "AECCAR0", "AECCAR2"], calc_dir)
                if not success:
                    print(f"{RED}Failed to generate CHGCAR_sum from AECCAR files.{RESET}")
                    # Don't return yet, try the minimal method as fallback if CHGCAR is available
            
            if success:
                success = run_cmd([bader_exec, "CHGCAR", "-ref", "CHGCAR_sum"], calc_dir)
                if not success:
                    print(f"{RED}Failed to generate Bader analysis with sophisticated method.{RESET}")
                    # Continue to try minimal method if CHGCAR is available
        
        # If sophisticated method failed or AECCAR files aren't available, try minimal method
        if (not aeccar_available or not success) and chgcar_available:
            print("Falling back to minimal Bader calculation method using only CHGCAR...")
            success = run_cmd([bader_exec, "CHGCAR"], calc_dir)
            if not success:
                print(f"{RED}Failed to generate Bader analysis with minimal method.{RESET}")
                return None, None, None
    
    # Final check if all needed files exist
    if not all([file_exists(potcar), file_exists(poscar), file_exists(acf)]):
        still_missing = []
        for check_file in [(potcar, "POTCAR"), (poscar, "POSCAR"), (acf, "ACF.dat")]:
            if not file_exists(check_file[0]):
                still_missing.append(check_file[1])
        print(f"{RED}Still missing required files after generation attempt: {BOLD_RED}{', '.join(still_missing)}{RESET}")
        return None, None, None
        
    return potcar, poscar, acf

def parse_zval(potcar):
    """Extract ZVAL values from POTCAR, robust to trailing semicolons."""
    zvals = []
    elems = []
    with open(potcar) as f:
        for line in f:
            # TITEL lines look like: "TITEL  = PAW_PBE Cs_sv 08Apr2002"
            # We need to pick out the element token (e.g. 'Cs_sv') not the date token.
            # handle lines with leading whitespace
            if line.strip().startswith("TITEL"):
                # take right-hand side of '=' and scan tokens for one that looks like an element
                rhs = line.split('=', 1)[1].strip()
                for tok in rhs.split():
                    base = tok.split('_')[0]
                    # prefer base tokens matching a chemical symbol (1 or 2 letters)
                    if re.fullmatch(r'[A-Z][a-z]?$', base):
                        elems.append(base)
                        break
                else:
                    # fallback: take the base of the first token after '='
                    elems.append(rhs.split()[0].split('_')[0])
            if "ZVAL" in line:
                # Extract the number immediately after 'ZVAL' to avoid capturing POMASS
                m = re.search(r"ZVAL\s*=\s*([-+]?\d*\.\d+|\d+)", line)
                if m:
                    zvals.append(float(m.group(1)))

    if len(elems) != len(zvals):
        # It's better to warn early — caller can decide how to handle mismatches
        print(f"Warning: parsed {len(elems)} element tokens but found {len(zvals)} ZVAL entries in {potcar}")

    return dict(zip(elems, zvals))


def read_acf(acf_file):
    """Read Bader CHARGE values."""
    charges = []
    with open(acf_file) as f:
        for line in f:
            # Split the line into tokens. This handles leading whitespace
            # and ignores header/separator lines that don't have the expected columns.
            tokens = line.split()
            # Expect at least 5 columns: index, X, Y, Z, CHARGE, ...
            if len(tokens) >= 5 and tokens[0].lstrip('-').isdigit():
                try:
                    charges.append(float(tokens[4]))
                except ValueError:
                    # If conversion fails, skip the line
                    continue
    return np.array(charges)

def plot_structure(poscar_path):
    """Plot 3D atomic structure with interactive z-cut slider using z_cut.py"""
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Full path to z_cut.py
    z_cut_path = os.path.join(script_dir, "z_cut.py")
    
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
        
        z_cut = float(input("Enter z-cutoff (Å): "))
        return structure, z_cut
    
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
        
        # Run the visualization
        z_cut_module.update_plot(z_cut_module.z_cut_guess, True, True)
        plt.show()
        
        # Get the z_cut value from the slider
        z_cut = z_cut_module.z_slider.val
        
        # Get the structure
        structure = z_cut_module.structure
        
        # Restore original directory
        os.chdir(current_dir)
        
        return structure, z_cut
        
    except Exception as e:
        print(f"Error loading z_cut.py: {e}")
        print("Falling back to basic visualization...")
        # Use the fallback code (same as above)
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
        
        z_cut = float(input("Enter z-cutoff (Å): "))
        return structure, z_cut

# ---------- Main routine ----------

def main():
    calc_dir = input_with_completion("Enter the calculation directory path: ").strip()
    if not os.path.isdir(calc_dir):
        print("Invalid directory.")
        return

    potcar, poscar, acf = check_required_files(calc_dir)
    if potcar is None or poscar is None or acf is None:
        print(f"{RED}Could not proceed with Bader analysis due to missing files.{RESET}")
        return
        
    print("Files ready:", potcar, poscar, acf)

    # Plot and get z_cut
    structure, z_cut = plot_structure(poscar)

    # Identify adsorbate atoms
    coords = np.array([s.coords for s in structure])
    elems = [str(s.specie) for s in structure]
    ads_indices = [i for i, c in enumerate(coords[:, 2]) if c > z_cut]
    ads_elems = [elems[i] for i in ads_indices]

    # Print composition summary
    unique, counts = np.unique(ads_elems, return_counts=True)
    print("\nAdsorbate atoms above z_cut:")
    for e, c in zip(unique, counts):
        print(f"  {e}: {c}")

    # Read ZVAL and Bader data
    zval_map = parse_zval(potcar)
    bader_charges = read_acf(acf)
    # Debug output: print sizes and check for mismatches
    # print(f"\nDebug: read {len(bader_charges)} Bader entries from {acf}")
    # print(f"Debug: found {len(elems)} atoms in POSCAR ({poscar})")

    if len(bader_charges) == 0:
        print("ERROR: No Bader charges read from ACF.dat. Aborting detailed adsorbate output.")
        return

    # Calculate and print per-adsorbate charges (from ACF.dat), then sums and ZVALs
    Q_total = 0.0
    sum_bader_ads = 0.0
    sum_zval_ads = 0.0

    print('\nPer-adsorbate charges (from ACF.dat)')
    print(' idx   elem   ZVAL    N_bader    q = ZVAL - N_bader')
    for i in ads_indices:
        if i < 0 or i >= len(bader_charges):
            print(f'  {i:4d}  {elems[i]:4s}   WARNING: index out of range for ACF.dat (skipped)')
            continue
        elem = elems[i]
        zval = zval_map.get(elem, 0.0)
        N_bader = bader_charges[i]
        q = zval - N_bader
        sum_bader_ads += N_bader
        sum_zval_ads += zval
        Q_total += q
        print(f' {i:4d}   {elem:4s}  {zval:6.3f}  {N_bader:9.6f}  {q:12.6f}')

    print(f"\nSum of adsorbate Bader charges (ACF.dat) = {sum_bader_ads:.6f} e")
    print(f"Sum of adsorbate ZVAL (from POTCAR) = {sum_zval_ads:.6f} e")
    print(f"Net adsorbate charge (ZVAL - Bader) = {Q_total:.6f} e")
    print(f"(Positive → electron loss, Negative → electron gain)")

if __name__ == "__main__":
    main()
    main()
