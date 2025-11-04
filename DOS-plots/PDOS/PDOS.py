# Working as intended on Sun Aug 17 06:43:49 PM EDT 2025
# The program expects vaspkit installed in the system.

#------------ Dependency check ----------
import importlib
import subprocess
import sys

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
dependencies = {
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "yaml": "pyyaml",  # Module name: package name
    "h5py": "h5py"  # Optional: for HDF5 file support
}

for module_name, package_name in dependencies.items():
    try:
        importlib.import_module(module_name)
    except ImportError:
        if module_name == "h5py":
            print(f"Warning: {package_name} is not installed. HDF5 support will be unavailable.")
            print(f"Install it with: pip install {package_name}")
        else:
            print(f"{package_name} is not installed. Installing...")
            install_dependency(package_name)

#-------- Imports -----------
import os
import glob
import readline
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
import argparse

# Try importing h5py for HDF5 support
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

# supress the warmings
desired_fonts = ['Noto Sans Devanagari', 'DejaVu Sans']
installed_font_names = {f.name.lower() for f in fm.fontManager.ttflist}
valid_fonts = [f for f in desired_fonts if f.lower() in installed_font_names]

if not valid_fonts:
    # Ensure at least a safe default to avoid "Font Family ... not found." messages
    valid_fonts = ['DejaVu Sans']

plt.rcParams['font.family'] = valid_fonts

# Bandgap detection functions adapted from TDOS5.py
def is_zero(x, tol=1e-5):
    return abs(x) < tol

# Legacy kept for compatibility (unused now)
def detect_bandgap_exact(energy, tdos_up, tdos_down):
    return None

# --- Advanced bandgap detection utilities ---

def _find_zero_intervals(energy, tdos_up, tdos_down, zero_tol=1e-5):
    intervals = []
    in_zero = False
    start_e = None
    prev_e = None
    for e, u, d in zip(energy, tdos_up, tdos_down):
        if abs(u) < zero_tol and abs(d) < zero_tol:
            if not in_zero:
                in_zero = True
                start_e = e
        else:
            if in_zero and prev_e is not None:
                intervals.append((start_e, prev_e))
                in_zero = False
        prev_e = e
    if in_zero and prev_e is not None:
        intervals.append((start_e, prev_e))
    return [(s, e) for s, e in intervals if e > s]

def detect_bandgap_advanced(energy, tdos_up, tdos_down, zero_tol=1e-5,
                            min_gap=0.1, max_gap=7.0, fermi_window=1.0):
    intervals = _find_zero_intervals(energy, tdos_up, tdos_down, zero_tol=zero_tol)
    if not intervals:
        return None
    candidates = []
    for s, e in intervals:
        w = e - s
        if min_gap <= w <= max_gap:
            candidates.append((s, e, w))
    if not candidates:
        return None
    fermi_gaps = [c for c in candidates if c[0] <= 0 <= c[1]]
    if fermi_gaps:
        fermi_gaps.sort(key=lambda x: (-x[2], abs((x[0]+x[1]) / 2)))
        s, e, w = fermi_gaps[0]
        return {'vbm': s, 'cbm': e, 'width': w, 'type': 'fermi_gap', 'message': 'Bandgap spans the Fermi level.'}
    # Edge-based offset gap: accept if either edge within ±fermi_window
    edge_candidates = []
    for s, e, w in candidates:
        v_dist = abs(s)
        c_dist = abs(e)
        edge_min = min(v_dist, c_dist)
        if edge_min <= fermi_window:
            edge_candidates.append((s, e, w, edge_min))
    if edge_candidates:
        edge_candidates.sort(key=lambda x: (x[3], -x[2]))
        s, e, w, _ = edge_candidates[0]
        return {'vbm': s, 'cbm': e, 'width': w, 'type': 'offset_gap',
                'message': 'Caution: Either the compound is metallic or there is bandgap shift.'}
    return None

def analyze_tdos_bandgap(energy, tdos_up, tdos_down, symmetry_tol=0.01):
    print("\n----- TDOS Analysis Results -----")
    
    # Check if data is spin-polarized by comparing if up and down are identical arrays
    is_spin_polarized = not np.array_equal(tdos_up, tdos_down)
    
    if is_spin_polarized:
        max_dos = max(np.max(np.abs(tdos_up)), np.max(np.abs(tdos_down)), 1e-12)
        sym_diff = np.mean(np.abs(tdos_up + tdos_down)) / max_dos
        if sym_diff < symmetry_tol:
            print("✅ Non-magnetic (TDOS-UP and DOWN are symmetric).")
        else:
            print("⚠️  Magnetic (TDOS-UP and DOWN differ noticeably).")
    else:
        print("ℹ️  Non-spin-polarized calculation detected.")
    
    gap = detect_bandgap_advanced(energy, tdos_up, tdos_down)
    if gap:
        if gap['type'] == 'fermi_gap':
            print(f"✅ Bandgap detected: {gap['vbm']:.3f} to {gap['cbm']:.3f} eV → Width: {gap['width']:.3f} eV")
        else:
            print(f"⚠️  Offset bandgap candidate: {gap['vbm']:.3f} to {gap['cbm']:.3f} eV → Width: {gap['width']:.3f} eV")
            print("    Note: Fermi lies outside this zero-DOS interval.")
        print(f"    {gap['message']}")
    else:
        print("⚠️  No bandgap detected (metallic or outside 0.1–7 eV range).\n")
    print("----------------------------------\n")
    return gap

# Import color manager
try:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from color_manager import apply_color_scheme, get_available_schemes, list_schemes_info
except ImportError:
    print("Warning: Could not import color manager. Color schemes may not work properly.")
    def apply_color_scheme(elements, plotting_info, scheme_name):
        return {}
    def get_available_schemes():
        return ['vesta']
    def list_schemes_info():
        print("Color scheme information not available.")

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

def read_pdos_files(location):
    pdos_files = {}
    for file_name in os.listdir(location):
        if not (file_name.startswith("PDOS_") and file_name.endswith(".dat")):
            continue

        parts = file_name.split("_")
        if len(parts) >= 3:
            element = parts[1]
            spin = parts[2].split(".")[0].upper()
            spin = 'DOWN' if spin == 'DW' else spin
        elif len(parts) == 2:
            element = parts[1].split(".")[0]
            spin = "UP"
        else:
            continue

        if element not in pdos_files:
            pdos_files[element] = {}

        full_path = os.path.join(location, file_name)
        with open(full_path, 'r') as f:
            next(f)
            pdos_files[element][spin] = [line.strip().split() for line in f if line.strip()]

    return pdos_files

def try_generate_tdos_dat_file(location):
    """Generate TDOS.dat file using vaspkit when needed"""
    required_files = ['INCAR', 'DOSCAR', 'PROCAR']
    if all(os.path.isfile(os.path.join(location, f)) for f in required_files):
        print("TDOS.dat not found. Trying to generate it using vaspkit...")
        try:
            subprocess.run('(echo 11; sleep 1; echo 111) | vaspkit', shell=True, cwd=location, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to run vaspkit for TDOS generation in {location}: {e}")
    else:
        print("Missing INCAR/DOSCAR/PROCAR files. Cannot run vaspkit for TDOS.")
    return False

def try_generate_pdos_dat_files(location):
    required_files = ['INCAR', 'DOSCAR', 'PROCAR']
    if all(os.path.isfile(os.path.join(location, f)) for f in required_files):
        print("PDOS files not found. Trying to generate them using vaspkit...")
        try:
            subprocess.run('(echo 11; sleep 1; echo 113) | vaspkit', shell=True, cwd=location, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to run vaspkit in {location}: {e}")
    else:
        print("Missing INCAR/DOSCAR/PROCAR files. Cannot run vaspkit.")
    return False

def read_tdos_file(location):
    """Read TDOS.dat file similar to TDOS5.py, with HDF5 fallback"""
    # First try to read from HDF5 if available
    tdos_energy, tdos_up, tdos_down = read_tdos_from_hdf5(location)
    if tdos_energy is not None:
        return tdos_energy, tdos_up, tdos_down
    
    # Fall back to reading TDOS.dat
    tdos_path = os.path.join(location, 'TDOS.dat')
    try:
        data = np.loadtxt(tdos_path)
        if data.shape[1] == 3:
            # Spin-polarized: energy, up, down
            energy = data[:, 0]
            tdos_up = data[:, 1]
            tdos_down = data[:, 2]
        elif data.shape[1] == 2:
            # Non-magnetic: energy, total only
            energy = data[:, 0]
            tdos_up = data[:, 1]
            tdos_down = data[:, 1]
        else:
            return None, None, None
        return energy, tdos_up, tdos_down
    except Exception as e:
        print(f"Warning: Could not read TDOS.dat: {e}")
        return None, None, None

def read_pdos_from_hdf5(location):
    """Read PDOS data from vaspout.h5 file and convert to compatible format"""
    if not HAS_H5PY:
        print("Warning: h5py is not installed. Cannot read HDF5 files.")
        return None, None
    
    h5file_path = os.path.join(location, 'vaspout.h5')
    if not os.path.isfile(h5file_path):
        return None, None
    
    try:
        print(f"Found vaspout.h5. Reading PDOS data from HDF5 file...")
        
        with h5py.File(h5file_path, 'r') as h5file:
            # Read energies and Fermi level
            energies = h5file['results/electron_dos/energies'][()]
            efermi = h5file['results/electron_dos/efermi'][()]
            
            # Read DOS data
            dos_data = h5file['results/electron_dos/dos'][()]
            # Sum over spin dimension if present
            if dos_data.ndim > 1:
                dos_total = np.sum(dos_data, axis=0)
            else:
                dos_total = dos_data
            
            # Read partial DOS: (nspin, natoms, norb, nE) or (natoms, norb, nE)
            dospar = h5file['results/electron_dos/dospar'][()]
            
            # Handle spin dimension
            if dospar.ndim == 4:
                nspin, natoms, norb, nE = dospar.shape
                # Sum over spin dimension
                dospar = np.sum(dospar, axis=0)  # (natoms, norb, nE)
            else:
                natoms, norb, nE = dospar.shape
            
            # --- Build per-atom element list ---
            pos_grp = h5file['results/positions']
            ion_types_bytes = pos_grp['ion_types'][()]  # one name per *type*
            
            # Get number of ions of each type
            if 'number_ion_types' in pos_grp:
                n_per_type = pos_grp['number_ion_types'][()]
            else:
                # Fallback: assume ion_types is already per atom
                n_per_type = np.ones_like(ion_types_bytes, dtype=int)
            
            # Build per-atom element list
            ion_types = [s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in ion_types_bytes]
            atom_symbols = []
            for sym, n in zip(ion_types, n_per_type):
                atom_symbols.extend([sym] * int(n))
            atom_symbols = np.array(atom_symbols)
            
            # Sanity check
            if natoms != len(atom_symbols):
                print(f"Warning: Mismatch between dospar atoms ({natoms}) and built symbols ({len(atom_symbols)})")
                # Try alternative: assume ion_types is already per-atom
                atom_symbols = np.array([s.decode('utf-8') if isinstance(s, bytes) else str(s) 
                                        for s in ion_types_bytes[:natoms]])
        
        # Shift energies so that EF = 0 eV
        energies = energies - efermi
        
        # Identify unique elements
        unique_elements = sorted(set(atom_symbols))
        
        # Convert to PDOS format: {element: {spin: [[energy, s, py, pz, px, dxy, dyz, dz2, dxz, dx2, tot], ...]}}
        pdos_files = {}
        
        for element in unique_elements:
            # Get indices of atoms of this element
            atom_indices = np.where(atom_symbols == element)[0]
            
            # Sum PDOS over all atoms of this element: (norb, nE)
            pdos_element = np.sum(dospar[atom_indices, :, :], axis=0)
            
            # Extract orbital-resolved PDOS assuming VASP order:
            # 0: s
            # 1-3: px, py, pz
            # 4-8: dxy, dyz, dz2, dxz, dx2-y2
            pdos_s = pdos_element[0, :]
            
            # Individual p orbitals
            if norb >= 4:
                pdos_py = pdos_element[1, :]  # Note: VASP order may be px, py, pz
                pdos_pz = pdos_element[2, :]
                pdos_px = pdos_element[3, :]
            else:
                pdos_py = pdos_pz = pdos_px = np.zeros_like(pdos_s)
            
            # Individual d orbitals
            if norb >= 9:
                pdos_dxy = pdos_element[4, :]
                pdos_dyz = pdos_element[5, :]
                pdos_dz2 = pdos_element[6, :]
                pdos_dxz = pdos_element[7, :]
                pdos_dx2 = pdos_element[8, :]
            else:
                pdos_dxy = pdos_dyz = pdos_dz2 = pdos_dxz = pdos_dx2 = np.zeros_like(pdos_s)
            
            # Calculate total for this element
            pdos_tot = pdos_s + pdos_py + pdos_pz + pdos_px + pdos_dxy + pdos_dyz + pdos_dz2 + pdos_dxz + pdos_dx2
            
            # Create data in format expected by plot_pdos: [[energy, s, py, pz, px, dxy, dyz, dz2, dxz, dx2, tot], ...]
            pdos_data = []
            for i, e in enumerate(energies):
                pdos_data.append([
                    str(e),
                    str(pdos_s[i]),
                    str(pdos_py[i]),
                    str(pdos_pz[i]),
                    str(pdos_px[i]),
                    str(pdos_dxy[i]),
                    str(pdos_dyz[i]),
                    str(pdos_dz2[i]),
                    str(pdos_dxz[i]),
                    str(pdos_dx2[i]),
                    str(pdos_tot[i])
                ])
            
            pdos_files[element] = {'UP': pdos_data}
        
        # Store total DOS for TDOS plotting
        tdos_data = []
        for i, e in enumerate(energies):
            tdos_data.append([str(e), str(dos_total[i])])
        
        print(f"Successfully read PDOS data for elements: {', '.join(unique_elements)}")
        return pdos_files, tdos_data
        
    except Exception as e:
        print(f"Warning: Could not read from vaspout.h5: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def read_tdos_from_hdf5(location):
    """Read total DOS from vaspout.h5 file"""
    if not HAS_H5PY:
        return None, None, None
    
    h5file_path = os.path.join(location, 'vaspout.h5')
    if not os.path.isfile(h5file_path):
        return None, None, None
    
    try:
        with h5py.File(h5file_path, 'r') as h5file:
            energies = h5file['results/electron_dos/energies'][()]
            dos_data = h5file['results/electron_dos/dos'][()]
            efermi = h5file['results/electron_dos/efermi'][()]
        
        # Sum over spin dimension if present
        if dos_data.ndim > 1:
            dos_total = np.sum(dos_data, axis=0)
        else:
            dos_total = dos_data
        
        # Shift energies so that EF = 0 eV
        energies = energies - efermi
        
        # For HDF5, return summed total DOS for both up and down
        return energies, dos_total, dos_total
        
    except Exception as e:
        print(f"Warning: Could not read TDOS from vaspout.h5: {e}")
        return None, None, None

def plot_pdos(pdos_files, plotting_info, title, spin_filter=None, fill=False, location=None, fill_colors=None, cutoff=None, show_grid=False, show_ylabel=False):
    colors = plt.cm.tab10.colors
    linestyles = ['-', '--', '-.', ':']
    element_color_map = {}
    color_index = 0

    # Check if TDOS is requested (standalone 'tot' in plotting_info)
    plot_tdos_flag = 'tot' in plotting_info and not plotting_info['tot']
    tdos_energy, tdos_up, tdos_down = None, None, None
    
    if plot_tdos_flag and location:
        tdos_energy, tdos_up, tdos_down = read_tdos_file(location)
        # Perform bandgap analysis when TDOS is plotted
        if tdos_energy is not None:
            analyze_tdos_bandgap(tdos_energy, tdos_up, tdos_down)

    # Process elements in the order they appear in plotting_info
    for element in plotting_info.keys():
        if element == 'tot' and plot_tdos_flag:
            # Plot TDOS
            if tdos_energy is not None:
                # Use custom color if specified in fill_colors, otherwise default
                color = fill_colors.get('tot', colors[color_index % len(colors)]) if fill_colors else colors[color_index % len(colors)]
                tdos_plotted = False
                
                if not spin_filter or spin_filter.upper() == 'UP':
                    # Apply cutoff filter if specified
                    if cutoff is None or max(tdos_up) >= cutoff:
                        label = 'TDOS' if not tdos_plotted else None
                        if fill:
                            plt.fill_between(tdos_energy, tdos_up, alpha=0.3, color=color)
                            plt.plot(tdos_energy, tdos_up, color=color, linewidth=1.5, label=label)
                        else:
                            plt.plot(tdos_energy, tdos_up, label=label, color=color)
                        tdos_plotted = True
                
                if tdos_down is not None and (not spin_filter or spin_filter.upper() == 'DOWN'):
                    # Apply cutoff filter if specified - use absolute values for DOWN spin
                    if cutoff is None or max(abs(tdos_down)) >= cutoff:
                        label = 'TDOS' if not tdos_plotted else None
                        if fill:
                            plt.fill_between(tdos_energy, tdos_down, alpha=0.3, color=color)
                            plt.plot(tdos_energy, tdos_down, color=color, linewidth=1.5, label=label)
                        else:
                            plt.plot(tdos_energy, tdos_down, label=label, color=color)
                
                color_index += 1
            continue
            
        if element in pdos_files:
            if element not in element_color_map:
                element_color_map[element] = colors[color_index % len(colors)]
                color_index += 1

            spins = pdos_files[element]
            for spin, data in spins.items():
                if spin_filter and spin.upper() != spin_filter.upper():
                    continue

                x = [float(row[0]) for row in data]

                if element in plotting_info:
                    args = plotting_info[element]
                    for arg_index, arg in enumerate(args):
                        if arg == 'p':
                            y = [sum(map(float, row[2:5])) for row in data]
                        elif arg == 'd':
                            y = [sum(map(float, row[5:10])) for row in data]
                        elif arg == 's':
                            y = [float(row[1]) for row in data]
                        elif arg == 'tot':
                            y = [float(row[-1]) for row in data]
                        else:
                            index_map = ['s', 'py', 'pz', 'px', 'dxy', 'dyz', 'dz2', 'dxz', 'dx2', 'tot']
                            idx = index_map.index(arg)
                            y = [float(row[idx]) for row in data]

                        # Apply cutoff filter if specified - use absolute values for consistency
                        if cutoff is not None and max(abs(val) for val in y) < cutoff:
                            continue  # Skip this curve if max absolute value is below cutoff

                        linestyle = linestyles[arg_index % len(linestyles)]
                        if spin_filter:
                            label = f"{element} {arg.lower()}"
                        else:
                            label = f"{element} {arg.lower()}" if spin.upper() == 'UP' else None
                        
                        color = fill_colors.get(element, element_color_map[element]) if fill_colors else element_color_map[element]
                        
                        if fill:
                            plt.fill_between(x, y, alpha=0.3, color=color)
                            plt.plot(x, y, color=color, linestyle=linestyle, linewidth=1.5, label=label)
                        else:
                            plt.plot(x, y, label=label, color=color, linestyle=linestyle)

    plt.axhline(0, color='black', linestyle='-')
    plt.axvline(0, color='black', linestyle='--')
    plt.xlabel("Energy (eV)")
    plt.ylabel("Density of States")
    plt.title(title)
    plt.legend()
    if show_grid:
        plt.grid(True, alpha=0.3)
    if not show_ylabel:
        plt.gca().set_yticks([])
    plt.show()

def parse_color_input(color_str):
    """Parse and validate color input"""
    available_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow']
    
    color_str = color_str.lower().strip()
    
    if color_str in available_colors:
        return color_str
    elif color_str.startswith('#') and len(color_str) == 7:
        try:
            int(color_str[1:], 16)  # Validate hex
            return color_str
        except ValueError:
            return None
    return None

def get_fill_colors(elements, plotting_info):
    """Interactive color selection for elements when --colour flag is used"""
    available_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow']
    fill_colors = {}
    
    print("\n=== Fill Color Selection ===")
    print("Available colors:", ', '.join(available_colors))
    print("You can also use hex colors like #FF5733 or press Enter for default colors")
    print()
    
    for element in elements:
        if element in plotting_info:  # Only ask for elements that will be plotted
            while True:
                color_input = input(f"Choose fill color for {element} (Enter for default): ").strip().lower()
                
                if not color_input:  # User pressed Enter - use default
                    break
                elif color_input in available_colors:
                    fill_colors[element] = color_input
                    print(f"✓ {element} will use {color_input}")
                    break
                elif color_input.startswith('#') and len(color_input) == 7:
                    # Validate hex color
                    try:
                        int(color_input[1:], 16)  # Check if it's valid hex
                        fill_colors[element] = color_input
                        print(f"✓ {element} will use {color_input}")
                        break
                    except ValueError:
                        print("Invalid hex color. Try again or press Enter for default.")
                else:
                    print(f"'{color_input}' is not a valid color. Try again or press Enter for default.")
    
    return fill_colors

def show_help():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                            PDOS & TDOS Plotter Help                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

🚀 PROGRAM USAGE:
   • Run in current directory:  python PDOS.py
   • Specify directory:         python PDOS.py /path/to/directory
   
📁 SETUP:
   • Place this script in a directory containing VASP output files
   • Recommended: vaspout.h5 file (direct HDF5 output from VASP)
   • Alternative: INCAR, DOSCAR, PROCAR files (for auto-generation via vaspkit)
   • PDOS_*.dat and TDOS.dat files (auto-generated if HDF5 unavailable)

📝 BASIC USAGE:
   Format: Element1 orbital1 orbital2, Element2 orbital3 orbital4
   Example: Ti s p d, O p tot

🔬 AVAILABLE ORBITALS:
   • Individual: s, py, pz, px, dxy, dyz, dz2, dxz, dx2
   • Combined:  p (py+pz+px), d (all d orbitals), tot (total)

📋 ALL AVAILABLE ARGUMENTS:

🌐 GLOBAL PLOTTING OPTIONS:
   
   all / all [orbitals]:
   • Plots all elements with specified orbitals (default: s, p, d)
   • Example: all
             all p d
             all s --fill -c vesta
             all p UP --colour jmol
   
   tot:
   • Plots Total Density of States (TDOS) when used alone
   • Example: tot
             tot red fill
             all p, tot --colour user

🧲 SPIN FILTERING:
   Add spin filter anywhere in the command:
   • UP / --UP     : Plot only spin-up component
   • DOWN / --DOWN : Plot only spin-down component  
   • DW / --DW     : Same as DOWN
   
   Examples: UP Ti s p d, O p tot
            Ti s p d, O p tot --DOWN
            all p --UP --cutoff 0.1

🎨 COLOR SCHEMES & CUSTOMIZATION:
   
   Built-in Color Schemes:
   • --colour vesta / -c vesta    : VESTA 3 crystal visualization colors
   • --colour jmol / -c jmol      : Jmol molecular visualization colors
   • --colour user / -c user      : Your customizable color scheme
   • --colour custom / -c custom  : Same as user scheme
   • --colour cpk / -c cpk        : Classic CPK chemistry colors
   • --colour mp / -c mp          : Materials Project database colors
   • --colour materials / -c materials : Same as Materials Project
   
   Interactive Color Selection:
   • --colour / colour            : Start interactive color picker
   • --color / color / -c         : Same as above
   
   Inline Colors (per element):
   • Add color after orbitals: Ti s p d red, O p tot blue
   • Available colors: red, blue, green, orange, purple, brown, pink, 
     gray, olive, cyan, magenta, yellow
   • Hex colors: Ti s p d #FF5733, O p tot #33C3FF
   
   Color Priority (highest to lowest):
   1. Inline colors (Ti s p d red)
   2. Interactive selection (--colour)  
   3. Color schemes (--colour vesta)
   4. Default matplotlib colors

🌊 FILL & VISUAL OPTIONS:
   
   Fill Options:
   • fill / --fill         : Fill area under curves with transparency
   • gridfill / --gridfill : Same as fill
   
   Grid Options:
   • --grid / grid         : Show grid lines on plot (disabled by default)
   
   Examples: Ti s p d, O p tot fill
            all p d fill --colour vesta --grid
            tot --colour user fill

📊 DATA FILTERING:
   
   Cutoff Filtering:
   • --cutoff X / cutoff=X : Hide curves with max absolute value below X
   • Useful for removing negligible contributions
   • Formats: --cutoff 0.1  OR  cutoff=0.1
   
   Examples: Ti s p d, O p tot --cutoff 0.05
            all p cutoff=0.1 fill --colour jmol
            tot --cutoff 0.02 --UP

🎯 COMPLETE ARGUMENT COMBINATIONS:

   Basic PDOS:           
   Ti s p d, O p tot

   With specific colors:          
   Ti s p d blue, O p tot red fill  

   Using color schemes:
   all --colour vesta fill --grid
   all p d --colour jmol UP --cutoff 0.1

   Spin-filtered with options:        
   Ti s p d, O p tot UP fill --colour user --grid

   Interactive with all options:   
   all fill --colour --cutoff 0.05 DOWN --grid

   TDOS plotting:           
   tot
   tot green --UP fill --grid
   all p, tot --colour cpk fill --cutoff 0.02

   Mixed comprehensive example:      
   Ti d red, O p blue, tot --colour vesta fill UP --grid --cutoff 0.03

🔧 ARGUMENT SYNTAX RULES:

   • Arguments can appear in any order
   • Multiple formats accepted: --UP, UP, --colour, colour, -c
   • Commas separate different elements: Ti s p d, O p tot
   • Spaces separate orbitals within elements: Ti s p d
   • Colors come after orbitals: Ti s p d red
   • Global options apply to entire plot: all p d fill --colour vesta

⚙️ SPECIAL COMMANDS:
   • 'help' : Show this help message
   • Ctrl+D : Exit the program

📖 COLOR CUSTOMIZATION GUIDE:
   To customize the 'user' color scheme:
   1. Edit: DOS-plots/PDOS/ElementColorSchemes.yaml
   2. Find the 'User:' section
   3. Modify RGB values: Element: [Red, Green, Blue] (0-255)
   4. Save and use with: --colour user

💡 PRO TIPS:
   • Use --grid for better data reading
   • Combine cutoff with all to focus on significant contributions
   • Save time with color schemes instead of manual color selection
   • Test different spin filters (UP/DOWN) for magnetic materials
   • Use fill for publication-quality plots
   • Inline colors override scheme colors for specific elements
   • Order matters for inline colors: specify element, orbitals, then color
   • Use 'all p d' to plot only p and d orbitals for all elements

""")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='PDOS and TDOS plotter for VASP calculations')
    parser.add_argument('directory', nargs='?', default=os.getcwd(),
                        help='Directory path where PDOS files exist (default: current directory)')
    args = parser.parse_args()
    
    location = os.path.abspath(args.directory)
    
    # Validate directory
    if not os.path.isdir(location):
        print(f"Error: '{location}' is not a valid directory.")
        sys.exit(1)
    
    # Try to read PDOS files, with HDF5 as first priority
    pdos_files = None
    tdos_data = None
    
    # First, try to read from vaspout.h5
    if HAS_H5PY:
        pdos_files, tdos_data = read_pdos_from_hdf5(location)
    
    # Fall back to .dat files if HDF5 reading failed
    if not pdos_files:
        print("No vaspout.h5 found or HDF5 reading failed. Trying to read PDOS .dat files...")
        pdos_files = read_pdos_files(location)
        if not pdos_files:
            generated = try_generate_pdos_dat_files(location)
            if generated:
                pdos_files = read_pdos_files(location)
    
    if not pdos_files:
        print(f"Error: No PDOS files found in '{location}' and could not generate them.")
        if HAS_H5PY:
            print("Required: vaspout.h5 file OR (INCAR, DOSCAR, PROCAR files)")
        else:
            print("Required files: INCAR, DOSCAR, PROCAR")
        sys.exit(1)
    
    last_dirs = os.path.normpath(location).split(os.sep)[-4:]
    title = os.path.join(*last_dirs)
    
    print(f"Working in directory: {location}")
    print("PDOS files found for elements:", ", ".join(pdos_files.keys()))
    
    while True:
        plotting_input = input("Enter plotting info (or 'help' for help, Ctrl+D to exit): ").strip()

        if plotting_input.lower() == 'help':
            show_help()
            continue

        try:
            plotting_info = {}
            spin_filter = None
            fill = False
            use_interactive_colors = False
            fill_colors = {}
            cutoff = None
            show_grid = False
            tokens = plotting_input.split()

            # Extract spin filter, fill option, --colour flag, and cutoff
            filtered_tokens = []
            use_all_elements = False
            all_orbitals = ['s', 'p', 'd']
            color_scheme = None

            i = 0
            while i < len(tokens):
                token = tokens[i]
                token_upper = token.upper()
                token_lower = token.lower()
                
                if token_upper in ['UP', '--UP']:
                    spin_filter = 'UP'
                    i += 1
                elif token_upper in ['DOWN', '--DOWN', 'DW', '--DW']:
                    spin_filter = 'DOWN'
                    i += 1
                elif token_lower in ['fill', '--fill', 'gridfill', '--gridfill']:
                    fill = True
                    i += 1
                elif token_lower in ['--grid', 'grid']:
                    show_grid = True
                    i += 1
                elif token_lower in ['--colour', '--color', 'colour', 'color', '-c']:
                    # Check if next token is a color scheme name
                    if i + 1 < len(tokens):
                        next_token = tokens[i + 1]
                        if next_token.lower() in get_available_schemes():
                            # It's a scheme name
                            color_scheme = next_token.lower()
                            i += 2  # Skip both --colour and scheme name
                        else:
                            # No scheme name provided, use interactive
                            use_interactive_colors = True
                            i += 1
                    else:
                        # No next token, use interactive
                        use_interactive_colors = True
                        i += 1
                elif token.startswith('--cutoff'):
                    try:
                        if '=' in token:
                            cutoff = float(token.split('=')[1])
                        elif i + 1 < len(tokens):
                            cutoff = float(tokens[i + 1])
                            i += 1  # Skip the value token
                        i += 1
                    except (IndexError, ValueError):
                        print("Invalid cutoff value. Ignoring.")
                        i += 1
                elif token.startswith('cutoff='):
                    try:
                        cutoff = float(token.split('=')[1])
                    except ValueError:
                        print("Invalid cutoff value. Ignoring.")
                    i += 1
                else:
                    filtered_tokens.append(token)
                    i += 1

            # Process 'all' keyword
            if filtered_tokens and filtered_tokens[0].lower() == 'all':
                use_all_elements = True
                filtered_tokens = filtered_tokens[1:]
                
                # Check if specific orbitals are provided after 'all'
                temp_orbitals = []
                for t in filtered_tokens:
                    if t.lower() in ['s', 'p', 'd', 'py', 'pz', 'px', 'dxy', 'dyz', 'dz2', 'dxz', 'dx2', 'tot']:
                        temp_orbitals.append(t.lower())
                    elif t == ',':
                        break
                
                if temp_orbitals:
                    all_orbitals = temp_orbitals
                    filtered_tokens = [t for t in filtered_tokens if t not in temp_orbitals]

            # Parse elements and orbitals
            if use_all_elements:
                for element in pdos_files.keys():
                    plotting_info[element] = all_orbitals
            else:
                segments = ' '.join(filtered_tokens).split(',')
                for segment in segments:
                    parts = segment.strip().split()
                    if not parts:
                        continue
                    
                    if parts[0].lower() == 'tot' and len(parts) == 1:
                        plotting_info['tot'] = []
                        continue
                    
                    element = parts[0]
                    orbitals = []
                    inline_color = None
                    
                    for part in parts[1:]:
                        if part.lower() in ['s', 'p', 'd', 'py', 'pz', 'px', 'dxy', 'dyz', 'dz2', 'dxz', 'dx2', 'tot']:
                            orbitals.append(part.lower())
                        else:
                            parsed_color = parse_color_input(part)
                            if parsed_color:
                                inline_color = parsed_color
                    
                    if element in pdos_files or element.lower() == 'tot':
                        plotting_info[element] = orbitals if orbitals else ['s', 'p', 'd']
                        if inline_color:
                            fill_colors[element] = inline_color

            # Apply color scheme if specified
            if color_scheme and not use_interactive_colors:
                scheme_colors = apply_color_scheme(list(pdos_files.keys()), plotting_info, color_scheme)
                for elem, color in scheme_colors.items():
                    if elem not in fill_colors:  # Don't override inline colors
                        fill_colors[elem] = color

            # Interactive color selection
            if use_interactive_colors and not color_scheme:
                fill_colors.update(get_fill_colors(list(pdos_files.keys()), plotting_info))

            if plotting_info:
                plot_pdos(pdos_files, plotting_info, title, spin_filter, fill, location, fill_colors, cutoff, show_grid)
            else:
                print("No valid plotting info provided.")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except EOFError:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()