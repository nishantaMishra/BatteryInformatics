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
for pkg in ["matplotlib", "numpy", "pyyaml"]:
    check_dependency(pkg)

#-------- Imports -----------
import os
import glob
import readline
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm

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
    """Read TDOS.dat file similar to TDOS5.py"""
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

📁 SETUP:
   • Place this script in a directory containing VASP output files
   • Required files: INCAR, DOSCAR, PROCAR (for auto-generation)
   • PDOS_*.dat and TDOS.dat files (will be auto-generated if missing)

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
   • '0'    : Change directory
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
    while True:
        while True:
            location = input_with_completion("Enter the directory path where PDOS files exist: ").strip()

            if not os.path.isdir(location):
                print("Not a directory. Try again.")
                continue

            pdos_files = read_pdos_files(location)
            if not pdos_files:
                generated = try_generate_pdos_dat_files(location)
                if (generated):
                    pdos_files = read_pdos_files(location)

            if not pdos_files:
                print("Still no PDOS files. Please try a different directory.")
                continue

            break

        last_dirs = os.path.normpath(location).split(os.sep)[-4:]
        title = os.path.join(*last_dirs)

        print("PDOS files found for elements:", ", ".join(pdos_files.keys()))

        while True:
            plotting_input = input("Enter plotting info (or '0' to change directory, 'help' for help): ").strip()

            if plotting_input == '0':
                break
            elif plotting_input.lower() == 'help':
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
                all_orbitals = ['s', 'p', 'd'
