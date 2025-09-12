# Working as intended on Sun Aug 17 28 02:52:37 AM EDT 2025
# The program expects vaspkit installed in the system.
# This doesn't sort the elements in the plot legend. 

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
for pkg in ["matplotlib", "glob", "readline", "subprocess"]:
    check_dependency(pkg)

#-------- Imports -----------
import os
import glob
import readline
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm

# Add this near the top after imports
plt.rcParams['font.family'] = ['Noto Sans Devanagari', 'DejaVu Sans']

# Function to enable tab-completion for file paths (tested only in linux)
def input_with_completion(prompt):
    def complete(text, state):
        options = [path for path in glob.glob(text + '*')]
        if state < len(options):
            return options[state] + ' '
        else:
            return None

    readline.set_completer_delims('\t')
    readline.parse_and_bind("tab: complete")

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

def plot_pdos(pdos_files, plotting_info, title, spin_filter=None, fill=False, location=None, fill_colors=None):
    colors = plt.cm.tab10.colors
    linestyles = ['-', '--', '-.', ':']
    element_color_map = {}
    color_index = 0

    # Check if TDOS is requested (standalone 'tot' in plotting_info)
    plot_tdos_flag = 'tot' in plotting_info and not plotting_info['tot']
    tdos_energy, tdos_up, tdos_down = None, None, None
    
    if plot_tdos_flag and location:
        tdos_energy, tdos_up, tdos_down = read_tdos_file(location)

    # Process elements in the order they appear in plotting_info
    for element in plotting_info.keys():
        if element == 'tot' and plot_tdos_flag:
            # Plot TDOS
            if tdos_energy is not None:
                # Use custom color if specified in fill_colors, otherwise default
                color = fill_colors.get('tot', colors[color_index % len(colors)]) if fill_colors else colors[color_index % len(colors)]
                tdos_plotted = False
                
                if not spin_filter or spin_filter.upper() == 'UP':
                    label = 'TDOS' if not tdos_plotted else None
                    if fill:
                        plt.fill_between(tdos_energy, tdos_up, alpha=0.3, color=color)
                        plt.plot(tdos_energy, tdos_up, color=color, linewidth=1.5, label=label)
                    else:
                        plt.plot(tdos_energy, tdos_up, label=label, color=color)
                    tdos_plotted = True
                
                if tdos_down is not None and (not spin_filter or spin_filter.upper() == 'DOWN'):
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
    plt.grid(True, alpha=0.3)
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

📝 BASIC USAGE:
   Format: Element1 orbital1 orbital2, Element2 orbital3 orbital4
   Example: Ti s p d, O p tot
   Total Density of States (TDOS) can be plotted with 'tot' alone.
   Example: Ti s p d, O p tot, tot

🔬 AVAILABLE ORBITALS:
   • Individual: s, py, pz, px, dxy, dyz, dz2, dxz, dx2
   • Combined:  p (py+pz+px), d (all d orbitals), tot (total)

🧲 SPIN FILTERING:
   Add spin filter anywhere in the command:
   • UP / --UP     : Plot only spin-up component
   • DOWN / --DOWN : Plot only spin-down component  
   • DW / --DW     : Same as DOWN
   
   Examples: UP Ti s p d, O p tot
            Ti s p d, O p tot --DOWN

🎨 COLOR CUSTOMIZATION:
   
   Method 1 - Inline Colors:
   • Add color after orbitals: Ti s p d red, O p tot blue
   • Available colors: red, blue, green, orange, purple, brown, pink, 
     gray, olive, cyan, magenta, yellow
   • Hex colors: Ti s p d #FF5733, O p tot #33C3FF
   
   Method 2 - Interactive Selection:
   • Add --colour or colour flag: Ti s p d, O p tot --colour
   • Program will prompt for color selection interactively

🌊 FILL OPTIONS:
   Add fill option anywhere in the command:
   • fill / --fill         : Fill area under curves
   • gridfill / --gridfill : Same as fill
   
   Examples: Ti s p d, O p tot fill
            fill --colour Ti s p d, O p tot

📊 TDOS PLOTTING:
   • Use 'tot' alone to plot Total Density of States
   • Examples: tot
              tot red fill
              tot --colour --UP

💡 COMPLETE EXAMPLES:
   
   Basic PDOS:           Ti s p d, O p tot
   With colors:          Ti s p d blue, O p tot red fill  
   Spin-filtered:        Ti s p d, O p tot UP fill
   Interactive colors:   Ti s p d, O p tot fill --colour
   TDOS only:           tot
   TDOS with options:   tot green --UP fill
   Mixed plotting:      Ti d, O p, tot --colour fill

⚙️ SPECIAL COMMANDS:
   • '0'    : Change directory
   • 'help' : Show this help message

📋 NOTES:
   • Files are auto-generated using vaspkit if missing
   • Colors work with or without fill option
   • Multiple elements can have different colors
   • Order matters: specify element, orbitals, then color

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
                if generated:
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
                tokens = plotting_input.split()

                # Extract spin filter, fill option, and --colour flag
                filtered_tokens = []
                for token in tokens:
                    if token.lower() in ['up', 'down', 'dw', '--up', '--down', '--dw']:
                        spin_filter = 'DOWN' if token.lower() in ['dw', '--dw'] else token.upper().replace('--', '')
                    elif token.lower() in ['fill', 'gridfill', '--fill', '--gridfill']:
                        fill = True
                    elif token.lower() in ['colour', '--colour', 'color', '--color']:
                        use_interactive_colors = True
                    else:
                        filtered_tokens.append(token)

                plotting_input = ' '.join(filtered_tokens)

                # Parse plotting info and inline colors
                for info in plotting_input.split(','):
                    parts = info.strip().split()
                    if parts:
                        if len(parts) == 1 and parts[0] == 'tot':
                            # Standalone 'tot' means plot TDOS
                            plotting_info['tot'] = []
                        else:
                            element = parts[0]
                            orbitals = []
                            
                            # Check if last part is a color
                            color = None
                            if len(parts) > 1:
                                potential_color = parse_color_input(parts[-1])
                                if potential_color:
                                    color = potential_color
                                    orbitals = parts[1:-1]  # Exclude color from orbitals
                                else:
                                    orbitals = parts[1:]    # No color, all are orbitals
                            
                            plotting_info[element] = orbitals
                            if color:
                                fill_colors[element] = color

                # Check if TDOS is requested and file exists
                plot_tdos_requested = 'tot' in plotting_info and not plotting_info['tot']
                if plot_tdos_requested:
                    tdos_path = os.path.join(location, 'TDOS.dat')
                    if not os.path.exists(tdos_path):
                        print("TDOS.dat not found. Attempting to generate it...")
                        if try_generate_tdos_dat_file(location):
                            print("TDOS.dat generated successfully.")
                        else:
                            print("Failed to generate TDOS.dat. TDOS plotting may not work.")

                # Interactive color selection if --colour flag is used
                if use_interactive_colors:
                    interactive_colors = get_fill_colors(plotting_info.keys(), plotting_info)
                    # Merge with inline colors, giving priority to interactive selection
                    fill_colors.update(interactive_colors)

                # Pass fill_colors to plot_pdos
                plot_pdos(pdos_files, plotting_info, title, spin_filter, fill, location, fill_colors)

            except Exception as e:
                print(f"Error: {e}")
                continue

if __name__ == "__main__":
    main()
