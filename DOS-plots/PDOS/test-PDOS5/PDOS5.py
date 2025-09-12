# Working as intended on Mon Jul 28 02:52:37 AM EDT 2025
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


def plot_pdos(pdos_files, plotting_info, title, spin_filter=None):
    colors = plt.cm.tab10.colors
    linestyles = ['-', '--', '-.', ':']
    element_color_map = {}
    color_index = 0

    # Process elements in the order they appear in plotting_info
    for element in plotting_info.keys():
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
                        label = f"{element} {arg.lower()}" if spin.upper() == 'UP' else None
                        plt.plot(x, y, label=label, color=element_color_map[element], linestyle=linestyle)

    plt.axhline(0, color='black', linestyle='-')
    plt.axvline(0, color='black', linestyle='--')
    plt.xlabel("Energy (eV)")
    plt.ylabel("Density of States")
    plt.title(title)
    plt.legend()
    plt.show()

def show_help():
    print("""
How to use this program:
1. You will be prompted to enter a directory containing PDOS_*.dat files.
2. After the files are loaded, you'll be asked to enter the plotting info.
3. Format for plotting info:
   ElementName orbital1 orbital2, ElementName orbital3 orbital4, ...
   Example: Ti s p d, O p tot
4. Orbitals you can use:
   s, p, d, py, pz, px, dxy, dyz, dz2, dxz, dx2, tot
5. You can optionally add 'UP' or 'DOWN' at the beginning or end of the line to restrict plotting to a spin:
   Example: UP Ti s p d, O p tot
            or Ti s p d, O p tot DOWN
6. Enter '0' to change directory or 'help' to see this message again.
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
                tokens = plotting_input.split()

                for info in plotting_input.split(','):
                    parts = info.strip().split()
                    if parts:
                        plotting_info[parts[0]] = parts[1:]

                plot_pdos(pdos_files, plotting_info, title, spin_filter)

            except Exception as e:
                print(f"Error: {e}")
                continue

if __name__ == "__main__":
    main()
