# Working as expected on Wednesday 13 March 2024 03:55:02 PM IST
"""Is capable of taking multiple arguments for every element.
User can Enter symbol of any element followed by arguments. User can enter 's', 'p', 'd', 'py', 'pz', 'px', 'dxy', 'dyz', 'dz2', 'dxz', 'dx2', and 'tot'
>>> Element name orbital orbital orbital, Element orbital orbital orbital
However this program cannot care about f orbitals. 
"""

import os
import matplotlib.pyplot as plt

def read_pdos_files(location):
    pdos_files = {}
    for file_name in os.listdir(location):
        if file_name.startswith("PDOS_") and file_name.endswith(".dat"):
            element = file_name.split("_")[1]
            spin = file_name.split("_")[2].split(".")[0]
            if element not in pdos_files:
                pdos_files[element] = {}
            with open(os.path.join(location, file_name), 'r') as file:
                next(file)  # Skip the header line
                pdos_files[element][spin] = [line.strip().split() for line in file.readlines() if line.strip()]
    return pdos_files

def plot_pdos(pdos_files, plotting_info):
    colors = plt.cm.tab10.colors  # Generating 10 colors
    linestyles = ['-', '--', '-.', ':']  # Define different linestyles
    linestyle_index = 0
    element_color_map = {}  # Map element names to colors
    for element, spins in pdos_files.items():
        if element not in element_color_map:
            element_color_map[element] = colors[len(element_color_map) % len(colors)]
        for spin, data in spins.items():
            x = [float(row[0]) for row in data]
            if element in plotting_info:
                args = plotting_info[element]
                for arg_index, arg in enumerate(args):
                    y = [sum(map(float, row[1:])) for row in data]
                    if arg in ['s', 'p', 'd', 'py', 'pz', 'px', 'dxy', 'dyz', 'dz2', 'dxz', 'dx2', 'tot']:
                        if arg == 'p':
                            y = [sum(map(float, row[2:5])) for row in data]
                        elif arg == 'd':
                            y = [sum(map(float, row[5:10])) for row in data]
                        elif arg == 's':
                            y = [float(row[1]) for row in data]
                        elif arg == 'tot':
                            y = [float(row[-1]) for row in data]
                        else:
                            index = ['s', 'py', 'pz', 'px', 'dxy', 'dyz', 'dz2', 'dxz', 'dx2', 'tot'].index(arg)
                            y = [float(row[index]) for row in data]
                        linestyle = linestyles[arg_index % len(linestyles)]  # Cycle through linestyles
                        if spin == 'UP':
                            plt.plot(x, y, label=f"{element} {arg.lower()}", color=element_color_map[element], linestyle=linestyle)
                        else:
                            plt.plot(x, y, color=element_color_map[element], linestyle=linestyle)  # Plot DW without label
    plt.axhline(0, color='black', linestyle='-')  # Solid line at y = 0
    plt.axvline(0, color='black', linestyle='--') # Dashed line at x = 0

def main():
    location = input("Enter the location where PDOS files exist: ")
    last_three_dirs = os.path.normpath(location).split(os.sep)[-4:]
    title = os.path.join(*last_three_dirs)
    pdos_files = read_pdos_files(location)
    print("Got PDOS files for the following elements:")
    print(", ".join(pdos_files.keys()))
    plotting_input = input("Enter the plotting info for each element (element name followed by arguments separated by space, multiple arguments separated by comma): ")
    plotting_info = {}
    for info in plotting_input.split(','):
        element_args = info.strip().split()
        element = element_args[0]
        args = element_args[1:]
        plotting_info[element] = args
    plot_pdos(pdos_files, plotting_info)
    plt.xlabel("Energy")
    plt.ylabel("Density of States")
    plt.title(title)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

