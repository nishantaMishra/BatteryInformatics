"""Working as intended on Wednesday 20 December 2023 10:07:55 AM IST

Program plots total density of states by reading TDOS file obtained from vaspkit.
TDOS4.py is a more advanced version that includes bandgap detection and analysis of magnetic properties.
"""

import os # pip install os
import numpy as np # pip install numpy
import matplotlib.pyplot as plt # pip install matplotlib

def read_tdos_file(file_path):
    try:
        # Read data from TDOS.dat file
        data = np.loadtxt(file_path)
        return data
    except Exception as e:
        print(f"Error reading TDOS file: {e}")
        return None

def plot_tdos(data, directory_name):
    if data is not None:
        # Split the data into columns
        energy = data[:, 0]
        tdos_up = data[:, 1]
        tdos_down = data[:, 2]

        # Create a plot
        plt.figure()
        plt.plot(energy, tdos_up, label='TDOS-UP')
        plt.plot(energy, tdos_down, label='TDOS-DOWN')
        plt.xlabel('Energy')
        plt.ylabel('DOS')
        plt.legend()
        plt.grid(True)
        plt.title(f'Total Density of States {os.path.basename(directory_name)}')
        plt.show()

        # Optionally, save the plot to a file
        # plt.savefig('_plot.png')

# Get user-defined directory path
user_directory = input("Enter the path to the directory containing TDOS.dat file: ")

# Formulate the path to TDOS.dat file
tdos_file_path = os.path.join(user_directory, 'TDOS.dat')

# Read and plot the TDOS data
tdos_data = read_tdos_file(tdos_file_path)
plot_tdos(tdos_data, user_directory)
