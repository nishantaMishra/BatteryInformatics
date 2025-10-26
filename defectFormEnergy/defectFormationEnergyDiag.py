import sys
import numpy as np
import re
import matplotlib
import matplotlib.pyplot as plt

class calculate_E_formation:
    """ calculate formation energy of a given defect with charge transition levels and plot it.
    expected to have input file such as Hf_vac_N_poor.in
    The output file, which will be named as, e.g. Hf_vac_N_poor.in_out can be used as input for the
    plot_multiple_formation_E class in this module"""

    def __init__(self, filein):
        """initialize an instance"""

        self.filein = filein
        self.fileout = filein + "_out"
        self.fileout_CTL = filein + "_CTL"

        self.VBM = 0.0
        self.band_gap = 0.0
        self.host_supercell_energy = 0.0
        self.host_type = 0  #1 - mono-atomic, 2-binary, 3-ternary crystal, etc.
        self.host_atom = []
        self.vacancies = []
        self.impurities = []
        self.charge_start = 0
        self.charge_end = 0
        self.chem_potentials = {} #dictionary type
        self.supercell_energy_list = []
        self.correction_terms_list = []

        # Read the input data
        self.read_data()
        self.n_charges = int(self.charge_end - self.charge_start + 1)
        self.supercell_energy = np.array(self.supercell_energy_list).reshape(self.n_charges, 2)
        self.correction_terms = np.array(self.correction_terms_list).reshape(self.n_charges, 2)

        # Write the formation energy for each charge state
        self.energy_step = 0.01 # eV
        self.energy_part = 0.0  # eV
        self.charge_dependent_part = 0.0 # eV
        self.fermi_level = []   #x-axis Fermi-level
        for i in range(0,int(self.band_gap/self.energy_step)):
            self.fermi_level.append(i * self.energy_step)
        self.all_data = np.array([])  # will stack actual the data array in self.write_data().
        self.formation_energy_data = [] #the lowest energy value for a given fermi level

        self.write_data()

        self.CTL_list = []   #charge transition levels
        self.CTL_list = self.find_CTL() #read the self.formation_energy_data point-by-point, see where the slope changes.
        self.CTL_count = int(len(self.CTL_list)/2)
        self.CTL_array = np.array(self.CTL_list).reshape(self.CTL_count,2)

        # Plot the formation energy
        self.plot_data()

    def read_data(self):
        with open(self.filein, 'r') as data_file:
            for line in data_file:
                if "&VBM" in line:
                    tmp_line =  data_file.readline()
                    self.VBM = float(tmp_line)
                elif "&band_gap" in line:
                    tmp_line = data_file.readline()
                    self.band_gap = float(tmp_line)
                elif "&Host_type" in line:
                    tmp_line = data_file.readline()
                    re_findall = re.findall(r"[\w]+", tmp_line)
                    self.host_type = len(re_findall)
                    for i in range(0, len(re_findall)):
                        self.host_atom.append(re_findall[i])
                elif  "&Vacancies" in line:
                    tmp_line = data_file.readline()
                    re_findall = re.findall(r"[\w]+", tmp_line)
                    for i in range(0, len(re_findall)):
                        self.vacancies.append(re_findall[i])
                elif "&Impurities" in line:
                    tmp_line = data_file.readline()
                    re_findall = re.findall(r"[\w]+", tmp_line)
                    for i in range(0, len(re_findall)):
                        self.impurities.append(re_findall[i])
                elif "&Host_supercell_energy" in line:
                    tmp_line = data_file.readline()
                    self.host_supercell_energy = float(tmp_line)
                elif "&Charge_state_range" in line:
                    tmp_line = data_file.readline()
                    self.charge_start = int(tmp_line.split()[0])
                    self.charge_end = int(tmp_line.split()[1])
                elif "&Chemical_potentials" in line:
                    chk = True
                    while chk == True:
                        tmp_line = data_file.readline()
                        tmp_line_split = tmp_line.split()
                        if len(tmp_line_split) > 1:
                            self.chem_potentials[tmp_line_split[0]] = float(tmp_line_split[1])
                        elif len(tmp_line_split) == 0:  # hit the end of the chem_potential data.
                            chk = False
                        elif "&" in tmp_line:
                            print( "Warning: there should be a space between every section. \n")
                            chk = False
                        else:
                            continue
                elif "&Defective_supercell_energy" in line:
                    for i in range(self.charge_start, self.charge_end+1):
                        tmp_line = data_file.readline()
                        if i == int(tmp_line.split()[0]):
                            self.supercell_energy_list.append(int(tmp_line.split()[0])) #charge state
                            self.supercell_energy_list.append(float(tmp_line.split()[1])) #supercell E
                        else:
                            sys.exit("check the charge state range and defective supercell \
                                energy\n")
                elif "&Correction_terms" in line:
                    for i in range(self.charge_start, self.charge_end+1):
                        tmp_line = data_file.readline()
                        self.correction_terms_list.append(int(i))   #charge state
                        self.correction_terms_list.append(float(tmp_line.split()[0])) #short-range
                else:
                    continue

    def write_data(self):
        Ry = 13.605692  #eV
        for i in range(0, self.n_charges):
            q = self.charge_start + i   # e.g. -2, -1, 0, 1, 2
            print ("\n" + "charge_state =" + str(q) + "\n")

            if q == self.correction_terms[i,0]: # Should contain the charge state, q.
                E_correction = self.correction_terms[i,1]
                print ("E_correction = " + str(E_correction) + "\n")
            else:
                sys.exit("Check the self.correction_terms np array. The first column should contain\
                    the charge state.\n")

            if q == self.supercell_energy[i,0]:
                Energy_part = self.supercell_energy[i,1] *Ry +E_correction -1*self.host_supercell_energy*Ry
                print ("Energy_part = " + str(Energy_part) + "\n")
            else:
                sys.exit("Check the self.supercell_energy np array.\n")

            formation_energy =  Energy_part + q* self.VBM
            #add the chemical potential parts.

            if self.vacancies:
                for c in self.vacancies:
                    formation_energy = formation_energy - (-1)* self.chem_potentials[c]
                    print ("chem_potential for " + str(c) + " (vacancy)  has been added. \n")
                    print ("Energy_part + q* VBM + chem_pot (so far) = " + str(formation_energy) + "\n")
            else:
                print('no vacancies are present\n')
            if self.impurities:
                for c in self.impurities:
                    formation_energy = formation_energy - (+1)* self.chem_potentials[c]
                    print ("chem_potential for " + str(c) + " (impurity)  has been added. \n")
                    print ("Energy_part + q* VBM + chem_pot (so far) = " + str(formation_energy) + "\n")
            else:
                print('no foreign impurities are present \n')

            data_list =[]

            for j in range(0,int(self.band_gap/self.energy_step)):
                data_list.append(formation_energy + q* self.fermi_level[j])

            if len(self.all_data) == 0:
                self.all_data = np.append(self.all_data, np.array(data_list))
            else:
                self.all_data = np.vstack((self.all_data, np.array(data_list)))

        for i in range(0,int(self.band_gap/self.energy_step)):
            tmp_array = self.all_data[:,i]
            minimum_E = np.sort(tmp_array)[0]
            self.formation_energy_data.append(minimum_E)

        with open(self.fileout, 'w') as fout:
            for i in range(0,int(self.band_gap/self.energy_step)):
                fout.write('{0:<15.7f} {1:<15.7f} \n'.format(self.fermi_level[i],\
                    self.formation_energy_data[i]))

    def find_CTL(self):
        #find the charge transition levels where the slope changes
        CTL = []
        slope_to_compare = self.charge_end
        slope_threshold = 0.5
        delta_x = self.energy_step

        fout_str = str(self.filein) + "_CTL"
        with open(fout_str, 'w') as fout:
            fout.write("charge transition levels for " + str(self.filein) + "\n")

        for i in range(1, int(self.band_gap/self.energy_step)):
            delta_y = self.formation_energy_data[i]-self.formation_energy_data[i-1]
            slope = delta_y/delta_x
            #print("slope = " + str(slope) + "\n")
            slope_diff = slope_to_compare - slope
            if abs(slope_diff) < slope_threshold:
                continue
            else:
                print("CTL from" + str(slope_to_compare) + " to " + str (slope_to_compare - 1) + " = " + \
                    str((self.fermi_level[i-1] + self.fermi_level[i])/2.0) + "\n")
                CTL_point = self.fermi_level[i-1] + delta_x/2.0
                CTL.append(CTL_point)
                CTL_energy = self.formation_energy_data[i-1]+slope*delta_x/2.0
                CTL.append(CTL_energy)
                with open(fout_str, 'a') as fout:
                    fout.write('{0:<3d} to  {1:<3d} = {2:<10.5f} \n'.format(slope_to_compare,\
                        slope_to_compare-1, CTL_point))
                slope_to_compare = slope_to_compare - 1
        return CTL

    def plot_data(self):
        x_data = self.fermi_level
        x_min = self.fermi_level[0]
        x_max = self.fermi_level[int(self.band_gap/self.energy_step)-1]+self.energy_step

        fig = plt.figure(figsize=(8,8))
        axes = fig.add_subplot(1,1,1)
        axes.tick_params(axis='both', which='major', labelsize=15)
        axes.tick_params(axis='both', which='minor', labelsize=12)
        plt.xlim (x_min, x_max)
        #plt.ylim(0,3)

        #plot all data
        for i in range(0, self.n_charges):
            q = self.charge_start + i   # e.g. -2, -1, 0, 1, 2
            y_data = self.all_data[i,:]
            label_str = fr'$Cu_{{Zn}}$ with q={q}'
            axes.plot(x_data, y_data, '-', label=label_str)

        #plot the lowest-energy data on top
        y_data = self.formation_energy_data
        axes.plot(x_data,y_data, color = "k"  , linewidth = 3, linestyle = '-')

        axes.legend(loc=0)

        axes.set_xlabel('Fermi level (eV)', fontsize = 18)
        axes.set_ylabel('Defect formation energy (eV)', fontsize = 18)
        axes.set_title(self.filein, fontsize = 18)

        x_CTL_data = self.CTL_array[:,0]
        y_CTL_data = self.CTL_array[:,1]
        axes.plot(x_CTL_data, y_CTL_data, color = 'b', marker = '.', markersize = 4 , mew =  4)

        #fig.savefig(str(self.fileout)+'.png') # commented this auto save due to error
        plt.legend(prop={'size': 18})
        plt.show()

class plot_multiple_formation_E:
    """ plot multiple formation energies together.
        there should be a input file, containing the name of each formation energy file
        and y_min and y_max
        For example, prepare a file, named as plot_together_Hf_La.in
        In the file,
        Hf_imp_C_poor.in_out
        Hf_vac_C_pool.in_out
        La_imp_C_poor.in_out
        La_vac_C_poor.in_out """

    def __init__(self, filein, x_min=None, x_max=None, y_min=None, y_max=None):
        """initialize an instance"""
        self.filein = filein
        self.x_range_chk = False
        if x_min != None:
            self.x_range_chk = True
            self.x_min = x_min
            self.x_max = x_max
        else:
            self.x_min = 0.0
            self.x_max = 10.0

        self.y_range_chk = False
        if y_min != None:
            self.y_range_chk = True
            self.y_min = y_min
            self.y_max = y_max
        else:
            self.y_min = 0.0
            self.y_max = 10.0

        self.plot_all()

    def plot_all(self):
        fig = plt.figure(figsize=(8,8))
        axes = fig.add_subplot(1,1,1)
        with open(self.filein, 'r') as fin:
            for line in fin:
                f_name = re.findall(r"[\w.]+", line)[0]
                print("Adding the following to the plot: " + f_name + "\n")
                data = np.genfromtxt(f_name)
                x_data = data[:,0]
                y_data = data[:,1]
                axes.plot(x_data, y_data, '-', label= f_name)
        axes.legend(loc=0, fontsize=20)
        axes.set_xlabel('E - E_VBM (eV)', fontsize = 18)
        axes.set_ylabel('Formation Energy (eV)', fontsize = 18)
        axes.set_title('Defect Formation Energy', fontsize = 20)

        if self.x_range_chk:
            plt.xlim(self.x_min, self.x_max)

        if self.y_range_chk:
            plt.ylim(self.y_min, self.y_max)

        fig.savefig(str(self.filein) + '.png')
        plt.show(fig)
        
        
# Following commands will run the above script.        
# For a single charge state
calculate_E_formation('dfe.in')
# For multiple charge state (which you are looking for, I believe)
plot_multiple_formation_E('multiple_dfe.in')
