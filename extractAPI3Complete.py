##########################################################################
#######  This program retrives entire entries of Battery Explorer. #######
##########################################################################
"""
 This program retrives all data of Battery Explorer of Materials project and 
 exports it in the form of a .csv file
"""
import pandas as pd
from mp_api.client import MPRester

with MPRester("gY51VNDnB45T6O32WEq1RPnGA5wqJNOU") as mpr: #Refer, https://github.com/materialsproject/api/blob/main/mp_api/client/routes/materials/electrodes.py
    docs = mpr.insertion_electrodes.search(fields=["battery_id", 
                                                   "material_ids",
                                                   "average_voltage",
                                                   "capacity_grav",
                                                   "capacity_vol",
                                                   "energy_grav",
                                                   "energy_vol",
                                                   "id_charge",
                                                   "id_discharge",
                                                   "working_ion",
                                                   "num_elements",
                                                   "max_delta_volume",
                                                   "max_voltage_step",
                                                   "stability_charge",
                                                   "stability_discharge",
                                                   "formula",
                                                   "battery_type",
                                                   "thermo_type",
                                                   "battery_formula",
                                                   "num_steps",
                                                   "last_updated",
                                                   "framework",
                                                   "framework_formula",
                                                   "elements",
                                                   "nelements",
                                                   "chemsys",
                                                   "formula_anonymous",
                                                   "formula_charge",
                                                   "formula_discharge",
                                                   "fracA_charge",
                                                   "fracA_discharge",
                                                   "host_structure",
                                                   "adj_pairs",
                                                   "entries_composition_summary",
                                                   "electrode_object",
                                                   "warnings"],
                                            #energy_grav=(0, 2),           These are different filters
                                            #average_voltage=(0.5, 1.6)    uncomment them for filtered   
                                            #elements=["Na", "Li", "Co"]   search
                                            )


# Creating a list of dictionaries to store the data
data_list = []
for details in docs:
    adj_pairs_data = []
    if hasattr(details, 'adj_pairs') and isinstance(details.adj_pairs, list) and details.adj_pairs:
        for pair in details.adj_pairs:
            if hasattr(pair, 'voltage') and hasattr(pair, 'working_ion'):
                voltage = pair.voltage
                working_ion = pair.working_ion
                adj_pairs_data.append(f"{voltage:.4f}, {working_ion}")
                
    data_list.append({
        'Battery ID': details.battery_id,
        'Material IDs': ', '.join(details.material_ids),
        'Gravimetric Capacity (mAh/g)': ', '.join(map(str, details.capacity_grav)) if isinstance(details.capacity_grav, list) else str(details.capacity_grav),
        'Volumetric Capacity (Ah/L)': ', '.join(map(str, details.capacity_vol)) if isinstance(details.capacity_vol, list) else str(details.capacity_vol),
        'Specific Energy (Wh/kg)': ', '.join(map(str, details.energy_grav)) if isinstance(details.energy_grav, list) else str(details.energy_grav),
        'Energy Density (Wh/L)': ', '.join(map(str, details.energy_vol)) if isinstance(details.energy_vol, list) else str(details.energy_vol),
        'Average Voltage (V)': ', '.join(map(str, details.average_voltage)) if isinstance(details.average_voltage, list) else str(details.average_voltage),
        'Charge ID': ', '.join(map(str, details.id_charge)) if isinstance(details.id_charge, list) else str(details.id_charge),
        'Discharge ID': ', '.join(map(str, details.id_discharge)) if isinstance(details.id_discharge, list) else str(details.id_discharge),
        'Working Ion' : ', '.join(map(str, details.working_ion)) if isinstance(details.working_ion, list) else str(details.working_ion),
        'Max Delta Volume' : ', '.join(map(str, details.max_delta_volume)) if isinstance(details.max_delta_volume, list) else str(details.max_delta_volume),
        'Max Voltage Step' : ', '.join(map(str, details.max_voltage_step)) if isinstance(details.max_voltage_step, list) else str(details.max_voltage_step),
        'Stability Charge' : ', '.join(map(str, details.stability_charge)) if isinstance(details.stability_charge, list) else str(details.stability_charge),
        'Stability Discharge' : ', '.join(map(str, details.stability_discharge)) if isinstance(details.stability_discharge, list) else str(details.stability_discharge),
        'Formula' : details.formula if hasattr(details, 'formula') else '', 
        'Number of Elements' : details.num_elements if hasattr(details, 'num_elements') else '',
        'Battery Type': details.battery_type if hasattr(details, 'battery_type') else '',
        'Thermo Type': details.thermo_type if hasattr(details, 'thermo_type') else '',
        'Battery Formula': details.battery_formula if hasattr(details, 'battery_formula') else '',
        'Number of Steps': details.num_steps if hasattr(details, 'num_steps') else '',
        'Last Updated': details.last_updated if hasattr(details, 'last_updated') else '',
        'Framework': details.framework if hasattr(details, 'framework') else '',
        'Framework Formula': details.framework_formula if hasattr(details, 'framework_formula') else '',
        'Elements': ', '.join([el.symbol for el in details.elements]) if hasattr(details, 'elements') and isinstance(details.elements, list) else '',
        'Number of Elements': details.nelements if hasattr(details, 'nelements') else '',
        'Chemical System': details.chemsys if hasattr(details, 'chemsys') else '',
        'Formula Anonymous': details.formula_anonymous if hasattr(details, 'formula_anonymous') else '',
        'Formula Charge': details.formula_charge if hasattr(details, 'formula_charge') else '',
        'Formula Discharge': details.formula_discharge if hasattr(details, 'formula_discharge') else '',
        'Fraction A Charge': details.fracA_charge if hasattr(details, 'fracA_charge') else '',
        'Fraction A Discharge': details.fracA_discharge if hasattr(details, 'fracA_discharge') else '',
        'Host Structure': details.host_structure if hasattr(details, 'host_structure') else '',
        'Adjacent Pairs': ', '.join(adj_pairs_data),
        'Entries Composition Summary': details.entries_composition_summary if hasattr(details, 'entries_composition_summary') else '',
        'Electrode Object': details.electrode_object if hasattr(details, 'electrode_object') else '',
        'Warnings': ', '.join(map(str, details.warnings)) if isinstance(details.warnings, list) else str(details.warnings)
    })

# Creating data frame and storing them in a .csv file
df = pd.DataFrame(data_list)
df.to_csv('battery_data.csv', index=False)


