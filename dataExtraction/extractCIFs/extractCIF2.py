""" This reads MPIDs from a CSV file and saves the information in cif file """

import pandas as pd
from pymatgen.ext.matproj import MPRester

# Replace 'API_KEY' with your actual Materials Project API key
api_key = 'API_KEY'

# Read the CSV file with the 'Charge IDs' column
csv_file_path = 'datafile.csv'  # Replace with your CSV file/ file path
data = pd.read_csv(csv_file_path)

# specifying column name 'Material ID' for reading material Ids
material_ids = data['Material ID'].tolist()

mpr = MPRester(api_key)

for material_id in material_ids:
    try:
        structure = mpr.get_structure_by_material_id(material_id)
        cif_filename = f'{material_id.replace("mp-", "")}.cif' # Removing 'mp-' from the name(did this due to my preferences)
        
        # Save the structure as a CIF file using to
        structure.to(filename=cif_filename, fmt='cif')
        print(f'CIF file saved for {material_id}')
    except Exception as e:
        print(f'Error retrieving CIF file for {material_id}: {e}')
