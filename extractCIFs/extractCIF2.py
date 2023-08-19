""" This reads MPIDs from a CSV file and saves the information in cif file """

import pandas as pd
from pymatgen.ext.matproj import MPRester

# Replace 'API_KEY' with your actual Materials Project API key
api_key = 'API_KEY'

# Read the CSV file with the 'Charge IDs' column
csv_file_path = 'final_combined_data3.csv'  # Replace with your CSV file path
data = pd.read_csv(csv_file_path)

# Extract Material IDs from the 'Charge IDs' column
material_ids = data['Charge ID'].tolist()

# Initialize the MPRester with your API key
mpr = MPRester(api_key)

# Loop through the Material IDs and retrieve structures
for material_id in material_ids:
    try:
        # Get the structure using the Material ID
        structure = mpr.get_structure_by_material_id(material_id)
        
        # Remove 'mp-' from the name
        cif_filename = f'{material_id.replace("mp-", "")}.cif'
        
        # Save the structure as a CIF file
        structure.to(filename=cif_filename, fmt='cif')
        print(f'CIF file saved for {material_id}')
    except Exception as e:
        print(f'Error retrieving CIF file for {material_id}: {e}')
