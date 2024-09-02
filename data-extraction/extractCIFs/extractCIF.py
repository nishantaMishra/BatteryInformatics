""" Program for extracting cif files from materials project using REST API """

from pymatgen.ext.matproj import MPRester

# Replace 'API_KEY' with your actual Materials Project API key
api_key = 'API_KEY'

# List of Material IDs for which you want to extract CIF files
material_ids = ['mp-123456', 'mp-789012', 
		'mp-345678', 'mp-10810', 
		'mp-773138', 'mp-773190', 
		'mp-771767', 'mp-510755', 
		'mp-772152', 'mp-756330', 
		'mp-771032', 'mp-771164' ]

mpr = MPRester(api_key)

for material_id in material_ids:
    try:
        # Get the structure
        structure = mpr.get_structure_by_material_id(material_id)
        
        # using 'to' method to save as cif
        cif_filename = f'{material_id}.cif'
        structure.to(filename=cif_filename, fmt='cif')
        print(f'CIF file saved for {material_id}')
    except Exception as e:
        print(f'Error retrieving CIF file for {material_id}: {e}')
