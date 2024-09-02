# BatteryData
This repository contains a Python program that retrieves battery data from the Materials Project database. 
It uses the [REST API (new API)](https://github.com/materialsproject/api) of Materials Project for the task.

## Extracting Data from MP
[This program](extractAPI3Complete.py) can be used for extracting data from Materials Project using API.
The program will extract specified data and save it in a csv file.

>Note: In its current form the program is going to retrive information about all the battery compounds in MP(thousands of them). But if one is interested in compounds of particular elements then `elements` of `docs = mpr.insertion_electrodes.search` must be accordingly edited.


## Extracting Crystallographic Information File (CIF) 
For extracting CIF files see the [extractCIFs](https://github.com/nishantaMishra/BatteryData/tree/main/extractCIFs) directory.
