# Encoding elements chemical formula

For embedding chemical formulae in data frame. This code can be used for embedding compositional information to the data frame or extracting the elements from chemical formula and writing them in separate column. 

Input files should have a specific structure. There must be a column 'Formula' containing chemical formula names and rest of the columns have their title as different element name. _data1.csv_ shows the format of input file.

## How to use ?

The code contains a list of tuples named `elementsCommaEncoding` which has to be edited appropriately to fit the specific use case of the user.

**Step#1**

Edit the `elementsCommaEncoding` list of tuples in [code](elementsEncoding.py); in order, the second value contains the value that you would like to encode.

For example,
1. For separating elements from the formula and writing them in separate columns edit elementsCommaEnergy as following
```python
elementsCommaEncoding = [('H', 'H'), ('He', 'He'), ('Li', 'Li'), ('Be', 'Be'), ('B', 'B'), ('C', 'C'), ... ]
```

[See output](example/case1_output.csv)

2. For ordinal encoding you may edit it as following,
```python
elementsCommaEncoding = [('H', 1), ('He', 2), ('Li', 3), ('Be', 4), ('B', 5), ('C', 6) ...  ]
```
[See output](example/case2_output.csv)

3. For encoding something arbitrary, for instance to encode presence of an element by its ionisation energy,
```python
elementsCommaEncoding = [('H', 1312), ('He', 2372), ('Li', 520.3), ('Be', 899.5), ('B', 800.7), ... ]
```

[See output](example/case3_output.csv)

**Step#2**

Specify the path of input file (here, data1.csv) and run the program by 

```bash
$ python3 elementsEncoding.py
```
