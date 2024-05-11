# Encoding elements chemical formula

For embedding chemical formulae in data frame.

Input files should have a specific structure. There must be a column 'Formula' containing chemical formula names and rest of the columns have their title as different element name. 

## How to use ?
Edit _elementsCommaEncoding_ list of tuples; in order the second value contains the value that you would like to encode.

For separating elements from the formula and writing them in separate columns edit elementsCommaEnergy as following
```python
elementsCommaEncoding = [('H', 'H'), ('He', 'He'), ('Li', 'Li'), ('Be', 'Be'), ('B', 'B'), ('C', 'C'), ... ]
```

For ordinal encoding you may edit it as following,
```python
elementsCommaEncoding = [('H', 1), ('He', 2), ('Li', 3), ('Be', 4), ('B', 5), ('C', 6) ...  ]
```

For encoding something arbitrary, for instance to encode presence of an element by its ionisation energy,
```python
elementsCommaEncoding = [('H', 1312), ('He', 2372), ('Li', 520.3), ('Be', 899.5), ('B', 800.7), ... ]
```
