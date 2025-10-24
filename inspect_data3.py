"""Full column inspection"""
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("DIALFA FILE - Row 7 (headers) and data rows:")
df = pd.read_excel('DIALFA/ACCESORIOS FEBRERO 2024.xls', engine='xlrd', header=None)
print("Row 7 (header):")
print(df.iloc[7])
print("\nRow 8 (subheader):")
print(df.iloc[8])
print("\nRow 9 (first data):")
print(df.iloc[9])

print("\n\nCITIZEN FILE - Row 9 (headers) and data rows:")
df = pd.read_excel('ULTIMAS PROFORMA DE CITIZEN/2417-DIALFA-SRL-ARGENTINA-CS-FITTINGS-PI-2417-DT-07-06-22.xls', 
                   engine='xlrd', header=None)
print("Row 9 (header):")
print(df.iloc[9])
print("\nRow 10 (first data):")
print(df.iloc[10])
print("\nRow 11 (second data):")
print(df.iloc[11])


