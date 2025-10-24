"""More detailed inspection"""
import pandas as pd

print("DIALFA FILE - Looking for data rows:")
df = pd.read_excel('DIALFA/ACCESORIOS FEBRERO 2024.xls', engine='xlrd', header=None)
print(df.iloc[6:30])  # Show rows around where data might be

print("\n\nCITIZEN FILE - Looking for column headers and data:")
df = pd.read_excel('ULTIMAS PROFORMA DE CITIZEN/2417-DIALFA-SRL-ARGENTINA-CS-FITTINGS-PI-2417-DT-07-06-22.xls', 
                   engine='xlrd', header=None)
print(df.iloc[8:20])  # Show rows around row 9 which seems to have headers


