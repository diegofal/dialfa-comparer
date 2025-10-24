"""
Script to inspect database and Excel file structures
"""
import pyodbc
import pandas as pd

print("=" * 80)
print("INSPECTING DATABASE STRUCTURE")
print("=" * 80)

try:
    conn = pyodbc.connect(
        'DRIVER={ODBC Driver 17 for SQL Server};'
        'SERVER=localhost;'
        'DATABASE=Spisa_local;'
        'UID=sa;'
        'PWD=Transc0reTransc0re!'
    )
    cursor = conn.cursor()
    
    # Get column names
    cursor.execute("SELECT TOP 1 * FROM dbo.articulos")
    columns = [column[0] for column in cursor.description]
    print("\nDatabase columns:")
    for i, col in enumerate(columns, 1):
        print(f"  {i}. {col}")
    
    # Get sample data
    cursor.execute("SELECT TOP 5 * FROM dbo.articulos")
    rows = cursor.fetchall()
    print(f"\nTotal columns: {len(columns)}")
    print(f"Sample rows retrieved: {len(rows)}")
    
    conn.close()
    
except Exception as e:
    print(f"Database error: {e}")

print("\n" + "=" * 80)
print("INSPECTING DIALFA EXCEL FILE")
print("=" * 80)

try:
    df = pd.read_excel('DIALFA/ACCESORIOS FEBRERO 2024.xls', engine='xlrd', header=None)
    print(f"\nShape: {df.shape}")
    print("\nFirst 10 rows:")
    print(df.head(10))
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 80)
print("INSPECTING CITIZEN PROFORMA FILE 1")
print("=" * 80)

try:
    df = pd.read_excel('ULTIMAS PROFORMA DE CITIZEN/2417-DIALFA-SRL-ARGENTINA-CS-FITTINGS-PI-2417-DT-07-06-22.xls', 
                       engine='xlrd', header=None)
    print(f"\nShape: {df.shape}")
    print("\nFirst 15 rows:")
    print(df.head(15))
except Exception as e:
    print(f"Error: {e}")


