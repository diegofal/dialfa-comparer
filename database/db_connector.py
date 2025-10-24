"""
SQL Server database connector for accessing the Spisa_local database
and retrieving product information from the articulos table.
"""
import pyodbc
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseConnector:
    def __init__(self, server='localhost', database='Spisa_local', 
                 username='sa', password='Transc0reTransc0re!'):
        """Initialize database connection parameters."""
        self.server = server
        self.database = database
        self.username = username
        self.password = password
        self.connection = None
        
    def connect(self):
        """Establish connection to SQL Server database."""
        try:
            # Try SQL Server Authentication first
            connection_string = (
                f'DRIVER={{ODBC Driver 17 for SQL Server}};'
                f'SERVER={self.server};'
                f'DATABASE={self.database};'
                f'UID={self.username};'
                f'PWD={self.password}'
            )
            self.connection = pyodbc.connect(connection_string)
            logger.info(f"Successfully connected to {self.database} database")
            return True
        except pyodbc.Error as e:
            logger.error(f"SQL Server Authentication failed: {e}")
            # Try Windows Authentication as fallback
            try:
                connection_string = (
                    f'DRIVER={{ODBC Driver 17 for SQL Server}};'
                    f'SERVER={self.server};'
                    f'DATABASE={self.database};'
                    f'Trusted_Connection=yes;'
                )
                self.connection = pyodbc.connect(connection_string)
                logger.info(f"Successfully connected using Windows Authentication")
                return True
            except pyodbc.Error as e2:
                logger.error(f"Windows Authentication also failed: {e2}")
                return False
    
    def get_articulos(self):
        """
        Retrieve all products from the articulos table with relevant fields.
        This is now the PRIMARY source of truth for products.
        Returns a pandas DataFrame with product matching fields.
        """
        if not self.connection:
            if not self.connect():
                raise Exception("Failed to connect to database")
        
        query = """
        SELECT 
            idArticulo,
            IdCategoria,
            codigo,
            descripcion,
            cantidad,
            preciounitario,
            orden,
            discontinuado,
            tipo,
            serie,
            espesor,
            size,
            proveedor,
            peso,
            precio_unitario_historico_1
        FROM dbo.articulos
        WHERE discontinuado = 0
        ORDER BY codigo
        """
        
        try:
            df = pd.read_sql(query, self.connection)
            
            # Create tipo_serie by combining tipo and serie
            df['tipo_serie'] = df['tipo'].astype(str) + (df['serie'].astype(str).replace('nan', ''))
            df['tipo_serie'] = df['tipo_serie'].str.strip()
            
            # Rename preciounitario to match expected column name
            df = df.rename(columns={
                'preciounitario': 'dialfa_precio'
            })
            
            # Clean up data types
            df['dialfa_precio'] = pd.to_numeric(df['dialfa_precio'], errors='coerce')
            df['espesor'] = df['espesor'].fillna('').astype(str)
            df['size'] = df['size'].fillna('').astype(str)
            df['tipo_serie'] = df['tipo_serie'].fillna('').astype(str)
            
            logger.info(f"Retrieved {len(df)} products from articulos table")
            logger.info(f"Products with prices: {df['dialfa_precio'].notna().sum()}")
            
            return df
        except Exception as e:
            logger.error(f"Error querying articulos table: {e}")
            raise
    
    def create_matching_key(self, row):
        """
        Create a composite matching key from tipo_serie, espesor, and size.
        This key will be used to match products across different price lists.
        """
        try:
            tipo_serie = str(row.get('tipo_serie', '')).strip().upper()
            espesor = str(row.get('espesor', '')).strip().upper()
            size = str(row.get('size', '')).strip().upper()
            return f"{tipo_serie}|{espesor}|{size}"
        except Exception as e:
            logger.warning(f"Error creating matching key: {e}")
            return None
    
    def get_articulos_with_matching_key(self):
        """
        Get articulos DataFrame with an additional matching_key column.
        """
        df = self.get_articulos()
        df['matching_key'] = df.apply(self.create_matching_key, axis=1)
        return df
    
    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

