"""
Extractor for Dialfa price list from Excel files.
Parses DIALFA/ACCESORIOS FEBRERO 2024.xls to extract current company prices.
"""
import pandas as pd
import logging
import xlrd
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DialfaExtractor:
    def __init__(self, file_path='DIALFA/ACCESORIOS FEBRERO 2024.xls'):
        """Initialize with path to Dialfa Excel file."""
        self.file_path = file_path
    
    def extract(self):
        """
        Extract product data from Dialfa Excel file.
        The file has headers starting at row 7, with product type in row 7,
        espesor in row 8, and data starting from row 9.
        Column 0 = Size, other columns = different product types with prices
        """
        try:
            logger.info(f"Extracting data from {self.file_path}")
            
            # Read file without header
            df = pd.read_excel(self.file_path, engine='xlrd', header=None)
            
            logger.info(f"Raw data shape: {df.shape}")
            
            # Extract header information
            # Row 7 contains product types (CODOS 90°, etc.)
            # Row 8 contains espesor (STD, etc.)
            # Row 9 onwards contains data: col 0 = size, other cols = prices
            
            header_row = 7
            espesor_row = 8
            data_start_row = 9
            
            # Get product types from row 7
            product_types = df.iloc[header_row].fillna('')
            # Get espesor from row 8
            espesores = df.iloc[espesor_row].fillna('')
            
            # Extract data starting from row 9
            data_rows = []
            for idx in range(data_start_row, len(df)):
                row = df.iloc[idx]
                size = row.iloc[0]
                
                # Skip if size is empty or NaN
                if pd.isna(size) or str(size).strip() == '':
                    continue
                
                # Extract each product type's price from this row
                for col_idx in range(1, len(row)):
                    price = row.iloc[col_idx]
                    
                    # Skip if price is empty, NaN, or 0
                    if pd.isna(price) or price == 0:
                        continue
                    
                    try:
                        price = float(price)
                        if price <= 0:
                            continue
                    except:
                        continue
                    
                    producto = str(product_types.iloc[col_idx]).strip()
                    espesor = str(espesores.iloc[col_idx]).strip()
                    
                    # Skip if no product type
                    if not producto or producto == 'nan':
                        continue
                    
                    data_rows.append({
                        'codigo': f"{producto}-{size}-{espesor}".replace('nan', '').replace('--', '-'),
                        'descripcion': f"{producto} {size} {espesor}".strip(),
                        'precio': price,
                        'tipo_serie': producto,
                        'espesor': espesor if espesor != 'nan' else '',
                        'size': str(size).strip()
                    })
            
            result_df = pd.DataFrame(data_rows)
            
            logger.info(f"Extracted {len(result_df)} products from Dialfa")
            return result_df
            
        except Exception as e:
            logger.error(f"Error extracting Dialfa data: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=['codigo', 'descripcion', 'precio', 
                                        'tipo_serie', 'espesor', 'size'])
    
    def _find_column(self, df, possible_names):
        """Find a column by trying multiple possible names."""
        for name in possible_names:
            if name in df.columns:
                return name
        return None

