"""
Extractor for Zaloze competitor price list.
Reads from tablas_accesorios_zaloze.xlsx with 3 tabs.
"""
import pandas as pd
import os
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ZalozeExtractor:
    def __init__(self, excel_file='PRECIOS ZALOZE/tablas_accesorios_zaloze.xlsx'):
        """Initialize with path to Excel file."""
        self.excel_file = excel_file
        
        # Sheet names in the Excel file
        self.sheet_names = {
            'tabla1': 'Radio Largo',
            'tabla2': 'Radio Corto-Tes-Casquetes',
            'tabla3': 'Reducciones'
        }
    
    @staticmethod
    def normalize_size(size_str):
        """
        Convert Unicode fractions to ASCII format to match database.
        Examples:
          ½ -> 1/2
          ¾ -> 3/4
          ¼ -> 1/4
          1½ -> 1 1/2
          2 x 1½ -> 2 x 1 1/2
        """
        if pd.isna(size_str):
            return size_str
        
        size_str = str(size_str)
        
        # Unicode to ASCII fraction mapping
        fraction_map = {
            '½': '1/2',
            '¾': '3/4',
            '¼': '1/4',
            '⅛': '1/8',
            '⅜': '3/8',
            '⅝': '5/8',
            '⅞': '7/8',
        }
        
        # First, handle compound fractions like "1½" -> "1 1/2"
        # Match patterns like digit followed by fraction
        for unicode_frac, ascii_frac in fraction_map.items():
            # Pattern: digit followed immediately by unicode fraction (e.g., "1½")
            pattern = r'(\d)' + re.escape(unicode_frac)
            size_str = re.sub(pattern, r'\1 ' + ascii_frac, size_str)
            
            # Then replace standalone fractions
            size_str = size_str.replace(unicode_frac, ascii_frac)
        
        return size_str
    
    def extract(self):
        """
        Extract competitor prices from Zaloze Excel file.
        Returns a DataFrame with columns:
        - codigo: Product code
        - descripcion: Product description
        - precio: Zaloze price
        - tipo_serie: Product type/series
        - espesor: Thickness
        - size: Size
        """
        if not os.path.exists(self.excel_file):
            logger.error(f"Excel file not found: {self.excel_file}")
            return pd.DataFrame(columns=['codigo', 'descripcion', 'precio', 
                                        'tipo_serie', 'espesor', 'size'])
        
        all_data = []
        
        # Read Excel file
        excel_file = pd.ExcelFile(self.excel_file)
        
        # Process Tabla 1: Codos Radio Largo
        if self.sheet_names['tabla1'] in excel_file.sheet_names:
            df1_raw = pd.read_excel(excel_file, sheet_name=self.sheet_names['tabla1'])
            df1 = self._process_tabla1_excel(df1_raw)
            all_data.append(df1)
            logger.info(f"Loaded {len(df1)} items from Tabla 1 (Codos Radio Largo)")
        
        # Process Tabla 2: Codos Radio Corto + Tes + Casquetes
        if self.sheet_names['tabla2'] in excel_file.sheet_names:
            df2_raw = pd.read_excel(excel_file, sheet_name=self.sheet_names['tabla2'])
            df2 = self._process_tabla2_excel(df2_raw)
            all_data.append(df2)
            logger.info(f"Loaded {len(df2)} items from Tabla 2 (Corto + Tes + Casquetes)")
        
        # Process Tabla 3: Reducciones
        if self.sheet_names['tabla3'] in excel_file.sheet_names:
            df3_raw = pd.read_excel(excel_file, sheet_name=self.sheet_names['tabla3'])
            df3 = self._process_tabla3_excel(df3_raw)
            all_data.append(df3)
            logger.info(f"Loaded {len(df3)} items from Tabla 3 (Reducciones)")
        
        if not all_data:
            logger.error("No data could be extracted from Excel file")
            return pd.DataFrame(columns=['codigo', 'descripcion', 'precio', 
                                        'tipo_serie', 'espesor', 'size'])
        
        # Merge all data
        merged_df = pd.concat(all_data, ignore_index=True)
        
        logger.info(f"Total {len(merged_df)} unique products from Zaloze")
        return merged_df
    
    def _process_tabla1_excel(self, df_raw):
        """Process Tabla 1 from Excel: Codos Radio Largo 90/45/180."""
        # First row contains headers, data starts from row 1
        headers = df_raw.iloc[0].tolist()
        
        # Create a clean DataFrame with proper column names
        df = pd.DataFrame({
            'size': df_raw.iloc[1:, 0].values,
            'codo_radio_largo_90_std': df_raw.iloc[1:, 1].values,
            'codo_radio_largo_90_xs': df_raw.iloc[1:, 2].values,
            'codo_radio_largo_45_std': df_raw.iloc[1:, 3].values,
            'codo_radio_largo_45_xs': df_raw.iloc[1:, 4].values,
            'cr_largo_corto_180_std': df_raw.iloc[1:, 5].values,
            'cr_largo_corto_180_xs': df_raw.iloc[1:, 6].values
        })
        
        # Normalize sizes to ASCII format
        df['size'] = df['size'].apply(self.normalize_size)
        
        return self._process_tabla1(df)
    
    def _process_tabla2_excel(self, df_raw):
        """Process Tabla 2 from Excel: Codos Radio Corto + Tes + Casquetes."""
        # First row contains headers, data starts from row 1
        headers = df_raw.iloc[0].tolist()
        
        # Create a clean DataFrame with proper column names
        df = pd.DataFrame({
            'size': df_raw.iloc[1:, 0].values,
            'codo_radio_corto_90_std': df_raw.iloc[1:, 1].values,
            'codo_radio_corto_90_xs': df_raw.iloc[1:, 2].values,
            'tes_std': df_raw.iloc[1:, 3].values,
            'tes_xs': df_raw.iloc[1:, 4].values,
            'casquetes_semielipticos_std': df_raw.iloc[1:, 5].values,
            'casquetes_semielipticos_xs': df_raw.iloc[1:, 6].values
        })
        
        # Normalize sizes to ASCII format
        df['size'] = df['size'].apply(self.normalize_size)
        
        return self._process_tabla2(df)
    
    def _process_tabla3_excel(self, df_raw):
        """Process Tabla 3 from Excel: Reducciones."""
        # First row contains headers, data starts from row 1
        headers = df_raw.iloc[0].tolist()
        
        # Create a clean DataFrame with proper column names
        df = pd.DataFrame({
            'size': df_raw.iloc[1:, 0].values,
            'red_con_std': df_raw.iloc[1:, 1].values,
            'red_exc_std': df_raw.iloc[1:, 2].values,
            'te_red_std': df_raw.iloc[1:, 3].values,
            'red_con_xs': df_raw.iloc[1:, 4].values,
            'red_exc_xs': df_raw.iloc[1:, 5].values,
            'te_red_xs': df_raw.iloc[1:, 6].values
        })
        
        # Normalize sizes to ASCII format
        df['size'] = df['size'].apply(self.normalize_size)
        
        return self._process_tabla3(df)
    
    def _process_tabla1(self, df):
        """Process Tabla 1: Codos Radio Largo 90/45/180."""
        products = []
        
        # Helper function to validate prices
        def is_valid_price(val):
            if pd.isna(val):
                return False
            if isinstance(val, str) and 'consultar' in val.lower():
                return False
            try:
                float(val)
                return True
            except (ValueError, TypeError):
                return False
        
        for _, row in df.iterrows():
            size = row['size']
            
            # Codo Radio Largo 90° STD
            if is_valid_price(row.get('codo_radio_largo_90_std')):
                products.append({
                    'codigo': f"ZALOZE-CODO RADIO LARGO 90°-{size}-STD",
                    'descripcion': f"CODO RADIO LARGO 90° {size} STD",
                    'precio': float(row['codo_radio_largo_90_std']),
                    'tipo_serie': 'CODO RADIO LARGO 90°',
                    'espesor': 'STD',
                    'size': size
                })
            
            # Codo Radio Largo 90° XS
            if is_valid_price(row.get('codo_radio_largo_90_xs')):
                products.append({
                    'codigo': f"ZALOZE-CODO RADIO LARGO 90°-{size}-XS",
                    'descripcion': f"CODO RADIO LARGO 90° {size} XS",
                    'precio': float(row['codo_radio_largo_90_xs']),
                    'tipo_serie': 'CODO RADIO LARGO 90°',
                    'espesor': 'XS',
                    'size': size
                })
            
            # Codo Radio Largo 45° STD
            if is_valid_price(row.get('codo_radio_largo_45_std')):
                products.append({
                    'codigo': f"ZALOZE-CODO RADIO LARGO 45°-{size}-STD",
                    'descripcion': f"CODO RADIO LARGO 45° {size} STD",
                    'precio': float(row['codo_radio_largo_45_std']),
                    'tipo_serie': 'CODO RADIO LARGO 45°',
                    'espesor': 'STD',
                    'size': size
                })
            
            # Codo Radio Largo 45° XS
            if is_valid_price(row.get('codo_radio_largo_45_xs')):
                products.append({
                    'codigo': f"ZALOZE-CODO RADIO LARGO 45°-{size}-XS",
                    'descripcion': f"CODO RADIO LARGO 45° {size} XS",
                    'precio': float(row['codo_radio_largo_45_xs']),
                    'tipo_serie': 'CODO RADIO LARGO 45°',
                    'espesor': 'XS',
                    'size': size
                })
            
            # CR. Largo y Corto 180° STD
            if is_valid_price(row.get('cr_largo_corto_180_std')):
                products.append({
                    'codigo': f"ZALOZE-CR. LARGO Y CORTO 180°-{size}-STD",
                    'descripcion': f"CR. LARGO Y CORTO 180° {size} STD",
                    'precio': float(row['cr_largo_corto_180_std']),
                    'tipo_serie': 'CR. LARGO Y CORTO 180°',
                    'espesor': 'STD',
                    'size': size
                })
            
            # CR. Largo y Corto 180° XS
            if is_valid_price(row.get('cr_largo_corto_180_xs')):
                products.append({
                    'codigo': f"ZALOZE-CR. LARGO Y CORTO 180°-{size}-XS",
                    'descripcion': f"CODO RADIO LARGO 180° {size} XS",
                    'precio': float(row['cr_largo_corto_180_xs']),
                    'tipo_serie': 'CR. LARGO Y CORTO 180°',
                    'espesor': 'XS',
                    'size': size
                })
        
        return pd.DataFrame(products)
    
    def _process_tabla2(self, df):
        """Process Tabla 2: Codos Radio Corto + Tes + Casquetes."""
        products = []
        
        # Helper function to validate prices
        def is_valid_price(val):
            if pd.isna(val):
                return False
            if isinstance(val, str) and 'consultar' in val.lower():
                return False
            try:
                float(val)
                return True
            except (ValueError, TypeError):
                return False
        
        for _, row in df.iterrows():
            size = row['size']
            
            # Codo Radio Corto 90° STD
            if is_valid_price(row.get('codo_radio_corto_90_std')):
                products.append({
                    'codigo': f"ZALOZE-CODO RADIO CORTO 90°-{size}-STD",
                    'descripcion': f"CODO RADIO CORTO 90° {size} STD",
                    'precio': float(row['codo_radio_corto_90_std']),
                    'tipo_serie': 'CODO RADIO CORTO 90°',
                    'espesor': 'STD',
                    'size': size
                })
            
            # Codo Radio Corto 90° XS
            if is_valid_price(row.get('codo_radio_corto_90_xs')):
                products.append({
                    'codigo': f"ZALOZE-CODO RADIO CORTO 90°-{size}-XS",
                    'descripcion': f"CODO RADIO CORTO 90° {size} XS",
                    'precio': float(row['codo_radio_corto_90_xs']),
                    'tipo_serie': 'CODO RADIO CORTO 90°',
                    'espesor': 'XS',
                    'size': size
                })
            
            # Tes STD
            if is_valid_price(row.get('tes_std')):
                products.append({
                    'codigo': f"ZALOZE-TEE-{size}-STD",
                    'descripcion': f"TEE {size} STD",
                    'precio': float(row['tes_std']),
                    'tipo_serie': 'TEE',
                    'espesor': 'STD',
                    'size': size
                })
            
            # Tes XS
            if is_valid_price(row.get('tes_xs')):
                products.append({
                    'codigo': f"ZALOZE-TEE-{size}-XS",
                    'descripcion': f"TEE {size} XS",
                    'precio': float(row['tes_xs']),
                    'tipo_serie': 'TEE',
                    'espesor': 'XS',
                    'size': size
                })
            
            # Casquetes STD
            if is_valid_price(row.get('casquetes_semielipticos_std')):
                products.append({
                    'codigo': f"ZALOZE-CASQUETE-{size}-STD",
                    'descripcion': f"CASQUETE {size} STD",
                    'precio': float(row['casquetes_semielipticos_std']),
                    'tipo_serie': 'CASQUETE',
                    'espesor': 'STD',
                    'size': size
                })
            
            # Casquetes XS
            if is_valid_price(row.get('casquetes_semielipticos_xs')):
                products.append({
                    'codigo': f"ZALOZE-CASQUETE-{size}-XS",
                    'descripcion': f"CASQUETE {size} XS",
                    'precio': float(row['casquetes_semielipticos_xs']),
                    'tipo_serie': 'CASQUETE',
                    'espesor': 'XS',
                    'size': size
                })
        
        return pd.DataFrame(products)
    
    def _process_tabla3(self, df):
        """Process Tabla 3: Reducciones."""
        products = []
        for _, row in df.iterrows():
            size = row['size']
            
            # Skip "consultar" values
            def is_valid_price(val):
                if pd.isna(val):
                    return False
                if isinstance(val, str) and 'consultar' in val.lower():
                    return False
                return True
            
            # Red. Concéntrica STD
            if is_valid_price(row.get('red_con_std')):
                products.append({
                    'codigo': f"ZALOZE-RED. CONCENTRICA-{size}-STD",
                    'descripcion': f"RED. CONCENTRICA {size} STD",
                    'precio': float(row['red_con_std']),
                    'tipo_serie': 'RED. CONCENTRICA',
                    'espesor': 'STD',
                    'size': size
                })
            
            # Red. Excéntrica STD
            if is_valid_price(row.get('red_exc_std')):
                products.append({
                    'codigo': f"ZALOZE-RED. EXCENTRICA-{size}-STD",
                    'descripcion': f"RED. EXCENTRICA {size} STD",
                    'precio': float(row['red_exc_std']),
                    'tipo_serie': 'RED. EXCENTRICA',
                    'espesor': 'STD',
                    'size': size
                })
            
            # Te Reducción STD
            if is_valid_price(row.get('te_red_std')):
                products.append({
                    'codigo': f"ZALOZE-TE RED.-{size}-STD",
                    'descripcion': f"TE RED. {size} STD",
                    'precio': float(row['te_red_std']),
                    'tipo_serie': 'TE RED.',
                    'espesor': 'STD',
                    'size': size
                })
            
            # Red. Concéntrica XS
            if is_valid_price(row.get('red_con_xs')):
                products.append({
                    'codigo': f"ZALOZE-RED. CONCENTRICA-{size}-XS",
                    'descripcion': f"RED. CONCENTRICA {size} XS",
                    'precio': float(row['red_con_xs']),
                    'tipo_serie': 'RED. CONCENTRICA',
                    'espesor': 'XS',
                    'size': size
                })
            
            # Red. Excéntrica XS
            if is_valid_price(row.get('red_exc_xs')):
                products.append({
                    'codigo': f"ZALOZE-RED. EXCENTRICA-{size}-XS",
                    'descripcion': f"RED. EXCENTRICA {size} XS",
                    'precio': float(row['red_exc_xs']),
                    'tipo_serie': 'RED. EXCENTRICA',
                    'espesor': 'XS',
                    'size': size
                })
            
            # Te Reducción XS
            if is_valid_price(row.get('te_red_xs')):
                products.append({
                    'codigo': f"ZALOZE-TE RED.-{size}-XS",
                    'descripcion': f"TE RED. {size} XS",
                    'precio': float(row['te_red_xs']),
                    'tipo_serie': 'TE RED.',
                    'espesor': 'XS',
                    'size': size
                })
        
        return pd.DataFrame(products)
