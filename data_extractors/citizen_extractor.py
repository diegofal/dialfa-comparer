"""
Extractor for Citizen supplier proforma files.
Parses both Citizen XLS proforma files and merges the results.
"""
import pandas as pd
import logging
import xlrd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CitizenExtractor:
    def __init__(self, file_paths=None):
        """Initialize with paths to Citizen proforma files."""
        if file_paths is None:
            self.file_paths = [
                'ULTIMAS PROFORMA DE CITIZEN/2417-DIALFA-SRL-ARGENTINA-CS-FITTINGS-PI-2417-DT-07-06-22.xls',
                'ULTIMAS PROFORMA DE CITIZEN/2805-DIALFA-SRL-ARGENTINA-CS-FITTINGS-PI-NO-2805.xls'
            ]
        else:
            self.file_paths = file_paths
    
    def extract(self):
        """
        Extract supplier data from both Citizen proforma files.
        Returns a merged DataFrame with columns:
        - part_number: Supplier part number
        - descripcion: Product description
        - precio_usd: Unit price in USD
        - tipo_serie: Product type/series (extracted from part number)
        - espesor: Thickness
        - size: Size
        """
        all_data = []
        
        for file_path in self.file_paths:
            try:
                logger.info(f"Extracting data from {file_path}")
                df = self._extract_single_file(file_path)
                if not df.empty:
                    df['source_file'] = file_path.split('/')[-1]
                    all_data.append(df)
            except Exception as e:
                logger.error(f"Error extracting from {file_path}: {e}")
        
        if not all_data:
            logger.warning("No data extracted from Citizen proforma files")
            return pd.DataFrame(columns=['part_number', 'descripcion', 'precio_usd', 'cantidad',
                                        'tipo_serie', 'espesor', 'size'])
        
        # Merge all dataframes
        merged_df = pd.concat(all_data, ignore_index=True)
        
        # DO NOT collapse duplicates here - keep all records from all proformas
        # This allows price_calculator.py to calculate min/max/weighted average correctly
        # Each proforma may have different prices for the same product, which is important info
        
        logger.info(f"Extracted {len(merged_df)} total products from Citizen proforma files")
        logger.info(f"Products from multiple proformas will be compared for min/max/weighted prices")
        return merged_df
    
    def _extract_single_file(self, file_path):
        """
        Extract data from a single Citizen proforma file.
        Header is at row 9 (0-indexed), data starts at row 10.
        Columns: NO., DESCRIPTION, SIZE (inch), THICKNESS, QUANTITY, WEIGHT, TOTAL WEIGHT, UNIT PRICE, T.AMOUNT
        """
        try:
            # Read file without header
            df = pd.read_excel(file_path, engine='xlrd', header=None)
            
            logger.info(f"Raw data shape: {df.shape}")
            
            # Header is at row 9, data starts at row 10
            header_row = 9
            data_start_row = 10
            
            # Get column names from header row
            headers = df.iloc[header_row].tolist()
            logger.info(f"Headers: {headers}")
            
            # Extract data rows
            data_df = df.iloc[data_start_row:].copy()
            data_df.columns = range(len(data_df.columns))
            
            # Map columns based on position:
            # Col 0: NO., Col 1: DESCRIPTION, Col 2: SIZE, Col 3: THICKNESS, 
            # Col 7: UNIT PRICE (USD), Col 4: QUANTITY
            
            result_rows = []
            for idx, row in data_df.iterrows():
                try:
                    # Skip if first column is empty (end of data)
                    if pd.isna(row.iloc[0]):
                        break
                    
                    no = str(row.iloc[0]).strip()
                    description = str(row.iloc[1]).strip() if len(row) > 1 else ''
                    size = str(row.iloc[2]).strip() if len(row) > 2 else ''
                    thickness = str(row.iloc[3]).strip() if len(row) > 3 else ''
                    quantity = row.iloc[4] if len(row) > 4 else 1  # Get quantity
                    unit_price = row.iloc[7] if len(row) > 7 else None
                    
                    # Skip if no valid data
                    if description == 'nan' or description == '':
                        continue
                    
                    # Convert price to float
                    try:
                        unit_price = float(unit_price)
                        if unit_price <= 0:
                            continue
                    except:
                        continue
                    
                    # Convert quantity to float (default to 1 if invalid)
                    try:
                        quantity = float(quantity)
                        if quantity <= 0:
                            quantity = 1
                    except:
                        quantity = 1
                    
                    result_rows.append({
                        'part_number': f"CS-{no}",
                        'descripcion': f"{description} {size} {thickness}",
                        'precio_usd': unit_price,
                        'cantidad': quantity,
                        'tipo_serie': self._extract_tipo_from_desc(description),
                        'espesor': thickness.replace('\n', ' ').strip(),
                        'size': size.replace('\n', ' ').strip()
                    })
                    
                except Exception as e:
                    logger.warning(f"Error processing row {idx}: {e}")
                    continue
            
            result_df = pd.DataFrame(result_rows)
            logger.info(f"Extracted {len(result_df)} items from {file_path}")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error in _extract_single_file: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def _find_column(self, df, possible_names):
        """Find a column by trying multiple possible names."""
        for name in possible_names:
            if name in df.columns:
                return name
        return None
    
    def _extract_tipo_serie(self, part_number):
        """Extract tipo_serie from part number."""
        try:
            # Example: CS-1234 -> CS
            parts = str(part_number).split('-')
            if len(parts) > 0:
                return parts[0].strip().upper()
        except:
            pass
        return ''
    
    def _extract_tipo_from_desc(self, description):
        """
        Extract product type from description to match database tipo_serie format.
        Database uses specific formats like:
        - "90D LR ELBOW" (soldable, long radius)
        - "90D SR ELBOW" (soldable, short radius)
        - "45D LR ELBOW" (45 degree, long radius)
        - "TEE"
        - etc.
        
        This should NOT try to "normalize" to Spanish - keep English format to match DB.
        """
        try:
            desc = str(description).upper().strip()
            
            # Keep the exact format from the description
            # Don't translate to Spanish - the database has English tipo_serie
            
            # For elbows, extract the full type: "90D LR ELBOW", "90D SR ELBOW", "45D LR ELBOW", etc.
            if 'ELBOW' in desc:
                # Extract the full elbow type (e.g., "90D LR ELBOW", "90D SR ELBOW", "45D LR ELBOW")
                words = desc.split()
                elbow_type_parts = []
                for i, word in enumerate(words):
                    if 'ELBOW' in word:
                        # Get the words before ELBOW (like "90D LR" or "45D SR")
                        start_idx = max(0, i-2)
                        elbow_type_parts = words[start_idx:i+1]
                        break
                
                if elbow_type_parts:
                    return ' '.join(elbow_type_parts)
                else:
                    # Fallback: just return "ELBOW" with degree if found
                    if '90D' in desc or '90 ' in desc:
                        if 'SR' in desc:
                            return '90D SR ELBOW'
                        else:
                            return '90D LR ELBOW'
                    elif '45D' in desc or '45 ' in desc:
                        return '45D LR ELBOW'
                    return 'ELBOW'
            
            # For TEE, CAP, COUPLING, REDUCER - keep as-is
            elif 'TEE' in desc:
                return 'TEE'
            elif 'CAP' in desc:
                return 'CAP'
            elif 'COUPLING' in desc:
                return 'COUPLING'
            elif 'REDUCER' in desc or 'RED' in desc:
                return 'REDUCER'
            else:
                # Return first word as type
                return desc.split()[0] if desc else ''
        except Exception as e:
            logger.warning(f"Error extracting tipo from description '{description}': {e}")
            return ''
    
    def _extract_espesor(self, description):
        """Extract espesor (thickness) from description."""
        try:
            import re
            desc = str(description).upper()
            # Look for patterns like "SCH 40", "SCH40", "STD", etc.
            match = re.search(r'SCH\s*(\d+|STD|XS|XXS)', desc)
            if match:
                return match.group(0).replace(' ', '')
        except:
            pass
        return ''
    
    def _extract_size(self, description):
        """Extract size from description."""
        try:
            import re
            desc = str(description)
            # Look for size patterns like 1/2", 3/4", 1", etc.
            match = re.search(r'(\d+/?\d*)\s*["\']', desc)
            if match:
                return match.group(1)
            # Look for size in mm or inches
            match = re.search(r'(\d+)\s*(MM|INCH)', desc, re.IGNORECASE)
            if match:
                return match.group(1) + match.group(2).upper()
        except:
            pass
        return ''

