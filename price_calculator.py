"""
Price calculator module for matching products and calculating margins.
Implements matching algorithm using embeddings (AI-powered) and database fields.
"""
import pandas as pd
import logging
from fuzzywuzzy import fuzz
from embedding_matcher import EmbeddingMatcher
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PriceCalculator:
    def __init__(self, db_articulos, dialfa_data, citizen_data, 
                 cintolo_data=None, zaloze_data=None, use_embeddings=True, cache_manager=None,
                 discount_percent=30, nationalization_percent=150):
        """
        Initialize with data from all sources.
        
        Args:
            db_articulos: DataFrame from database articulos table
            dialfa_data: DataFrame with Dialfa prices
            citizen_data: DataFrame with Citizen supplier prices
            cintolo_data: DataFrame with Cintolo competitor prices (optional)
            zaloze_data: DataFrame with Zaloze competitor prices (optional)
            use_embeddings: Use AI embeddings for matching (default: True)
            cache_manager: DataCache instance for caching embeddings
            discount_percent: Discount percentage to apply to ALL products (default: 30%)
            nationalization_percent: Percentage increase for FOB nationalization (default: 150%)
        """
        self.db_articulos = db_articulos
        self.dialfa_data = dialfa_data
        self.citizen_data = citizen_data
        self.cintolo_data = cintolo_data if cintolo_data is not None else pd.DataFrame()
        self.zaloze_data = zaloze_data if zaloze_data is not None else pd.DataFrame()
        
        # Pricing parameters
        self.discount_percent = discount_percent
        self.nationalization_percent = nationalization_percent
        
        logger.info(f"Pricing config: Discount {discount_percent}% universal, Nationalization +{nationalization_percent}%")
        
        # Initialize embedding matcher if enabled
        self.use_embeddings = use_embeddings
        self.embedding_matcher = None
        if use_embeddings:
            try:
                api_key = os.getenv('OPENAI_API_KEY')
                if api_key:
                    self.embedding_matcher = EmbeddingMatcher(api_key=api_key, cache_manager=cache_manager)
                    logger.info("✓ AI-powered matching enabled (using embeddings)")
                else:
                    logger.warning("OPENAI_API_KEY not found. Falling back to fuzzy matching.")
                    self.use_embeddings = False
            except Exception as e:
                logger.warning(f"Failed to initialize embedding matcher: {e}. Falling back to fuzzy matching.")
                self.use_embeddings = False
    
    def create_matching_key(self, row):
        """Create composite matching key from tipo_serie, espesor, size."""
        try:
            tipo_serie = str(row.get('tipo_serie', '')).strip().upper()
            espesor = str(row.get('espesor', '')).strip().upper()
            size = str(row.get('size', '')).strip().upper()
            
            # Remove 'nan' strings
            if tipo_serie == 'NAN':
                tipo_serie = ''
            if espesor == 'NAN':
                espesor = ''
            if size == 'NAN':
                size = ''
            
            # Normalize size: remove quotes and standardize format
            size = size.replace('"', '').replace("'", "").strip()
            # Normalize spaces in fractions: "1 1/2" should match "1.1/2" and "11/2"
            # First remove dots before digits that precede slashes
            import re
            size = re.sub(r'\.(\d+/)', r' \1', size)  # "1.1/2" -> "1 1/2"
            # Ensure there's a space before fractions like "11/2" -> "1 1/2"
            size = re.sub(r'(\d)([1234567890]/)', r'\1 \2', size)  # "11/2" -> "1 1/2"
            # Standardize multiple spaces to single space
            size = re.sub(r'\s+', ' ', size).strip()
            
            # Normalize tipo_serie variations
            # Normalize 90 degree elbows: "CODO 90", "90D LR ELBOW", "90 LR ELBOW", etc. → "CODO 90 LR"
            if '90' in tipo_serie and ('CODO' in tipo_serie or 'ELBOW' in tipo_serie or 'LR' in tipo_serie):
                # Check if it's long radius (LR) or short radius (SR)
                if 'SR' in tipo_serie or 'RADIO CORTO' in tipo_serie or 'SHORT RADIUS' in tipo_serie:
                    tipo_serie = 'CODO 90 SR'
                else:
                    # Default to LR (long radius) for 90 degree elbows
                    tipo_serie = 'CODO 90 LR'
            # Normalize 45 degree elbows
            elif '45' in tipo_serie and ('CODO' in tipo_serie or 'ELBOW' in tipo_serie):
                if 'SR' in tipo_serie or 'RADIO CORTO' in tipo_serie:
                    tipo_serie = 'CODO 45 SR'
                else:
                    tipo_serie = 'CODO 45 LR'
            
            # Normalize espesor
            if espesor in ['STANDARD', 'STD']:
                espesor = 'STD'
            elif espesor in ['EXTRA PESADO', 'XS', 'EX PESADO']:
                espesor = 'XS'
            
            key = f"{tipo_serie}|{espesor}|{size}"
            return key if key != '||' else None
        except Exception as e:
            logger.warning(f"Error creating matching key: {e}")
            return None
    
    def match_products(self):
        """
        Match products across all data sources using database as PRIMARY source.
        The database articulos table is the source of truth.
        Returns a comprehensive comparison DataFrame.
        """
        logger.info("Starting product matching process (Database as PRIMARY source)")
        
        # Start with DATABASE data as base (not Dialfa Excel)
        result = self.db_articulos.copy()
        
        # Add matching keys to all dataframes
        result['matching_key'] = result.apply(self.create_matching_key, axis=1)
        
        self.citizen_data['matching_key'] = self.citizen_data.apply(self.create_matching_key, axis=1)
        
        if not self.cintolo_data.empty:
            self.cintolo_data['matching_key'] = self.cintolo_data.apply(self.create_matching_key, axis=1)
        
        if not self.zaloze_data.empty:
            self.zaloze_data['matching_key'] = self.zaloze_data.apply(self.create_matching_key, axis=1)
        
        logger.info(f"Base products from database: {len(result)}")
        
        # Merge with Citizen data
        if not self.citizen_data.empty:
            # Calculate min, max, and weighted average prices per matching_key
            # Group by matching_key to get price statistics
            
            # Calculate weighted average (precio promedio ponderado)
            # Formula: Σ(precio × cantidad) / Σ(cantidad)
            def weighted_avg_price(group):
                if 'cantidad' in group.columns:
                    total_weighted = (group['precio_usd'] * group['cantidad']).sum()
                    total_quantity = group['cantidad'].sum()
                    if total_quantity > 0:
                        return total_weighted / total_quantity
                    else:
                        return group['precio_usd'].mean()
                else:
                    # Fallback to simple average if cantidad not available
                    return group['precio_usd'].mean()
            
            citizen_grouped = self.citizen_data.groupby('matching_key').agg({
                'precio_usd': ['min', 'max'],  # Min and max prices
                'descripcion': 'first',  # Take first description
                'size': 'first',
                'espesor': 'first'
            }).reset_index()
            
            # Calculate weighted average separately
            weighted_prices = self.citizen_data.groupby('matching_key').apply(weighted_avg_price).reset_index(name='precio_fob_ponderado')
            
            # Flatten column names
            citizen_grouped.columns = ['matching_key', 'precio_fob_min', 'precio_fob_max',
                                      'descripcion', 'size', 'espesor']
            
            # Merge weighted average
            citizen_grouped = citizen_grouped.merge(weighted_prices, on='matching_key', how='left')
            
            # Create combined citizen product display field
            citizen_grouped['citizen_producto'] = (
                citizen_grouped['descripcion'].astype(str) + ' ' + 
                citizen_grouped['size'].astype(str) + ' ' + 
                citizen_grouped['espesor'].astype(str)
            )
            
            # Rename columns for final dataframe
            citizen_grouped = citizen_grouped.rename(columns={
                'descripcion': 'citizen_descripcion',
                'size': 'citizen_size',
                'espesor': 'citizen_espesor'
            })
            
            result = result.merge(citizen_grouped, on='matching_key', how='left')
        else:
            result['citizen_producto'] = None
            result['citizen_descripcion'] = None
            result['citizen_size'] = None
            result['citizen_espesor'] = None
            result['precio_fob_min'] = None
            result['precio_fob_max'] = None
            result['precio_fob_ponderado'] = None
        
        # Merge with Cintolo data
        if not self.cintolo_data.empty:
            cintolo_merge = self.cintolo_data[['matching_key', 'codigo', 'descripcion', 'precio']].copy()
            cintolo_merge = cintolo_merge.rename(columns={
                'codigo': 'cintolo_codigo',
                'descripcion': 'cintolo_descripcion',
                'precio': 'cintolo_precio'
            })
            result = result.merge(cintolo_merge, on='matching_key', how='left')
        else:
            result['cintolo_codigo'] = None
            result['cintolo_descripcion'] = None
            result['cintolo_precio'] = None
        
        # Merge with Zaloze data
        if not self.zaloze_data.empty:
            zaloze_merge = self.zaloze_data[['matching_key', 'codigo', 'descripcion', 'precio']].copy()
            zaloze_merge = zaloze_merge.rename(columns={
                'codigo': 'zaloze_codigo',
                'descripcion': 'zaloze_descripcion',
                'precio': 'zaloze_precio'
            })
            result = result.merge(zaloze_merge, on='matching_key', how='left')
        else:
            result['zaloze_codigo'] = None
            result['zaloze_descripcion'] = None
            result['zaloze_precio'] = None
        
        logger.info(f"Matched {len(result)} products with exact key matching")
        
        # Try AI-powered matching for unmatched items (Citizen only for now)
        if self.use_embeddings and self.embedding_matcher:
            logger.info("🤖 Starting AI-powered matching with embeddings...")
            result = self._embedding_match_citizen(result)
        else:
            # Fallback to fuzzy matching
            logger.info("Using traditional fuzzy matching...")
            result = self._fuzzy_match_unmatched(result)
        
        return result
    
    def _embedding_match_citizen(self, df):
        """
        Apply AI-powered embedding matching for Citizen products that didn't match exactly.
        Only processes unmatched products for cost efficiency.
        """
        unmatched_mask = df['precio_fob_ponderado'].isna()
        unmatched_count = unmatched_mask.sum()
        
        if unmatched_count == 0:
            logger.info("✓ All Citizen products matched exactly with keys")
            return df
        
        logger.info(f"🔍 AI matching {unmatched_count} unmatched products against {len(self.citizen_data)} Citizen products...")
        
        # Only process unmatched products (cost optimization)
        unmatched_df = df[unmatched_mask].copy()
        
        try:
            # Use embedding matcher
            matched_df, stats = self.embedding_matcher.match_products(
                source_df=unmatched_df,
                target_df=self.citizen_data,
                source_name="Dialfa",
                target_name="Citizen",
                threshold=0.75  # Adjusted threshold based on testing
            )
            
            # Merge results back into main dataframe
            # Use embedding results if available
            for idx in matched_df.index:
                if pd.notna(matched_df.loc[idx, 'citizen_precio_usd_emb']):
                    # Get matched Citizen product
                    citizen_desc = matched_df.loc[idx, 'citizen_descripcion_emb']
                    citizen_size = matched_df.loc[idx, 'citizen_size_emb']
                    citizen_espesor = matched_df.loc[idx, 'citizen_espesor_emb']
                    precio_usd = matched_df.loc[idx, 'citizen_precio_usd_emb']
                    
                    df.loc[idx, 'citizen_producto'] = f"{citizen_desc} {citizen_size} {citizen_espesor}"
                    df.loc[idx, 'citizen_descripcion'] = citizen_desc
                    df.loc[idx, 'citizen_size'] = citizen_size
                    df.loc[idx, 'citizen_espesor'] = citizen_espesor
                    # For embeddings match, set all prices to the same value (no min/max available)
                    df.loc[idx, 'precio_fob_min'] = precio_usd
                    df.loc[idx, 'precio_fob_max'] = precio_usd
                    df.loc[idx, 'precio_fob_ponderado'] = precio_usd
                    score = matched_df.loc[idx, 'citizen_match_score']
                    df.loc[idx, 'match_type'] = f'AI Embedding ({score}%)'
            
            logger.info(f"✓ AI matching: {stats['matched']} new matches found ({stats['match_rate']}% match rate)")
            
        except Exception as e:
            logger.error(f"❌ Embedding matching failed: {e}")
            logger.info("Falling back to fuzzy matching...")
            return self._fuzzy_match_unmatched(df)
        
        return df
    
    def _fuzzy_match_unmatched(self, df):
        """Apply fuzzy matching for products that didn't match exactly (fallback method)."""
        unmatched_mask = df['precio_fob_ponderado'].isna()
        unmatched_count = unmatched_mask.sum()
        
        if unmatched_count == 0:
            logger.info("All products matched exactly")
            return df
        
        logger.info(f"Attempting fuzzy matching for {unmatched_count} unmatched products")
        
        for idx in df[unmatched_mask].index:
            # Use descripcion from database
            dialfa_desc = str(df.loc[idx, 'descripcion']).lower()
            
            # Find best match in Citizen data by description
            if not self.citizen_data.empty:
                best_score = 0
                best_match = None
                
                for _, citizen_row in self.citizen_data.iterrows():
                    citizen_desc = str(citizen_row['descripcion']).lower()
                    score = fuzz.token_set_ratio(dialfa_desc, citizen_desc)
                    
                    if score > best_score and score > 70:  # Threshold for fuzzy match
                        best_score = score
                        best_match = citizen_row
                
                if best_match is not None:
                    precio_usd = best_match['precio_usd']
                    df.loc[idx, 'citizen_producto'] = f"{best_match['descripcion']} {best_match['size']} {best_match['espesor']}"
                    df.loc[idx, 'citizen_descripcion'] = best_match['descripcion']
                    df.loc[idx, 'citizen_size'] = best_match['size']
                    df.loc[idx, 'citizen_espesor'] = best_match['espesor']
                    # For fuzzy match, set all prices to the same value (no min/max available)
                    df.loc[idx, 'precio_fob_min'] = precio_usd
                    df.loc[idx, 'precio_fob_max'] = precio_usd
                    df.loc[idx, 'precio_fob_ponderado'] = precio_usd
                    df.loc[idx, 'match_type'] = f'Fuzzy ({best_score}%)'
        
        newly_matched = df[unmatched_mask]['precio_fob_ponderado'].notna().sum()
        logger.info(f"Fuzzy matching found {newly_matched} additional matches")
        
        return df
    
    def calculate_margins(self, df):
        """
        Calculate margin %, markup %, and competitor comparisons.
        Includes discount pricing and nationalized FOB pricing.
        
        Args:
            df: DataFrame with matched products
            
        Returns:
            DataFrame with calculated margins and comparisons
        """
        logger.info("Calculating margins and comparisons")
        
        result = df.copy()
        
        # 1. Calculate discounted Dialfa price (apply to ALL products if discount > 0)
        if self.discount_percent > 0:
            result['precio_con_descuento'] = result.apply(
                lambda row: row['dialfa_precio'] * (1 - self.discount_percent / 100) 
                if pd.notna(row['dialfa_precio']) 
                else None,
                axis=1
            )
        else:
            result['precio_con_descuento'] = None
        
        # 2. Calculate nationalized Citizen FOB prices for min, max, and ponderado
        result['precio_fob_min_nacionalizado'] = result.apply(
            lambda row: row['precio_fob_min'] * (1 + self.nationalization_percent / 100)
            if pd.notna(row['precio_fob_min'])
            else None,
            axis=1
        )
        
        result['precio_fob_max_nacionalizado'] = result.apply(
            lambda row: row['precio_fob_max'] * (1 + self.nationalization_percent / 100)
            if pd.notna(row['precio_fob_max'])
            else None,
            axis=1
        )
        
        result['citizen_precio_nacionalizado'] = result.apply(
            lambda row: row['precio_fob_ponderado'] * (1 + self.nationalization_percent / 100)
            if pd.notna(row['precio_fob_ponderado'])
            else None,
            axis=1
        )
        
        # 3. Determine which prices to use for margin calculations
        # Use discounted price if available, otherwise regular price
        result['dialfa_precio_final'] = result.apply(
            lambda row: row['precio_con_descuento'] 
            if pd.notna(row.get('precio_con_descuento')) 
            else row['dialfa_precio'],
            axis=1
        )
        
        # Use nationalized Citizen price for margin calculations
        result['citizen_precio_final'] = result['citizen_precio_nacionalizado']
        
        # 4. Calculate Margin % = (Dialfa_Final - Citizen_Nationalized) / Dialfa_Final * 100
        result['margin_percent'] = (
            (result['dialfa_precio_final'] - result['citizen_precio_final']) / 
            result['dialfa_precio_final'] * 100
        ).round(2)
        
        # 5. Calculate Markup % = (Dialfa_Final - Citizen_Nationalized) / Citizen_Nationalized * 100
        result['markup_percent'] = (
            (result['dialfa_precio_final'] - result['citizen_precio_final']) / 
            result['citizen_precio_final'] * 100
        ).round(2)
        
        # 6. Calculate difference with Cintolo (using final Dialfa price)
        result['diff_vs_cintolo'] = (
            result['dialfa_precio_final'] - result['cintolo_precio']
        ).round(2)
        
        result['diff_percent_cintolo'] = (
            (result['dialfa_precio_final'] - result['cintolo_precio']) / 
            result['cintolo_precio'] * 100
        ).round(2)
        
        # 7. Calculate difference with Zaloze (using final Dialfa price)
        result['diff_vs_zaloze'] = (
            result['dialfa_precio_final'] - result['zaloze_precio']
        ).round(2)
        
        result['diff_percent_zaloze'] = (
            (result['dialfa_precio_final'] - result['zaloze_precio']) / 
            result['zaloze_precio'] * 100
        ).round(2)
        
        # 8. Determine market position (using final Dialfa price)
        result['market_position'] = result.apply(self._calculate_position, axis=1)
        
        # 9. Add match status flag
        result['match_status'] = result.apply(self._get_match_status, axis=1)
        
        # 10. Add margin category for color coding
        result['margin_category'] = result['margin_percent'].apply(self._categorize_margin)
        
        logger.info("Calculations completed")
        return result
    
    def _calculate_position(self, row):
        """Calculate market position (cheapest/middle/expensive) using final prices."""
        prices = []
        
        # Use final Dialfa price (discounted if applicable)
        if pd.notna(row.get('dialfa_precio_final')):
            prices.append(('Dialfa', row['dialfa_precio_final']))
        if pd.notna(row.get('cintolo_precio')):
            prices.append(('Cintolo', row['cintolo_precio']))
        if pd.notna(row.get('zaloze_precio')):
            prices.append(('Zaloze', row['zaloze_precio']))
        
        if len(prices) < 2:
            return 'N/A'
        
        prices.sort(key=lambda x: x[1])
        
        dialfa_price = row.get('dialfa_precio_final')
        if pd.notna(dialfa_price):
            if prices[0][0] == 'Dialfa':
                return 'Cheapest'
            elif prices[-1][0] == 'Dialfa':
                return 'Most Expensive'
            else:
                return 'Middle'
        
        return 'N/A'
    
    def _get_match_status(self, row):
        """Get match status for the product."""
        if pd.notna(row.get('precio_fob_ponderado')):
            if 'match_type' in row and pd.notna(row.get('match_type')):
                return row['match_type']
            return 'Exact Match'
        return 'Unmatched'
    
    def _categorize_margin(self, margin):
        """Categorize margin for color coding."""
        if pd.isna(margin):
            return 'unknown'
        elif margin < 20:
            return 'low'  # Red
        elif margin < 40:
            return 'medium'  # Yellow
        else:
            return 'high'  # Green
    
    def generate_report(self):
        """
        Generate complete price comparison report.
        Returns DataFrame with all calculations.
        """
        matched_df = self.match_products()
        final_df = self.calculate_margins(matched_df)
        
        # Reorder columns for better readability
        column_order = [
            'codigo', 'descripcion', 
            'dialfa_precio', 'precio_con_descuento',
            'citizen_producto', 
            'precio_fob_min', 'precio_fob_max', 'precio_fob_ponderado',
            'precio_fob_min_nacionalizado', 'precio_fob_max_nacionalizado', 'citizen_precio_nacionalizado',
            'margin_percent', 'markup_percent',
            'cintolo_precio', 'diff_vs_cintolo', 'diff_percent_cintolo',
            'zaloze_precio', 'diff_vs_zaloze', 'diff_percent_zaloze',
            'market_position', 'match_status', 'margin_category',
            'tipo', 'serie', 'tipo_serie', 'espesor', 'size', 'proveedor', 'matching_key'
        ]
        
        # Only include columns that exist
        column_order = [col for col in column_order if col in final_df.columns]
        final_df = final_df[column_order]
        
        return final_df

