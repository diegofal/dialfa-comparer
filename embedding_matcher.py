"""
Embedding-based product matching using OpenAI's text-embedding-3-small.
Optimized for cost efficiency: generates embeddings once and uses local cosine similarity.
"""
import logging
import os
import numpy as np
from openai import OpenAI
from typing import List, Dict, Tuple
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingMatcher:
    def __init__(self, api_key: str = None, cache_manager=None):
        """
        Initialize the embedding matcher with OpenAI client.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            cache_manager: DataCache instance for caching embeddings
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key parameter.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = "text-embedding-3-small"
        self.embeddings_cache = {}
        self.cache_manager = cache_manager
        
        logger.info(f"Initialized EmbeddingMatcher with model: {self.model}")
        if cache_manager:
            logger.info("Embeddings caching enabled")
    
    def _create_product_text(self, row: pd.Series) -> str:
        """
        Create a rich text representation of a product for embedding.
        Combines multiple fields to capture semantic meaning.
        
        Args:
            row: Product row from DataFrame
            
        Returns:
            Text representation of the product
        """
        parts = []
        
        # Add description (most important)
        if pd.notna(row.get('descripcion')):
            parts.append(str(row['descripcion']))
        
        # Add tipo_serie
        if pd.notna(row.get('tipo_serie')):
            parts.append(f"Tipo: {row['tipo_serie']}")
        
        # Add espesor
        if pd.notna(row.get('espesor')):
            parts.append(f"Espesor: {row['espesor']}")
        
        # Add size
        if pd.notna(row.get('size')):
            parts.append(f"Tamaño: {row['size']}")
        
        # Add codigo if available
        if pd.notna(row.get('codigo')):
            parts.append(f"Código: {row['codigo']}")
        
        # Add part_number for Citizen products
        if pd.notna(row.get('part_number')):
            parts.append(f"Part#: {row['part_number']}")
        
        return " | ".join(parts)
    
    def generate_embeddings_batch(self, texts: List[str], cache_key: str = None) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.
        Uses cache if available and cache_manager is provided.
        
        Args:
            texts: List of text strings to embed
            cache_key: Key for caching (e.g., 'dialfa', 'citizen')
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Try to load from cache first
        if cache_key and self.cache_manager:
            cached_embeddings = self.cache_manager.load_embeddings(cache_key)
            if cached_embeddings is not None and len(cached_embeddings) == len(texts):
                logger.info(f"Using cached embeddings for {cache_key} (saved API cost!)")
                # Convert to list if it's a numpy array, otherwise return as-is
                if hasattr(cached_embeddings, 'tolist'):
                    return cached_embeddings.tolist()
                return cached_embeddings
        
        try:
            # OpenAI API allows up to 2048 texts per batch
            # For cost optimization, we process all at once if possible
            logger.info(f"Generating embeddings for {len(texts)} texts...")
            
            response = self.client.embeddings.create(
                model=self.model,
                input=texts,
                encoding_format="float"
            )
            
            embeddings = [item.embedding for item in response.data]
            
            # Log token usage for cost tracking
            total_tokens = response.usage.total_tokens
            cost = (total_tokens / 1_000_000) * 0.02  # $0.02 per 1M tokens
            logger.info(f"Generated {len(embeddings)} embeddings. Tokens: {total_tokens}, Cost: ${cost:.6f}")
            
            # Save to cache if cache_manager is available
            if cache_key and self.cache_manager:
                import numpy as np
                embeddings_array = np.array(embeddings)
                self.cache_manager.save_embeddings(cache_key, embeddings_array)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        This is a local operation (no API cost).
        
        Args:
            vec1: First embedding vector
            vec2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def _normalize_field(self, value, field_type='general'):
        """
        Normalize field values for strict comparison.
        
        Args:
            value: Field value to normalize
            field_type: Type of field ('tipo_serie', 'size', 'espesor', 'general')
        
        Returns:
            Normalized value as string
        """
        import re
        
        if pd.isna(value) or value is None:
            return ''
        
        value_str = str(value).strip().upper()
        
        if field_type == 'tipo_serie':
            # Map English to Spanish nomenclature and vice versa for matching
            # This allows products with different nomenclatures to match
            
            # Remove extra spaces and standardize
            value_str = re.sub(r'\s+', ' ', value_str).strip()
            # Remove trailing .0 from series numbers (e.g., "2000.0" -> "2000")
            value_str = re.sub(r'\.0+$', '', value_str)
            
            # Normalize common product types to a canonical form
            # 90° Long Radius Elbows
            if any(x in value_str for x in ['90D LR ELBOW', '90 LR ELBOW', 'CODO RADIO LARGO 90', 'CODOS 90° RADIO LARGO', 'CODO R.L. 90']):
                return 'ELBOW_90_LR'
            
            # 90° Short Radius Elbows
            if any(x in value_str for x in ['90D SR ELBOW', '90 SR ELBOW', 'CODO RADIO CORTO 90', 'CODOS 90°', 'CODO 90']):
                # Check if it's explicitly long radius
                if 'LARGO' not in value_str and 'LR' not in value_str and 'LONG' not in value_str:
                    return 'ELBOW_90_SR'
            
            # 45° Elbows
            if any(x in value_str for x in ['45D LR ELBOW', '45 LR ELBOW', '45D ELBOW', 'CODO 45', 'CODOS 45', 
                                              'CODO RADIO LARGO 45', 'CODO RADIO LARGO A 45']):
                return 'ELBOW_45'
            
            # 180° Elbows / Returns
            if any(x in value_str for x in ['180', 'CODO 180', 'CR. LARGO Y CORTO 180', 'RETURN']):
                return 'ELBOW_180'
            
            # Tees - need to distinguish between normal tees and reducing tees
            if value_str in ['TEE', 'TE', 'T']:
                # For embedding matcher, we can't check the row description
                # so we return 'TEE' and rely on embedding similarity
                # But the description field will be part of the embedding
                return 'TEE'
            
            # Caps / Casquetes
            if any(x in value_str for x in ['CAP', 'CAS', 'CASQUETE', 'SEMIELIPTICO']):
                return 'CAP'
            
            # Reducers - all types
            if any(x in value_str for x in ['RED.', 'REDUCER', 'REDUCCION', 'CON. RED', 'EXC. RED', 'RED. TEE', 'TE RED']):
                if 'TEE' in value_str or 'TE RED' in value_str:
                    return 'REDUCER_TEE'
                elif 'CON' in value_str or 'CONCENTRIC' in value_str or 'CONCENTRICA' in value_str:
                    return 'REDUCER_CONCENTRIC'
                elif 'EXC' in value_str or 'ECCENTRIC' in value_str or 'EXCENTRICA' in value_str:
                    return 'REDUCER_ECCENTRIC'
                return 'REDUCER'
            
            # Caps
            if 'CAP' in value_str or 'CAS' in value_str:
                return 'CAP'
            
            # Crosses
            if 'CRUZ' in value_str or 'CROSS' in value_str:
                return 'CROSS'
            
            # Nipples
            if 'NIPPLE' in value_str:
                if 'NPT' in value_str:
                    return 'NIPPLE_NPT'
                elif 'BSPT' in value_str:
                    return 'NIPPLE_BSPT'
                return 'NIPPLE'
            
            # Flanges
            if 'W.N.R.F' in value_str or 'WELD NECK' in value_str or 'BRIDA' in value_str:
                return 'FLANGE_WELD_NECK'
            
            # Blind flanges
            if 'BLIND' in value_str:
                return 'FLANGE_BLIND'
            
            # Return as-is if no match (minimal normalization)
            return value_str
        
        elif field_type == 'size':
            # Normalize size: remove quotes and standardize format
            value_str = value_str.replace('"', '').replace("'", '').strip()
            # Normalize spaces in fractions: "1 1/2" should match "1.1/2" and "11/2"
            value_str = re.sub(r'\.(\d+/)', r' \1', value_str)  # "1.1/2" -> "1 1/2"
            value_str = re.sub(r'(\d)(\d/)', r'\1 \2', value_str)  # "11/2" -> "1 1/2"
            value_str = re.sub(r'\s+', ' ', value_str).strip()  # Multiple spaces to single
            return value_str
        
        elif field_type == 'espesor':
            # Normalize espesor
            if value_str in ['STANDARD', 'STD', 'ST']:
                return 'STD'
            elif value_str in ['EXTRA PESADO', 'XS', 'EX PESADO', 'EXTRA STRONG']:
                return 'XS'
            elif value_str in ['XXS', 'DOBLE EXTRA PESADO']:
                return 'XXS'
            return value_str
        
        return value_str
    
    def _fields_match(self, source_row, target_row, check_tipo_serie=True):
        """
        Check if critical fields (tipo_serie, size, espesor) match between source and target.
        This enforces business rules before embedding similarity.
        
        Args:
            source_row: Source product row
            target_row: Target product row
            check_tipo_serie: If True, enforces tipo_serie match. If False, only checks size and espesor.
        
        Returns:
            True if all critical fields match, False otherwise
        """
        # Normalize and compare tipo_serie (if required)
        if check_tipo_serie:
            source_tipo = self._normalize_field(source_row.get('tipo_serie'), 'tipo_serie')
            target_tipo = self._normalize_field(target_row.get('tipo_serie'), 'tipo_serie')
            
            if source_tipo != target_tipo:
                return False
        
        # Normalize and compare size
        source_size = self._normalize_field(source_row.get('size'), 'size')
        target_size = self._normalize_field(target_row.get('size'), 'size')
        
        if source_size != target_size:
            return False
        
        # Normalize and compare espesor
        source_espesor = self._normalize_field(source_row.get('espesor'), 'espesor')
        target_espesor = self._normalize_field(target_row.get('espesor'), 'espesor')
        
        if source_espesor != target_espesor:
            return False
        
        return True
    
    def match_products(
        self, 
        source_df: pd.DataFrame, 
        target_df: pd.DataFrame,
        source_name: str = "Source",
        target_name: str = "Target",
        threshold: float = 0.85,
        enforce_business_rules: bool = True,
        partial_business_rules: bool = False
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Match products from source to target using embeddings WITH or WITHOUT business rules.
        
        Args:
            source_df: Source DataFrame (e.g., Dialfa products)
            target_df: Target DataFrame (e.g., Citizen products)
            source_name: Name for source dataset (for logging)
            target_name: Name for target dataset (for logging)
            threshold: Minimum similarity score to consider a match (0-1)
            enforce_business_rules: If True, enforces strict matching rules:
                - tipo_serie must match exactly (normalized)
                - size must match exactly (normalized)
                - espesor must match exactly (normalized)
                If False, uses only embedding similarity (useful for competitors with different nomenclature)
            partial_business_rules: If True, enforces ONLY size and espesor matching, allows tipo_serie to vary
                (useful for competitors with different nomenclature for product types)
            
        Returns:
            Tuple of (matched_df, stats_dict)
        """
        logger.info(f"Starting {'rule-based ' if enforce_business_rules else ''}embedding matching: {source_name} ({len(source_df)}) -> {target_name} ({len(target_df)})")
        if enforce_business_rules:
            logger.info("Business rules: tipo_serie, size, and espesor must match exactly")
        elif partial_business_rules:
            logger.info("Partial business rules: size and espesor must match, tipo_serie uses semantic matching")
        else:
            logger.info("Business rules: DISABLED - using pure semantic matching")
        
        if target_df.empty:
            logger.warning(f"Target dataset {target_name} is empty. Skipping matching.")
            return source_df, {'matched': 0, 'unmatched': len(source_df)}
        
        # Initialize result columns
        result_df = source_df.copy()
        
        if target_name.lower() == 'citizen':
            result_df['citizen_descripcion_emb'] = None
            result_df['citizen_size_emb'] = None
            result_df['citizen_espesor_emb'] = None
            result_df['citizen_precio_usd_emb'] = None
            result_df['citizen_match_score'] = None
        elif target_name.lower() == 'cintolo':
            result_df['cintolo_codigo_emb'] = None
            result_df['cintolo_descripcion_emb'] = None
            result_df['cintolo_precio_emb'] = None
            result_df['cintolo_match_score'] = None
        elif target_name.lower() == 'zaloze':
            result_df['zaloze_codigo_emb'] = None
            result_df['zaloze_descripcion_emb'] = None
            result_df['zaloze_precio_emb'] = None
            result_df['zaloze_match_score'] = None
        
        matched_count = 0
        unmatched_count = 0
        rule_filtered_count = 0
        
        # Step 1: Pre-generate all embeddings (more efficient than one-by-one)
        logger.info(f"Generating embeddings for {target_name} products...")
        target_texts = [self._create_product_text(row) for _, row in target_df.iterrows()]
        target_embeddings = self.generate_embeddings_batch(target_texts, cache_key=target_name.lower())
        target_embeddings_np = np.array(target_embeddings)
        
        logger.info(f"Generating embeddings for {source_name} products...")
        source_texts = [self._create_product_text(row) for _, row in source_df.iterrows()]
        source_embeddings = self.generate_embeddings_batch(source_texts, cache_key=source_name.lower())
        source_embeddings_np = np.array(source_embeddings)
        
        # Step 2: Calculate full similarity matrix
        logger.info("Calculating similarity matrix...")
        similarity_matrix = np.dot(source_embeddings_np, target_embeddings_np.T)
        source_norms = np.linalg.norm(source_embeddings_np, axis=1, keepdims=True)
        target_norms = np.linalg.norm(target_embeddings_np, axis=1, keepdims=True)
        similarity_matrix = similarity_matrix / (source_norms @ target_norms.T)
        
        # Step 3: Process each source product with or without business rules
        if enforce_business_rules or partial_business_rules:
            logger.info("Applying business rules to filter matches...")
        else:
            logger.info("Matching based purely on semantic similarity...")
            
        for i, source_row in enumerate(source_df.itertuples()):
            source_dict = source_df.iloc[i].to_dict()
            
            if enforce_business_rules or partial_business_rules:
                # Find all target products that match the business rules
                # If partial_business_rules, don't check tipo_serie (only size and espesor)
                check_tipo = not partial_business_rules
                valid_target_mask = []
                for j, target_row in target_df.iterrows():
                    valid_target_mask.append(self._fields_match(source_dict, target_row, check_tipo_serie=check_tipo))
                
                valid_target_mask = np.array(valid_target_mask)
                
                if not valid_target_mask.any():
                    # No products match the business rules
                    unmatched_count += 1
                    rule_filtered_count += 1
                    continue
                
                # Get similarities only for valid targets
                valid_similarities = similarity_matrix[i].copy()
                valid_similarities[~valid_target_mask] = -1  # Mask out invalid matches
            else:
                # Use all similarities without business rule filtering
                valid_similarities = similarity_matrix[i]
            
            best_idx = valid_similarities.argmax()
            best_score = valid_similarities[best_idx]
            
            if best_score >= threshold:
                matched_count += 1
                best_match = target_df.iloc[best_idx]
                
                # Store match
                if target_name.lower() == 'citizen':
                    result_df.loc[source_row.Index, 'citizen_descripcion_emb'] = best_match.get('descripcion')
                    result_df.loc[source_row.Index, 'citizen_size_emb'] = best_match.get('size')
                    result_df.loc[source_row.Index, 'citizen_espesor_emb'] = best_match.get('espesor')
                    result_df.loc[source_row.Index, 'citizen_precio_usd_emb'] = best_match.get('precio_usd')
                    result_df.loc[source_row.Index, 'citizen_match_score'] = round(best_score * 100, 2)
                    
                    logger.debug(f"✓ Match: {source_dict.get('descripcion', '')[:40]} -> {best_match['descripcion'][:40]} (score: {best_score:.3f})")
                
                elif target_name.lower() == 'cintolo':
                    result_df.loc[source_row.Index, 'cintolo_codigo_emb'] = best_match.get('codigo')
                    result_df.loc[source_row.Index, 'cintolo_descripcion_emb'] = best_match.get('descripcion')
                    result_df.loc[source_row.Index, 'cintolo_precio_emb'] = best_match.get('precio')
                    result_df.loc[source_row.Index, 'cintolo_match_score'] = round(best_score * 100, 2)
                
                elif target_name.lower() == 'zaloze':
                    result_df.loc[source_row.Index, 'zaloze_codigo_emb'] = best_match.get('codigo')
                    result_df.loc[source_row.Index, 'zaloze_descripcion_emb'] = best_match.get('descripcion')
                    result_df.loc[source_row.Index, 'zaloze_precio_emb'] = best_match.get('precio')
                    result_df.loc[source_row.Index, 'zaloze_match_score'] = round(best_score * 100, 2)
            else:
                unmatched_count += 1
        
        logger.info(f"✓ Matching complete:")
        logger.info(f"  - Matched: {matched_count}")
        if enforce_business_rules:
            logger.info(f"  - Unmatched (no business rule match): {rule_filtered_count}")
            logger.info(f"  - Unmatched (low similarity): {unmatched_count - rule_filtered_count}")
        else:
            logger.info(f"  - Unmatched (low similarity): {unmatched_count}")
        logger.info(f"  - Total unmatched: {unmatched_count}")
        
        stats = {
            'matched': matched_count,
            'unmatched': unmatched_count,
            'rule_filtered': rule_filtered_count if enforce_business_rules else 0,
            'match_rate': round((matched_count / len(source_df)) * 100, 2) if len(source_df) > 0 else 0,
            'threshold': threshold
        }
        
        return result_df, stats

