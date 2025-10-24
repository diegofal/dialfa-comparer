"""
Data cache manager for storing and loading extracted data.
Saves expensive operations (OCR, database queries, AI embeddings) to files.
"""
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCache:
    def __init__(self, cache_dir='data_cache'):
        """
        Initialize data cache manager.
        
        Args:
            cache_dir: Directory to store cached CSV files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Cache file names
        self.files = {
            'dialfa': self.cache_dir / 'dialfa_products.csv',
            'db_articulos': self.cache_dir / 'db_articulos.csv',
            'citizen': self.cache_dir / 'citizen_products.csv',
            'cintolo': self.cache_dir / 'cintolo_products.csv',
            'zaloze': self.cache_dir / 'zaloze_products.csv',
            'metadata': self.cache_dir / 'cache_metadata.csv',
            'embeddings_dialfa': self.cache_dir / 'embeddings_dialfa.pkl',
            'embeddings_citizen': self.cache_dir / 'embeddings_citizen.pkl',
            'embeddings_cintolo': self.cache_dir / 'embeddings_cintolo.pkl',
            'embeddings_zaloze': self.cache_dir / 'embeddings_zaloze.pkl'
        }
        
        logger.info(f"Cache directory: {self.cache_dir.absolute()}")
    
    def cache_exists(self, source_name):
        """Check if cache file exists for a data source."""
        cache_file = self.files.get(source_name)
        if cache_file and cache_file.exists():
            return True
        return False
    
    def all_caches_exist(self):
        """Check if all required cache files exist."""
        required = ['db_articulos', 'citizen', 'cintolo', 'zaloze']
        return all(self.cache_exists(source) for source in required)
    
    def save(self, source_name, df, metadata=None):
        """
        Save DataFrame to cache.
        
        Args:
            source_name: Name of data source (dialfa, citizen, etc.)
            df: DataFrame to cache
            metadata: Optional dict with extraction metadata
        """
        cache_file = self.files.get(source_name)
        if not cache_file:
            logger.warning(f"Unknown source name: {source_name}")
            return
        
        try:
            df.to_csv(cache_file, index=False, encoding='utf-8')
            logger.info(f"✓ Cached {len(df)} records to {cache_file.name}")
            
            # Save metadata
            if metadata or True:  # Always save timestamp
                self._update_metadata(source_name, len(df), metadata)
                
        except Exception as e:
            logger.error(f"Failed to cache {source_name}: {e}")
    
    def load(self, source_name):
        """
        Load DataFrame from cache.
        
        Args:
            source_name: Name of data source
            
        Returns:
            DataFrame or None if not found
        """
        cache_file = self.files.get(source_name)
        if not cache_file or not cache_file.exists():
            return None
        
        try:
            df = pd.read_csv(cache_file, encoding='utf-8')
            logger.info(f"✓ Loaded {len(df)} records from {cache_file.name}")
            return df
        except Exception as e:
            logger.error(f"Failed to load cache {source_name}: {e}")
            return None
    
    def _update_metadata(self, source_name, record_count, extra_metadata=None):
        """Update cache metadata file."""
        metadata_file = self.files['metadata']
        
        # Load existing metadata
        if metadata_file.exists():
            try:
                metadata_df = pd.read_csv(metadata_file)
            except:
                metadata_df = pd.DataFrame()
        else:
            metadata_df = pd.DataFrame()
        
        # Create new metadata record
        new_record = {
            'source': source_name,
            'timestamp': datetime.now().isoformat(),
            'record_count': record_count
        }
        
        if extra_metadata:
            new_record.update(extra_metadata)
        
        # Remove old entry for this source
        metadata_df = metadata_df[metadata_df['source'] != source_name]
        
        # Add new entry
        metadata_df = pd.concat([metadata_df, pd.DataFrame([new_record])], ignore_index=True)
        
        # Save
        metadata_df.to_csv(metadata_file, index=False)
    
    def get_metadata(self):
        """Get cache metadata as DataFrame."""
        metadata_file = self.files['metadata']
        if metadata_file.exists():
            try:
                return pd.read_csv(metadata_file)
            except:
                return pd.DataFrame()
        return pd.DataFrame()
    
    def clear_cache(self, source_name=None):
        """
        Clear cache files.
        
        Args:
            source_name: Specific source to clear, or None to clear all
        """
        if source_name:
            cache_file = self.files.get(source_name)
            if cache_file and cache_file.exists():
                cache_file.unlink()
                logger.info(f"Cleared cache: {source_name}")
        else:
            # Clear all cache files
            for name, cache_file in self.files.items():
                if cache_file.exists():
                    cache_file.unlink()
            logger.info("Cleared all cache files")
    
    def save_embeddings(self, source_name, embeddings_array):
        """
        Save embeddings array to cache (as pickle for numpy arrays).
        
        Args:
            source_name: Name of data source (dialfa, citizen, etc.)
            embeddings_array: Numpy array of embeddings
        """
        cache_key = f'embeddings_{source_name}'
        cache_file = self.files.get(cache_key)
        
        if not cache_file:
            logger.warning(f"Unknown embeddings source: {source_name}")
            return
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embeddings_array, f)
            
            logger.info(f"✓ Cached {len(embeddings_array)} embeddings to {cache_file.name}")
            
            # Save metadata
            self._update_metadata(cache_key, len(embeddings_array), {'type': 'embeddings'})
            
        except Exception as e:
            logger.error(f"Failed to cache embeddings for {source_name}: {e}")
    
    def load_embeddings(self, source_name):
        """
        Load embeddings array from cache.
        
        Args:
            source_name: Name of data source
            
        Returns:
            Numpy array or None if not found
        """
        cache_key = f'embeddings_{source_name}'
        cache_file = self.files.get(cache_key)
        
        if not cache_file or not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                embeddings = pickle.load(f)
            
            logger.info(f"✓ Loaded {len(embeddings)} embeddings from {cache_file.name}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to load embeddings for {source_name}: {e}")
            return None
    
    def embeddings_exist(self, source_name):
        """Check if embeddings cache exists for a data source."""
        cache_key = f'embeddings_{source_name}'
        cache_file = self.files.get(cache_key)
        return cache_file and cache_file.exists()
    
    def print_status(self):
        """Print cache status."""
        print("\n" + "="*70)
        print("CACHE STATUS")
        print("="*70)
        
        metadata = self.get_metadata()
        
        for source in ['db_articulos', 'citizen', 'cintolo', 'zaloze']:
            exists = self.cache_exists(source)
            status = "[OK] CACHED" if exists else "[X] NOT CACHED"
            
            if exists and not metadata.empty and source in metadata['source'].values:
                row = metadata[metadata['source'] == source].iloc[0]
                timestamp = row['timestamp']
                count = row['record_count']
                print(f"{source:15} {status:15} | {count:4} records | {timestamp}")
            else:
                print(f"{source:15} {status:15}")
        
        print("="*70)

