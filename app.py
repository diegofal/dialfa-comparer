"""
Flask web application for price comparison system.
Displays comprehensive price comparison between Dialfa, Citizen, and competitors.
"""
from flask import Flask, render_template, jsonify, request
import pandas as pd
import logging
import sys
import traceback
from dotenv import load_dotenv
import argparse
import os

# Load environment variables from .env file
load_dotenv()

from database.db_connector import DatabaseConnector
from data_extractors.dialfa_extractor import DialfaExtractor
from data_extractors.citizen_extractor import CitizenExtractor
from data_extractors.cintolo_extractor import CintoloExtractor
from data_extractors.zaloze_extractor import ZalozeExtractor
from price_calculator import PriceCalculator
from data_cache import DataCache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables
cached_results = None
extraction_errors = []
data_cache = DataCache()
last_params = {}  # Track last used parameters to invalidate cache when they change

# Check for force refresh from environment variable (survives Flask reload)
force_refresh = os.environ.get('FORCE_REFRESH', 'false').lower() == 'true'


def load_and_process_data(use_cache=True, discount_percent=30, nationalization_percent=150):
    """
    Load data from all sources and process comparisons.
    
    Args:
        use_cache: If True, use cached CSV files when available
        discount_percent: Discount percentage to apply to ALL products (default: 30%)
        nationalization_percent: Nationalization percentage for FOB prices
    """
    global cached_results, extraction_errors
    
    extraction_errors = []
    
    try:
        logger.info("=" * 80)
        logger.info("Starting data extraction and processing")
        if use_cache:
            logger.info("Cache mode: ENABLED (use --refresh to force regeneration)")
        else:
            logger.info("Cache mode: DISABLED (regenerating all data)")
        logger.info("=" * 80)
        
        # Print cache status
        if use_cache:
            data_cache.print_status()
        
        # [1/4] Load database products
        logger.info("\n[1/4] Loading products from database (PRIMARY SOURCE)...")
        if use_cache and data_cache.cache_exists('db_articulos'):
            db_articulos = data_cache.load('db_articulos')
            logger.info(f"✓ Loaded from cache: {len(db_articulos)} products")
        else:
            try:
                db_connector = DatabaseConnector()
                db_articulos = db_connector.get_articulos()
                logger.info(f"✓ Extracted from database: {len(db_articulos)} products")
                data_cache.save('db_articulos', db_articulos)
            except Exception as e:
                error_msg = f"CRITICAL: Database connection failed: {str(e)}"
                logger.error(error_msg)
                extraction_errors.append(error_msg)
                return pd.DataFrame()
        
        # [2/4] Extract Citizen supplier data
        logger.info("\n[2/4] Loading Citizen supplier data...")
        if use_cache and data_cache.cache_exists('citizen'):
            citizen_data = data_cache.load('citizen')
            logger.info(f"✓ Loaded from cache: {len(citizen_data)} products")
        else:
            try:
                citizen_extractor = CitizenExtractor()
                citizen_data = citizen_extractor.extract()
                logger.info(f"✓ Extracted: {len(citizen_data)} products")
                data_cache.save('citizen', citizen_data)
            except Exception as e:
                error_msg = f"Citizen extraction failed: {str(e)}"
                logger.error(error_msg)
                extraction_errors.append(error_msg)
                citizen_data = pd.DataFrame()
        
        # [3/4] Extract Cintolo competitor data
        logger.info("\n[3/4] Loading Cintolo competitor data...")
        if use_cache and data_cache.cache_exists('cintolo'):
            cintolo_data = data_cache.load('cintolo')
            logger.info(f"✓ Loaded from cache: {len(cintolo_data)} products")
        else:
            try:
                cintolo_extractor = CintoloExtractor()
                cintolo_data = cintolo_extractor.extract()
                logger.info(f"✓ Extracted: {len(cintolo_data)} products")
                data_cache.save('cintolo', cintolo_data)
            except Exception as e:
                error_msg = f"Cintolo extraction failed: {str(e)}"
                logger.error(error_msg)
                extraction_errors.append(error_msg)
                cintolo_data = pd.DataFrame()
        
        # [4/4] Extract Zaloze competitor data (OCR - very slow!)
        logger.info("\n[4/4] Loading Zaloze competitor data...")
        if use_cache and data_cache.cache_exists('zaloze'):
            zaloze_data = data_cache.load('zaloze')
            logger.info(f"✓ Loaded from cache: {len(zaloze_data)} products (OCR skipped!)")
        else:
            try:
                logger.info("⚠️  Running OCR on images (this may take 1-2 minutes)...")
                zaloze_extractor = ZalozeExtractor()
                zaloze_data = zaloze_extractor.extract()
                logger.info(f"✓ Extracted: {len(zaloze_data)} products")
                data_cache.save('zaloze', zaloze_data)
            except Exception as e:
                error_msg = f"Zaloze extraction failed: {str(e)}"
                logger.error(error_msg)
                extraction_errors.append(error_msg)
                zaloze_data = pd.DataFrame()
        
        # Calculate price comparisons
        logger.info("\nCalculating price comparisons...")
        calculator = PriceCalculator(
            db_articulos=db_articulos,  # PRIMARY SOURCE
            dialfa_data=db_articulos,   # Use same data for now
            citizen_data=citizen_data,
            cintolo_data=cintolo_data,
            zaloze_data=zaloze_data,
            use_embeddings=True,  # Enable AI-powered matching
            cache_manager=data_cache,  # Pass cache manager for embeddings
            discount_percent=discount_percent,
            nationalization_percent=nationalization_percent
        )
        
        results = calculator.generate_report()
        
        logger.info(f"\n✓ Generated report with {len(results)} products")
        logger.info("=" * 80)
        
        # Replace NaN with None for JSON serialization
        results = results.where(pd.notna(results), None)
        
        cached_results = results
        return results
        
    except Exception as e:
        error_msg = f"Critical error in data processing: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        extraction_errors.append(error_msg)
        return pd.DataFrame()


@app.route('/')
def index():
    """Render main dashboard page."""
    try:
        # Load data if not already cached (use cache by default)
        if cached_results is None:
            load_and_process_data(use_cache=not force_refresh)
        
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering index: {e}")
        return f"Error: {str(e)}", 500


@app.route('/api/data')
def get_data():
    """API endpoint to get comparison data as JSON."""
    global cached_results, last_params
    
    try:
        # Get pricing configuration from query parameters
        discount_percent = float(request.args.get('discount_percent', 30))
        nationalization_percent = float(request.args.get('nationalization_percent', 150))
        
        # Check if parameters changed - if so, invalidate cache
        current_params = {
            'discount_percent': discount_percent,
            'nationalization_percent': nationalization_percent
        }
        
        params_changed = last_params != current_params
        
        if params_changed:
            logger.info(f"Parameters changed: {last_params} → {current_params}")
            cached_results = None  # Invalidate cache
            last_params = current_params
        
        # Load data if not cached or parameters changed
        if cached_results is None or params_changed:
            load_and_process_data(
                use_cache=not force_refresh,
                discount_percent=discount_percent,
                nationalization_percent=nationalization_percent
            )
        
        if cached_results is None or cached_results.empty:
            return jsonify({
                'data': [],
                'errors': extraction_errors,
                'summary': {
                    'total_products': 0,
                    'matched_products': 0,
                    'average_margin': 0,
                    'average_markup': 0
                }
            })
        
        # Convert DataFrame to list of dictionaries
        # Convert numpy/pandas types to native Python types for JSON serialization
        data = cached_results.astype(object).where(pd.notna(cached_results), None).to_dict('records')
        
        # Calculate summary statistics and convert to native Python types
        summary = {
            'total_products': int(len(cached_results)),
            'matched_products': int(cached_results['precio_fob_ponderado'].notna().sum()),
            'unmatched_citizen': int(cached_results['precio_fob_ponderado'].isna().sum()),
            'matched_cintolo': int(cached_results['cintolo_precio'].notna().sum()),
            'unmatched_cintolo': int(cached_results['cintolo_precio'].isna().sum()),
            'matched_zaloze': int(cached_results['zaloze_precio'].notna().sum()),
            'unmatched_zaloze': int(cached_results['zaloze_precio'].isna().sum()),
            'average_margin': float(cached_results['margin_percent'].mean()) if 'margin_percent' in cached_results else 0.0,
            'average_markup': float(cached_results['markup_percent'].mean()) if 'markup_percent' in cached_results else 0.0,
            'low_margin_products': int((cached_results['margin_category'] == 'low').sum()) if 'margin_category' in cached_results else 0,
            'medium_margin_products': int((cached_results['margin_category'] == 'medium').sum()) if 'margin_category' in cached_results else 0,
            'high_margin_products': int((cached_results['margin_category'] == 'high').sum()) if 'margin_category' in cached_results else 0
        }
        
        return jsonify({
            'data': data,
            'errors': extraction_errors,
            'summary': summary
        })
        
    except Exception as e:
        logger.error(f"Error in get_data: {e}\n{traceback.format_exc()}")
        return jsonify({
            'error': str(e),
            'data': [],
            'errors': extraction_errors + [str(e)]
        }), 500


@app.route('/api/refresh')
def refresh_data():
    """API endpoint to force refresh data."""
    global cached_results
    cached_results = None
    load_and_process_data(use_cache=False)  # Force regeneration
    return jsonify({'status': 'success', 'message': 'Data refreshed'})


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Price Comparison System')
    parser.add_argument('--refresh', action='store_true',
                        help='Force refresh: regenerate all cached data (ignores CSV cache)')
    parser.add_argument('--clear-cache', action='store_true',
                        help='Clear all cached CSV files and exit')
    parser.add_argument('--status', action='store_true',
                        help='Show cache status and exit')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port to run Flask app (default: 5000)')
    parser.add_argument('--no-preload', action='store_true',
                        help='Do not load data on startup (load on first request instead)')
    
    args = parser.parse_args()
    
    # Handle cache management commands
    if args.clear_cache:
        logger.info("Clearing all cached data...")
        data_cache.clear_cache()
        logger.info("✓ Cache cleared successfully")
        sys.exit(0)
    
    if args.status:
        data_cache.print_status()
        sys.exit(0)
    
    # Set or clear environment variable for refresh mode
    if args.refresh:
        os.environ['FORCE_REFRESH'] = 'true'
        logger.info("⚠️  REFRESH MODE: Will regenerate all data (ignoring cache)")
    else:
        # Explicitly clear the flag if not in refresh mode
        os.environ['FORCE_REFRESH'] = 'false'
    
    # Load data on startup (unless --no-preload is specified)
    if not args.no_preload:
        logger.info("\n" + "="*80)
        logger.info("PRELOADING DATA ON STARTUP...")
        logger.info("="*80)
        load_and_process_data(use_cache=not force_refresh)
        logger.info("\n✓ Data preloaded successfully!")
        logger.info("="*80 + "\n")
    
    # Start Flask application
    logger.info("Starting Flask web server...")
    app.run(debug=True, host='0.0.0.0', port=args.port, use_reloader=False)  # Disable reloader to avoid double loading

