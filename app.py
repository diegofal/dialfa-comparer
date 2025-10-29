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
cached_matched_data = None  # Cache matched data (independent of pricing params)
extraction_errors = []
data_cache = DataCache()
last_params = {}  # Track last used parameters to invalidate cache when they change

# Check for force refresh from environment variable (survives Flask reload)
force_refresh = os.environ.get('FORCE_REFRESH', 'false').lower() == 'true'


def load_and_process_data(use_cache=True, discount_percent=30, nationalization_percent=150, 
                          cintolo_discount=20, zaloze_discount=20):
    """
    Load data from all sources and process comparisons.
    
    Args:
        use_cache: If True, use cached CSV files when available
        discount_percent: Discount percentage to apply to ALL products (default: 30%)
        nationalization_percent: Nationalization percentage for FOB prices
        cintolo_discount: Discount percentage for Cintolo prices (default: 20%)
        zaloze_discount: Discount percentage for Zaloze prices (default: 20%)
    """
    global cached_results, cached_matched_data, extraction_errors
    
    extraction_errors = []
    
    try:
        # Check if we can reuse matched data (if only pricing params changed)
        if cached_matched_data is not None:
            logger.info("✓ Reusing cached matched data, recalculating margins only...")
            calculator = PriceCalculator(
                db_articulos=cached_matched_data['db_articulos'],
                dialfa_data=cached_matched_data['db_articulos'],
                citizen_data=cached_matched_data['citizen_data'],
                cintolo_data=cached_matched_data['cintolo_data'],
                zaloze_data=cached_matched_data['zaloze_data'],
                use_embeddings=False,  # Skip embedding initialization
                discount_percent=discount_percent,
                nationalization_percent=nationalization_percent,
                cintolo_discount=cintolo_discount,
                zaloze_discount=zaloze_discount
            )
            
            # Use the pre-matched data
            results = calculator.calculate_margins(cached_matched_data['matched_df'])
            
            # Reorder columns for better readability
            column_order = [
                'codigo', 'descripcion', 
                'dialfa_precio', 'precio_con_descuento',
                'citizen_producto', 
                'precio_fob_min', 'precio_fob_max', 'precio_fob_ponderado',
                'precio_fob_min_nacionalizado', 'precio_fob_max_nacionalizado', 'citizen_precio_nacionalizado',
                'margin_percent', 'markup_percent',
                'cintolo_descripcion', 'cintolo_precio', 'diff_vs_cintolo', 'diff_percent_cintolo',
                'zaloze_descripcion', 'zaloze_precio', 'diff_vs_zaloze', 'diff_percent_zaloze',
                'market_position', 'match_status', 'margin_category',
                'tipo', 'serie', 'tipo_serie', 'espesor', 'size', 'proveedor', 'matching_key'
            ]
            
            # Only include columns that exist
            column_order = [col for col in column_order if col in results.columns]
            results = results[column_order]
            
            logger.info(f"\n✓ Recalculated margins for {len(results)} products")
            
            # Replace NaN with None for JSON serialization
            results = results.where(pd.notna(results), None)
            
            cached_results = results
            return results
        
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
            nationalization_percent=nationalization_percent,
            cintolo_discount=cintolo_discount,
            zaloze_discount=zaloze_discount
        )
        
        # First, do matching (expensive - only once)
        matched_df = calculator.match_products()
        
        # Cache the matched data for future recalculations (global variable)
        cached_matched_data = {
            'db_articulos': db_articulos,
            'citizen_data': citizen_data,
            'cintolo_data': cintolo_data,
            'zaloze_data': zaloze_data,
            'matched_df': matched_df
        }
        globals()['cached_matched_data'] = cached_matched_data  # Ensure it's global
        
        # Then calculate margins (cheap - can be redone with different params)
        results = calculator.calculate_margins(matched_df)
        
        # Reorder columns for better readability
        column_order = [
            'codigo', 'descripcion', 
            'dialfa_precio', 'precio_con_descuento',
            'citizen_producto', 
            'precio_fob_min', 'precio_fob_max', 'precio_fob_ponderado',
            'precio_fob_min_nacionalizado', 'precio_fob_max_nacionalizado', 'citizen_precio_nacionalizado',
            'margin_percent', 'markup_percent',
            'cintolo_descripcion', 'cintolo_precio', 'diff_vs_cintolo', 'diff_percent_cintolo',
            'zaloze_descripcion', 'zaloze_precio', 'diff_vs_zaloze', 'diff_percent_zaloze',
            'market_position', 'match_status', 'margin_category',
            'tipo', 'serie', 'tipo_serie', 'espesor', 'size', 'proveedor', 'matching_key'
        ]
        
        # Only include columns that exist
        column_order = [col for col in column_order if col in results.columns]
        results = results[column_order]
        
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
        cintolo_discount = float(request.args.get('cintolo_discount', 20))
        zaloze_discount = float(request.args.get('zaloze_discount', 20))
        
        # Check if parameters changed - if so, invalidate cache
        current_params = {
            'discount_percent': discount_percent,
            'nationalization_percent': nationalization_percent,
            'cintolo_discount': cintolo_discount,
            'zaloze_discount': zaloze_discount
        }
        
        params_changed = last_params != current_params
        
        if params_changed:
            logger.info(f"Pricing parameters changed: {last_params} → {current_params}")
            logger.info("Will recalculate margins only (reusing matched data)")
            cached_results = None  # Invalidate results cache only, NOT matched_data
            last_params = current_params
        
        # Load data if not cached or parameters changed
        # Note: load_and_process_data will reuse cached_matched_data if available
        if cached_results is None or params_changed:
            load_and_process_data(
                use_cache=not force_refresh,
                discount_percent=discount_percent,
                nationalization_percent=nationalization_percent,
                cintolo_discount=cintolo_discount,
                zaloze_discount=zaloze_discount
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
    global cached_results, cached_matched_data
    cached_results = None
    cached_matched_data = None  # Clear matched data too, forcing full re-matching
    load_and_process_data(use_cache=False)  # Force regeneration
    return jsonify({'status': 'success', 'message': 'Data refreshed'})


@app.route('/api/export/html', methods=['GET', 'POST'])
def export_html():
    """API endpoint to export data as a standalone HTML file."""
    from flask import make_response
    from datetime import datetime
    
    try:
        if request.method == 'POST':
            # Receive filtered data from frontend
            request_data = request.get_json()
            data = request_data.get('data', [])
            discount_percent = float(request_data.get('discount_percent', 30))
            nationalization_percent = float(request_data.get('nationalization_percent', 150))
            cintolo_discount = float(request_data.get('cintolo_discount', 20))
            zaloze_discount = float(request_data.get('zaloze_discount', 20))
            filters = request_data.get('filters', {})
            
            if not data:
                return "No data to export", 400
            
            # Calculate summary statistics from filtered data
            df = pd.DataFrame(data)
            summary = {
                'total_products': int(len(df)),
                'matched_products': int(df['precio_fob_ponderado'].notna().sum()) if 'precio_fob_ponderado' in df.columns else 0,
                'unmatched_citizen': int(df['precio_fob_ponderado'].isna().sum()) if 'precio_fob_ponderado' in df.columns else 0,
                'matched_cintolo': int(df['cintolo_precio'].notna().sum()) if 'cintolo_precio' in df.columns else 0,
                'unmatched_cintolo': int(df['cintolo_precio'].isna().sum()) if 'cintolo_precio' in df.columns else 0,
                'matched_zaloze': int(df['zaloze_precio'].notna().sum()) if 'zaloze_precio' in df.columns else 0,
                'unmatched_zaloze': int(df['zaloze_precio'].isna().sum()) if 'zaloze_precio' in df.columns else 0,
                'average_margin': float(df['margin_percent'].mean()) if 'margin_percent' in df.columns and len(df) > 0 else 0.0,
                'average_markup': float(df['markup_percent'].mean()) if 'markup_percent' in df.columns and len(df) > 0 else 0.0,
                'low_margin_products': int((df['margin_category'] == 'low').sum()) if 'margin_category' in df.columns else 0,
                'medium_margin_products': int((df['margin_category'] == 'medium').sum()) if 'margin_category' in df.columns else 0,
                'high_margin_products': int((df['margin_category'] == 'high').sum()) if 'margin_category' in df.columns else 0
            }
            
            # Add filter information to the export
            filter_text = []
            if filters.get('margin'):
                filter_labels = {'high': 'Alto (>40%)', 'medium': 'Medio (20-40%)', 'low': 'Bajo (<20%)'}
                filter_text.append(f"Margen: {filter_labels.get(filters['margin'], filters['margin'])}")
            if filters.get('match'):
                filter_labels = {'matched': 'Mapeados', 'unmatched': 'Sin Mapear'}
                filter_text.append(f"Estado: {filter_labels.get(filters['match'], filters['match'])}")
            if filters.get('position'):
                filter_text.append(f"Posición: {filters['position']}")
            
            filter_info = " | ".join(filter_text) if filter_text else "Sin filtros aplicados"
            
        else:
            # GET method - export all data (legacy support)
            discount_percent = float(request.args.get('discount_percent', 30))
            nationalization_percent = float(request.args.get('nationalization_percent', 150))
            cintolo_discount = float(request.args.get('cintolo_discount', 20))
            zaloze_discount = float(request.args.get('zaloze_discount', 20))
            
            # Ensure data is loaded with current parameters
            if cached_results is None:
                load_and_process_data(
                    use_cache=not force_refresh,
                    discount_percent=discount_percent,
                    nationalization_percent=nationalization_percent,
                    cintolo_discount=cintolo_discount,
                    zaloze_discount=zaloze_discount
                )
            
            if cached_results is None or cached_results.empty:
                return "No data available to export", 400
            
            # Convert DataFrame to list of dictionaries
            data = cached_results.astype(object).where(pd.notna(cached_results), None).to_dict('records')
            
            # Calculate summary statistics
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
            
            filter_info = "Todos los productos (sin filtros)"
        
        # Generate HTML content
        html_content = generate_export_html(data, summary, discount_percent, nationalization_percent, 
                                            cintolo_discount, zaloze_discount, filter_info)
        
        # Create response with proper headers for download
        response = make_response(html_content)
        response.headers['Content-Type'] = 'text/html; charset=utf-8'
        response.headers['Content-Disposition'] = f'attachment; filename=comparacion_precios_{datetime.now().strftime("%Y-%m-%d_%H-%M")}.html'
        
        return response
        
    except Exception as e:
        logger.error(f"Error in export_html: {e}\n{traceback.format_exc()}")
        return f"Error generating HTML export: {str(e)}", 500


def generate_export_html(data, summary, discount_percent, nationalization_percent, 
                         cintolo_discount, zaloze_discount, filter_info="Sin filtros aplicados"):
    """Generate a standalone HTML file with embedded styles and data."""
    from datetime import datetime
    
    # Helper function to format numbers
    def fmt(val):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return '-'
        try:
            return f"{float(val):.2f}"
        except (ValueError, TypeError):
            return str(val)
    
    # Generate table rows
    rows_html = []
    for row in data:
        margin_class = f"margin-{row.get('margin_category', '')}" if row.get('margin_category') else ""
        
        # Market position formatting
        position = row.get('market_position', 'N/A')
        position_class = 'position-cheapest' if position == 'Cheapest' else \
                        'position-expensive' if position == 'Most Expensive' else \
                        'position-middle' if position == 'Middle' else ''
        
        # Match status badge
        match_status = row.get('match_status', 'Unknown')
        badge_class = 'badge-danger' if match_status == 'Unmatched' else 'badge-success'
        
        rows_html.append(f'''
            <tr class="{margin_class}">
                <td>{row.get('codigo', '-')}</td>
                <td>{row.get('descripcion', '-')}</td>
                <td class="price">${fmt(row.get('dialfa_precio'))}</td>
                <td class="price">{'$' + fmt(row.get('precio_con_descuento')) if row.get('precio_con_descuento') else '-'}</td>
                <td>{row.get('citizen_producto', '-')}</td>
                <td class="price" style="text-align: center;">{str(round(row.get('cantidad'))) + ' uds' if row.get('cantidad') else '-'}</td>
                <td class="price">${fmt(row.get('precio_fob_min'))}</td>
                <td class="price">${fmt(row.get('precio_fob_max'))}</td>
                <td class="price">${fmt(row.get('precio_fob_ponderado'))}</td>
                <td class="price">{'$' + fmt(row.get('citizen_precio_nacionalizado')) if row.get('citizen_precio_nacionalizado') else '-'}</td>
                <td class="price">{fmt(row.get('markup_percent')) + '%' if row.get('markup_percent') is not None else '-'}</td>
                <td class="price">{fmt(row.get('margin_percent')) + '%' if row.get('margin_percent') is not None else '-'}</td>
                <td title="{row.get('cintolo_descripcion', 'Sin match')}">{row.get('cintolo_descripcion', '-')}</td>
                <td class="price">{'$' + fmt(row.get('cintolo_precio')) if row.get('cintolo_precio') else '-'}</td>
                <td class="price">{fmt(row.get('diff_percent_cintolo')) + '%' if row.get('diff_percent_cintolo') else '-'}</td>
                <td title="{row.get('zaloze_descripcion', 'Sin match')}">{row.get('zaloze_descripcion', '-')}</td>
                <td class="price">{'$' + fmt(row.get('zaloze_precio')) if row.get('zaloze_precio') else '-'}</td>
                <td class="price">{fmt(row.get('diff_percent_zaloze')) + '%' if row.get('diff_percent_zaloze') else '-'}</td>
                <td class="{position_class}">{position}</td>
                <td><span class="badge {badge_class}">{match_status}</span></td>
            </tr>
        ''')
    
    # Generate full HTML document
    html = f'''<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Comparación de Precios - Dialfa - {datetime.now().strftime("%Y-%m-%d %H:%M")}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f5f5f5;
            padding: 20px;
        }}
        
        .container {{
            max-width: 100%;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        h1 {{
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 28px;
        }}
        
        .subtitle {{
            color: #7f8c8d;
            margin-bottom: 20px;
            font-size: 14px;
        }}
        
        .export-info {{
            background-color: #e8f4f8;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 5px;
        }}
        
        .export-info strong {{
            color: #2c3e50;
        }}
        
        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        
        .card.green {{
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        }}
        
        .card.orange {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }}
        
        .card.blue {{
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        }}
        
        .card-title {{
            font-size: 14px;
            opacity: 0.9;
            margin-bottom: 10px;
        }}
        
        .card-value {{
            font-size: 32px;
            font-weight: bold;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            font-size: 12px;
        }}
        
        thead th {{
            background-color: #34495e;
            color: white;
            padding: 12px 8px;
            text-align: left;
            font-weight: 600;
            position: sticky;
            top: 0;
        }}
        
        tbody td {{
            padding: 10px 8px;
            border-bottom: 1px solid #ecf0f1;
        }}
        
        tbody tr:hover {{
            background-color: #ecf0f1;
        }}
        
        .margin-low {{
            background-color: #ffebee !important;
            color: #c62828;
        }}
        
        .margin-medium {{
            background-color: #fff8e1 !important;
            color: #f57c00;
        }}
        
        .margin-high {{
            background-color: #e8f5e9 !important;
            color: #2e7d32;
        }}
        
        .price {{
            text-align: right;
            font-family: 'Courier New', monospace;
        }}
        
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 3px;
            font-size: 11px;
            font-weight: 600;
        }}
        
        .badge-success {{
            background-color: #d4edda;
            color: #155724;
        }}
        
        .badge-danger {{
            background-color: #f8d7da;
            color: #721c24;
        }}
        
        .position-cheapest {{
            color: #27ae60;
            font-weight: bold;
        }}
        
        .position-expensive {{
            color: #e74c3c;
            font-weight: bold;
        }}
        
        .position-middle {{
            color: #f39c12;
            font-weight: bold;
        }}
        
        @media print {{
            body {{
                padding: 0;
            }}
            .container {{
                box-shadow: none;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Sistema de Comparación de Precios - Dialfa</h1>
        <p class="subtitle">Comparación integral: Dialfa vs Citizen (Proveedor) vs Competencia (Cintolo, Zaloze)</p>
        
        <div class="export-info">
            <strong>📅 Fecha de exportación:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}<br>
            <strong>💰 Descuento Dialfa:</strong> {discount_percent}%<br>
            <strong>🌍 Nacionalización:</strong> {nationalization_percent}%<br>
            <strong>🏪 Descuento Cintolo:</strong> {cintolo_discount}%<br>
            <strong>🏪 Descuento Zaloze:</strong> {zaloze_discount}%<br>
            <strong>🔍 Filtros aplicados:</strong> {filter_info}<br>
            <strong>📦 Total de productos:</strong> {summary['total_products']}
        </div>
        
        <div class="summary-cards">
            <div class="card">
                <div class="card-title">Total Productos Dialfa</div>
                <div class="card-value">{summary['total_products']}</div>
            </div>
            <div class="card green">
                <div class="card-title">✓ Match Citizen</div>
                <div class="card-value">{summary['matched_products']}</div>
            </div>
            <div class="card orange">
                <div class="card-title">✗ Sin Match Citizen</div>
                <div class="card-value">{summary['unmatched_citizen']}</div>
            </div>
            <div class="card blue">
                <div class="card-title">Margen Promedio</div>
                <div class="card-value">{summary['average_margin']:.1f}%</div>
            </div>
        </div>
        
        <div class="summary-cards">
            <div class="card" style="background: linear-gradient(135deg, #a8e063 0%, #56ab2f 100%);">
                <div class="card-title">✓ Match Cintolo</div>
                <div class="card-value">{summary['matched_cintolo']}</div>
            </div>
            <div class="card" style="background: linear-gradient(135deg, #fc4a1a 0%, #f7b733 100%);">
                <div class="card-title">✗ Sin Match Cintolo</div>
                <div class="card-value">{summary['unmatched_cintolo']}</div>
            </div>
            <div class="card" style="background: linear-gradient(135deg, #89f7fe 0%, #66a6ff 100%);">
                <div class="card-title">✓ Match Zaloze</div>
                <div class="card-value">{summary['matched_zaloze']}</div>
            </div>
            <div class="card" style="background: linear-gradient(135deg, #fddb92 0%, #d1fdff 100%); color: #333;">
                <div class="card-title">✗ Sin Match Zaloze</div>
                <div class="card-value">{summary['unmatched_zaloze']}</div>
            </div>
        </div>
        
        <h2 style="color: #2c3e50; margin-top: 30px; margin-bottom: 15px;">Detalle de Productos</h2>
        
        <table>
            <thead>
                <tr>
                    <th>Código Dialfa</th>
                    <th>Descripción</th>
                    <th>Precio Dialfa</th>
                    <th>Precio c/ Descuento</th>
                    <th>Producto Citizen</th>
                    <th>Cantidad Total</th>
                    <th>FOB Min</th>
                    <th>FOB Max</th>
                    <th>FOB Ponderado</th>
                    <th>FOB Pond. + Nac.</th>
                    <th>Rentabilidad %</th>
                    <th>Margen Bruto %</th>
                    <th>Match Cintolo</th>
                    <th>Precio Cintolo</th>
                    <th>Dif vs Cintolo</th>
                    <th>Match Zaloze</th>
                    <th>Precio Zaloze</th>
                    <th>Dif vs Zaloze</th>
                    <th>Posición Mercado</th>
                    <th>Estado</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows_html)}
            </tbody>
        </table>
        
        <div style="margin-top: 30px; padding: 15px; background-color: #f8f9fa; border-radius: 5px; font-size: 12px; color: #6c757d;">
            <strong>Notas:</strong><br>
            • <strong>Rentabilidad %:</strong> Markup = (Precio Final - Costo) / Costo × 100<br>
            • <strong>Margen Bruto %:</strong> Margen = (Precio Final - Costo) / Precio Final × 100<br>
            • <strong>Categorías de Margen:</strong> Bajo (<20%), Medio (20-40%), Alto (>40%)<br>
            • Este reporte fue generado automáticamente por el Sistema de Comparación de Precios - Dialfa
        </div>
    </div>
</body>
</html>'''
    
    return html


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

