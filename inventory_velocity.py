"""
Inventory Velocity Analysis Module
Calculates product rotation/turnover metrics based on sales data.
Reuses logic from dialfa-analytics STOCK_VELOCITY_SUMMARY.
"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class InventoryVelocityAnalyzer:
    """Analyzes inventory velocity/turnover for products."""
    
    def __init__(self, db_connector):
        """
        Initialize velocity analyzer.
        
        Args:
            db_connector: DatabaseConnector instance for querying sales data
        """
        self.db_connector = db_connector
    
    def calculate_velocity(self, articulos_df, period_months=12):
        """
        Calculate velocity metrics for all products.
        
        Args:
            articulos_df: DataFrame with product data (must have 'idArticulo' and 'cantidad')
            period_months: Number of months to analyze (default: 12)
        
        Returns:
            DataFrame with velocity metrics merged with product data
        """
        try:
            logger.info(f"Calculating inventory velocity for {len(articulos_df)} products ({period_months} months)")
            
            # Get sales data from database
            sales_data = self.db_connector.get_sales_data(period_months)
            
            if sales_data.empty:
                logger.warning("No sales data found. All products will be marked as 'No Movement'")
                return self._create_empty_velocity_data(articulos_df)
            
            logger.info(f"Retrieved sales data for {len(sales_data)} products with sales")
            
            # Merge with articulos to get current stock
            velocity_df = articulos_df[['idArticulo', 'cantidad', 'codigo', 'descripcion']].copy()
            velocity_df = velocity_df.merge(
                sales_data,
                left_on='idArticulo',
                right_on='IdArticulo',
                how='left'
            )
            
            # Calculate velocity metrics
            velocity_df = self._calculate_metrics(velocity_df, period_months)
            
            # Classify velocity categories
            velocity_df = self._classify_velocity(velocity_df)
            
            # Classify demand categories (ABC analysis by volume)
            velocity_df = self._classify_demand(velocity_df)
            
            # Select and rename final columns
            result_cols = [
                'idArticulo',
                'velocity_category',
                'demand_category',
                'monthly_sales_avg',
                'months_of_stock',
                'annual_turnover_percent',
                'last_sale_date',
                'total_sold',
                'order_count'
            ]
            
            result_df = velocity_df[result_cols].copy()
            
            logger.info(f"✓ Velocity calculation complete:")
            logger.info(f"  - High Velocity: {(result_df['velocity_category'] == 'High Velocity').sum()}")
            logger.info(f"  - Medium Velocity: {(result_df['velocity_category'] == 'Medium Velocity').sum()}")
            logger.info(f"  - Low Velocity: {(result_df['velocity_category'] == 'Low Velocity').sum()}")
            logger.info(f"  - No Movement: {(result_df['velocity_category'] == 'No Movement').sum()}")
            logger.info(f"✓ Demand classification complete:")
            logger.info(f"  - High Demand (A): {(result_df['demand_category'] == 'High Demand').sum()}")
            logger.info(f"  - Medium Demand (B): {(result_df['demand_category'] == 'Medium Demand').sum()}")
            logger.info(f"  - Low Demand (C): {(result_df['demand_category'] == 'Low Demand').sum()}")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error calculating velocity: {e}")
            return self._create_empty_velocity_data(articulos_df)
    
    def _calculate_metrics(self, df, period_months):
        """Calculate core velocity metrics."""
        
        # Fill NaN values for products without sales
        df['TotalSold'] = df['TotalSold'].fillna(0)
        df['OrderCount'] = df['OrderCount'].fillna(0)
        df['LastSaleDate'] = pd.to_datetime(df['LastSaleDate'], errors='coerce')
        
        # Rename for consistency
        df['total_sold'] = df['TotalSold']
        df['order_count'] = df['OrderCount']
        df['last_sale_date'] = df['LastSaleDate']
        
        # Calculate average monthly sales
        df['monthly_sales_avg'] = df['total_sold'] / period_months
        
        # Calculate months of stock remaining
        # If avg monthly sales > 0, calculate how many months current stock will last
        # Otherwise, set to 999 (infinite)
        df['months_of_stock'] = np.where(
            df['monthly_sales_avg'] > 0,
            df['cantidad'] / df['monthly_sales_avg'],
            999
        )
        
        # Calculate annual turnover percentage
        # (Monthly sales avg * 12) / current stock * 100
        # This shows how many times per year the inventory turns over
        df['annual_turnover_percent'] = np.where(
            df['cantidad'] > 0,
            (df['monthly_sales_avg'] * 12) / df['cantidad'] * 100,
            0
        )
        
        # Cap values for display purposes
        df['months_of_stock'] = df['months_of_stock'].clip(upper=999)
        df['annual_turnover_percent'] = df['annual_turnover_percent'].clip(upper=9999)
        
        return df
    
    def _classify_velocity(self, df):
        """
        Classify products into velocity categories based on annual turnover.
        
        Categories (based on dialfa-analytics logic):
        - High Velocity: >= 400% annual turnover (4+ turns per year)
        - Medium Velocity: 200-400% annual turnover (2-4 turns per year)
        - Low Velocity: 0-200% annual turnover (<2 turns per year)
        - No Movement: 0% turnover (no sales in period)
        """
        
        def categorize(row):
            turnover = row['annual_turnover_percent']
            has_sales = row['total_sold'] > 0
            
            if not has_sales or turnover == 0:
                return 'No Movement'
            elif turnover >= 400:
                return 'High Velocity'
            elif turnover >= 200:
                return 'Medium Velocity'
            else:
                return 'Low Velocity'
        
        df['velocity_category'] = df.apply(categorize, axis=1)
        
        return df
    
    def _classify_demand(self, df):
        """
        Classify products into demand categories using ABC analysis by volume.
        
        This identifies "hot products" (papa caliente) based on absolute sales volume,
        independent of stock levels.
        
        Categories (Pareto principle):
        - High Demand (A): Top 20% of products by monthly sales volume (PAPA CALIENTE 🔥)
        - Medium Demand (B): Next 30% of products
        - Low Demand (C): Bottom 50% of products
        
        Products with zero sales are always Low Demand.
        """
        
        # Products with no sales are always Low Demand
        df['demand_category'] = 'Low Demand'
        
        # Filter products with sales
        has_sales = df['monthly_sales_avg'] > 0
        
        if has_sales.sum() == 0:
            # No products with sales, all are Low Demand
            return df
        
        # Calculate percentiles for products with sales
        # We use monthly_sales_avg as the metric for demand
        sales_values = df.loc[has_sales, 'monthly_sales_avg']
        
        # Calculate thresholds
        # Top 20% = High Demand (A)
        # Next 30% (20-50 percentile) = Medium Demand (B)
        # Bottom 50% = Low Demand (C)
        p80 = sales_values.quantile(0.80)  # Top 20%
        p50 = sales_values.quantile(0.50)  # Top 50%
        
        # Classify products with sales
        df.loc[has_sales & (df['monthly_sales_avg'] >= p80), 'demand_category'] = 'High Demand'
        df.loc[has_sales & (df['monthly_sales_avg'] >= p50) & (df['monthly_sales_avg'] < p80), 'demand_category'] = 'Medium Demand'
        # Low Demand already set as default
        
        return df
    
    def _create_empty_velocity_data(self, articulos_df):
        """Create empty velocity data for when no sales data is available."""
        
        result_df = pd.DataFrame({
            'idArticulo': articulos_df['idArticulo'],
            'velocity_category': 'No Movement',
            'demand_category': 'Low Demand',
            'monthly_sales_avg': 0.0,
            'months_of_stock': 999.0,
            'annual_turnover_percent': 0.0,
            'last_sale_date': pd.NaT,
            'total_sold': 0.0,
            'order_count': 0
        })
        
        return result_df
    
    def get_velocity_summary(self, velocity_df):
        """
        Get summary statistics for velocity data.
        
        Args:
            velocity_df: DataFrame with velocity metrics
        
        Returns:
            Dictionary with summary statistics
        """
        try:
            summary = {
                'total_products': len(velocity_df),
                'high_velocity_count': int((velocity_df['velocity_category'] == 'High Velocity').sum()),
                'medium_velocity_count': int((velocity_df['velocity_category'] == 'Medium Velocity').sum()),
                'low_velocity_count': int((velocity_df['velocity_category'] == 'Low Velocity').sum()),
                'no_movement_count': int((velocity_df['velocity_category'] == 'No Movement').sum()),
                'high_demand_count': int((velocity_df['demand_category'] == 'High Demand').sum()),
                'medium_demand_count': int((velocity_df['demand_category'] == 'Medium Demand').sum()),
                'low_demand_count': int((velocity_df['demand_category'] == 'Low Demand').sum()),
                'avg_monthly_sales': float(velocity_df['monthly_sales_avg'].mean()),
                'avg_months_of_stock': float(velocity_df[velocity_df['months_of_stock'] < 999]['months_of_stock'].mean()),
                'avg_turnover_percent': float(velocity_df['annual_turnover_percent'].mean())
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error calculating velocity summary: {e}")
            return {
                'total_products': 0,
                'high_velocity_count': 0,
                'medium_velocity_count': 0,
                'low_velocity_count': 0,
                'no_movement_count': 0,
                'high_demand_count': 0,
                'medium_demand_count': 0,
                'low_demand_count': 0,
                'avg_monthly_sales': 0.0,
                'avg_months_of_stock': 0.0,
                'avg_turnover_percent': 0.0
            }

