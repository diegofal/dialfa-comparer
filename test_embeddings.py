"""
Quick test script for embedding-based matching.
Tests the embedding matcher with a small sample of products.
"""
import os
import pandas as pd
from dotenv import load_dotenv
from embedding_matcher import EmbeddingMatcher
from database.db_connector import DatabaseConnector
from data_extractors.citizen_extractor import CitizenExtractor

# Load environment variables from .env file
load_dotenv()

def main():
    print("=" * 80)
    print("Testing Embedding-Based Product Matching")
    print("=" * 80)
    
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("\nX Error: OPENAI_API_KEY not found in environment variables")
        print("Please set it before running this test:")
        print("  export OPENAI_API_KEY=sk-your-api-key-here")
        return
    
    print(f"\n[OK] API Key found: {api_key[:10]}...{api_key[-4:]}")
    
    # Load data
    print("\n[1/3] Loading Dialfa products from database...")
    db_connector = DatabaseConnector()
    dialfa_df = db_connector.get_articulos()
    print(f"[OK] Loaded {len(dialfa_df)} Dialfa products")
    
    print("\n[2/3] Loading Citizen products...")
    citizen_extractor = CitizenExtractor()
    citizen_df = citizen_extractor.extract()
    print(f"[OK] Loaded {len(citizen_df)} Citizen products")
    
    # Test with a small sample for quick validation
    print("\n[3/3] Testing embedding matching (sample of 50 products)...")
    sample_size = min(50, len(dialfa_df))
    dialfa_sample = dialfa_df.head(sample_size).copy()
    
    # Initialize matcher
    matcher = EmbeddingMatcher(api_key=api_key)
    
    # Perform matching
    matched_df, stats = matcher.match_products(
        source_df=dialfa_sample,
        target_df=citizen_df,
        source_name="Dialfa",
        target_name="Citizen",
        threshold=0.85
    )
    
    # Display results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Total products tested: {sample_size}")
    print(f"Matched: {stats['matched']} ({stats['match_rate']}%)")
    print(f"Unmatched: {stats['unmatched']}")
    print(f"Threshold: {stats['threshold']}")
    
    # Show some examples
    print("\n" + "=" * 80)
    print("SAMPLE MATCHES (Top 5)")
    print("=" * 80)
    
    matched_rows = matched_df[matched_df['citizen_match_score'].notna()].head(5)
    
    if len(matched_rows) > 0:
        for idx, row in matched_rows.iterrows():
            print(f"\n[OK] Match Score: {row['citizen_match_score']}%")
            print(f"  Dialfa: {row['descripcion'][:60]}")
            print(f"  Citizen: {row['citizen_descripcion_emb'][:60]}")
            print(f"  Price: USD ${row['citizen_precio_usd_emb']}")
    else:
        print("\nNo matches found. You may need to lower the threshold.")
    
    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)

if __name__ == '__main__':
    main()

