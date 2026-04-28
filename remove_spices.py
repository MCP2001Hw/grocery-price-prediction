#!/usr/bin/env python3
"""
Script to remove spice and seasoning products from Data.csv
Based on manual identification patterns
"""

import pandas as pd
import re
import sys

def is_spice_or_seasoning(product_name, ingredients):
    """
    Determine if a product is primarily a spice or seasoning based on name and ingredients.
    
    Args:
        product_name (str): Name of the product
        ingredients (str): Ingredient list of the product
    
    Returns:
        bool: True if product is a spice/seasoning, False otherwise
    """
    
    # Convert to lowercase for case-insensitive matching
    name_lower = product_name.lower()
    ingredients_lower = str(ingredients).lower() if pd.notna(ingredients) else ""
    
    # Primary spice/seasoning indicators in product name
    spice_keywords = [
        'seasoning', 'spice', 'herbs', 'herb', 'curry powder', 'chili powder', 'chilli powder',
        'garlic salt', 'onion salt', 'aromat', 'paprika', 'turmeric', 'cumin', 'coriander',
        'black pepper', 'white pepper', 'cayenne', 'chipotle chilli flakes', 'chilli flakes',
        'garlic powder', 'onion powder', 'ginger powder', 'mustard powder',
        'garlic & ginger paste', 'chipotle chilli paste', 'spice mix', 'herb mix',
        'fajita seasoning', 'taco seasoning', 'bbq rub', 'dry rub'
    ]
    
    # Check if product name contains clear spice/seasoning indicators
    for keyword in spice_keywords:
        if keyword in name_lower:
            return True
    
    # Additional checks for products that are primarily seasonings/pastes
    paste_indicators = ['paste', 'pur??e', 'puree']
    spice_paste_words = ['garlic', 'ginger', 'chilli', 'chipotle', 'curry', 'spice']
    
    # Check for spice pastes
    for paste_word in paste_indicators:
        if paste_word in name_lower:
            for spice_word in spice_paste_words:
                if spice_word in name_lower:
                    # Additional check: if it's primarily a spice paste (not a cooking sauce with other main ingredients)
                    if not any(main_ingredient in name_lower for main_ingredient in 
                              ['tomato', 'coconut', 'cream', 'butter', 'cheese', 'meat', 'chicken', 'beef']):
                        return True
    
    # Check for standalone spice products (weight typically under 100g and specific patterns)
    weight_match = re.search(r'(\d+)g', name_lower)
    if weight_match:
        weight = int(weight_match.group(1))
        if weight <= 100:  # Small packages often indicate spices
            # Check if ingredients suggest it's primarily spices
            spice_ingredient_indicators = [
                'ground', 'dried', 'powder', 'flakes', 'seeds', 'extract',
                'paprika', 'cumin', 'coriander', 'turmeric', 'chilli', 'pepper',
                'garlic', 'onion', 'herbs', 'spices', 'salt'
            ]
            
            # Count spice-related words in ingredients
            spice_word_count = sum(1 for indicator in spice_ingredient_indicators 
                                 if indicator in ingredients_lower)
            
            # If more than 3 spice-related words and small package, likely a spice
            if spice_word_count >= 3:
                # Additional filter: exclude if it's clearly a food product with spices as flavoring
                food_product_indicators = [
                    'pasta', 'noodle', 'chip', 'crisp', 'biscuit', 'chocolate', 'candy',
                    'sauce', 'soup', 'meal', 'flour', 'rice', 'lentil', 'bean'
                ]
                
                is_food_product = any(food_word in name_lower for food_word in food_product_indicators)
                if not is_food_product:
                    return True
    
    return False

def remove_spices_from_csv(input_file='Data.csv', output_file='Data_no_spices.csv'):
    """
    Remove spice and seasoning products from the CSV file.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file
    """
    
    print(f"Reading {input_file}...")
    try:
        # Read the CSV file
        df = pd.read_csv(input_file, encoding='latin-1')
        print(f"Original dataset has {len(df)} rows")
        
        # Identify spice/seasoning products
        spice_mask = df.apply(lambda row: is_spice_or_seasoning(
            row['Name of the product'], 
            row['Ingredient of the product']
        ), axis=1)
        
        spice_products = df[spice_mask]
        print(f"\nFound {len(spice_products)} spice/seasoning products:")
        print("-" * 80)
        
        for idx, row in spice_products.iterrows():
            print(f"- {row['Name of the product']}")
        
        # Remove spice products
        df_cleaned = df[~spice_mask]
        print(f"\nAfter removal: {len(df_cleaned)} rows remaining")
        print(f"Removed {len(df) - len(df_cleaned)} products")
        
        # Save cleaned dataset
        df_cleaned.to_csv(output_file, index=False, encoding='latin-1')
        print(f"\nCleaned dataset saved to {output_file}")
        
        # Also update the original file
        if input_file == 'Data.csv':
            df_cleaned.to_csv('Data.csv', index=False, encoding='latin-1')
            print(f"Original {input_file} updated with cleaned data")
        
        return df_cleaned, spice_products
        
    except FileNotFoundError:
        print(f"Error: File {input_file} not found!")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

def main():
    """Main function"""
    print("🌶️  Spice and Seasoning Removal Tool")
    print("=" * 50)
    
    # Remove spices from the dataset
    cleaned_df, removed_df = remove_spices_from_csv()
    
    print("\n✅ Spice removal completed successfully!")
    print(f"📊 Dataset reduced from {len(cleaned_df) + len(removed_df)} to {len(cleaned_df)} products")
    
    # Show some statistics
    print(f"\n📈 Removal Statistics:")
    print(f"   • Products removed: {len(removed_df)}")
    print(f"   • Products remaining: {len(cleaned_df)}")
    print(f"   • Removal percentage: {(len(removed_df) / (len(cleaned_df) + len(removed_df)) * 100):.1f}%")

if __name__ == "__main__":
    main()