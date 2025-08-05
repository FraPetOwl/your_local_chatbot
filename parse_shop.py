# parse_shop.py
# This script parses the cleaned shop text file (shop_clean.txt) and converts it into a 
# structured JSON format for the LLM

import json

def parse_shop_data(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        lines = [line.strip() for line in file.readlines()]
    
    # Initialize variables
    current_category = None
    categories = {}
    last_product_name = None
    
    for line in lines:
        if not line:
            continue
            
        # Skip header/footer sections and navigation items
        if line in ['Superior Sounds', 'Home', 'About', '© 2025 Superior Sounds'] or \
           'All products' in line or line.startswith('|') or \
           line in ['Stores', 'English', 'Finnish', 'German', 'Swedish', 'Hungarian', 
                   'French', 'Spanish', 'Italian', 'Danish', 'Norwegian', 'Czech', 
                   'Lithuanian', 'Dutch', 'Estonian', 'Russian', 'Slovak', 
                   'We care about your privacy', 'Show details', 'Hide details',
                   'Accept selected', 'Accept all',
                   'We use cookies and similar technologies to provide the best experience on our website.',
                   'Powered by']:
            continue
            
        # Check if line is a main category
        if line in ['Audio', 'Lighting', 'DJ Gear', 'Accessories, Cables, Misc', 
                   'Visuals', 'Soundboards', 'Add-Ons', 'Microphones', 'Truss']:
            current_category = line
            categories[current_category] = []
            last_product_name = None
        
        # If line contains "From" and we have a current category, it's a price
        elif line.startswith('From') and current_category and last_product_name:
            # Extract price
            price = int(line.split('From')[1].strip().split(' ')[0])
            
            categories[current_category].append({
                "name": last_product_name,
                "price_from": price
            })
        else:
            # If we're in a category and the line isn't "From", it's probably a product name
            if current_category and not line.startswith('From'):
                last_product_name = line
    
    return categories

# Parse the data and convert to JSON
data = parse_shop_data('Shop_clean.txt')
formatted_json = json.dumps(data, indent=2, ensure_ascii=False)

# Save the JSON to a file
with open('shop_data.json', 'w', encoding='utf-8') as f:
    f.write(formatted_json)

print("JSON data has been saved to shop_data.json")
