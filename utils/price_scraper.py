import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import quote
import time
import random

class IndiaMArtScraper:
    def __init__(self):
        self.base_url = "https://dir.indiamart.com/search.mp?ss="
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }
        
    def get_price_data(self, part_name):
        try:
            search_query = f"{part_name} motorcycle spare part"
            encoded_query = quote(search_query)
            url = self.base_url + encoded_query
            
            time.sleep(random.uniform(1, 3))
            
            response = requests.get(url, headers=self.headers)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            products = []
            # Find all product cards
            product_cards = soup.find_all('div', class_='cardbody')
            
            for card in product_cards[:5]:  # Get first 5 products
                try:
                    # Extract product title
                    title = card.find('div', class_='producttitle').text.strip()
                    
                    # Extract price and unit
                    price_elem = card.find('p', class_='price')
                    if price_elem:
                        price_text = price_elem.get_text(strip=True)
                        unit = price_elem.find('span', class_='unit')
                        unit_text = unit.get_text(strip=True) if unit else ''
                        
                        # Extract numeric price
                        price = float(price_text.replace('₹', '').replace(',', '').split('/')[0])
                        
                        # Extract company name
                        company_elem = card.find('div', class_='companyname')
                        company_name = company_elem.text.strip() if company_elem else 'N/A'
                        
                        # Extract location
                        location_elem = card.find('div', class_='newLocationUi')
                        location = location_elem.text.strip() if location_elem else 'N/A'
                    
                    # Extract rating data with default values
                    rating = None
                    num_ratings = 0
                    rating_div = card.find('div', id=lambda x: x and x.startswith('sellerrating_'))
                    
                    if rating_div:
                        # Get rating value
                        rating_value = rating_div.find('span', class_='ratingValue')
                        if rating_value and rating_value.text:
                            try:
                                rating = float(rating_value.text.split('/')[0])
                            except (ValueError, AttributeError):
                                rating = None
                        
                        # Get number of ratings
                        rating_count = rating_div.find('span', class_='color')
                        if rating_count and rating_count.text:
                            try:
                                num_ratings = int(rating_count.text.strip().strip('()'))
                            except (ValueError, AttributeError):
                                num_ratings = 0
                                print(f"Error parsing rating count: {rating_count.text}")
                    
                    if price > 0:
                        products.append({
                            'title': title,
                            'price': price,
                            'unit': unit_text,
                            'company': company_name,
                            'location': location,
                            'rating': rating,
                            'num_ratings': num_ratings
                        })
                except Exception as e:
                    print(f"Error processing product: {str(e)}")
                    continue
            
            if products:
                prices = [p['price'] for p in products]
                return {
                    'part_name': part_name,
                    'avg_price': sum(prices) / len(prices),
                    'min_price': min(prices),
                    'max_price': max(prices),
                    'products': products  # Include detailed product information
                }
            return None
            
        except Exception as e:
            print(f"Error scraping {part_name}: {str(e)}")
            return None