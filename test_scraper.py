from utils.price_scraper import IndiaMArtScraper
import time

def test_price_scraper():
    scraper = IndiaMArtScraper()
    
    # Test with some sample parts
    test_parts = [
        "engine oil",
        "brake pad",
        "air filter",
        "chain sprocket",
        "clutch plate"
    ]
    
    print("Testing IndiaMART Price Scraper...")
    print("-" * 50)
    
    for part in test_parts:
        print(f"\nTesting price scraping for: {part}")
        result = scraper.get_price_data(part)
        
        if result:
            print(f"✓ Success!")
            print("\nSummary:")
            print(f"Average Price: ₹{result['avg_price']:,.2f}")
            print(f"Price Range: ₹{result['min_price']:,.2f} - ₹{result['max_price']:,.2f}")
            
            print("\nDetailed Product Information:")
            print("-" * 50)
            for i, product in enumerate(result['products'], 1):
                print(f"\nProduct {i}:")
                print(f"Title: {product['title']}")
                print(f"Price: ₹{product['price']:,.2f}{product['unit']}")
                print(f"Seller: {product['company']}")
                print(f"Location: {product['location']}")
                print("-" * 25)
        else:
            print("✗ Failed to get price data")
        
        time.sleep(2)  # Delay between requests

def test_single_part():
    scraper = IndiaMArtScraper()
    
    while True:
        part_name = input("\nEnter part name to test (or 'quit' to exit): ")
        
        if part_name.lower() == 'quit':
            break
            
        print(f"\nSearching for: {part_name}")
        result = scraper.get_price_data(part_name)
        
        if result:
            print("\nResults found:")
            print("\nSummary:")
            print(f"Average Price: ₹{result['avg_price']:,.2f}")
            print(f"Price Range: ₹{result['min_price']:,.2f} - ₹{result['max_price']:,.2f}")
            
            print("\nDetailed Product Information:")
            print("-" * 50)
            for i, product in enumerate(result['products'], 1):
                print(f"\nProduct {i}:")
                print(f"Title: {product['title']}")
                print(f"Price: ₹{product['price']:,.2f}{product['unit']}")
                print(f"Seller: {product['company']}")
                print(f"Location: {product['location']}")
                print("-" * 25)
        else:
            print("No price data found")

if __name__ == "__main__":
    print("IndiaMART Price Scraper Test")
    print("1. Test multiple predefined parts")
    print("2. Test single part input")
    
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == "1":
        test_price_scraper()
    elif choice == "2":
        test_single_part()
    else:
        print("Invalid choice")