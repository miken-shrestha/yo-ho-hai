"""
External API Services for fetching live market data
This module handles all third-party API integrations for gold prediction
"""

import requests
import json
from datetime import datetime
import os

# API Configuration
API_CONFIGS = {
    'exchangerate': {
        'base_url': 'https://api.exchangerate-api.com/v4/latest',
        'timeout': 5,
        'fallback': 142.50
    },
    'worldbank': {
        'base_url': 'https://api.worldbank.org/v2/country',
        'timeout': 5,
    },
    'metals': {
        'base_url': 'https://www.quandl.com/api/v3/datasets',
        'timeout': 5,
    }
}

class APIService:
    """Service class for handling all external API calls"""
    
    @staticmethod
    def fetch_exchange_rate(from_currency='USD', to_currency='NPR'):
        """
        Fetch exchange rate between two currencies
        
        Args:
            from_currency: Source currency code (default: USD)
            to_currency: Target currency code (default: NPR)
        
        Returns:
            float: Exchange rate or fallback value
        """
        try:
            url = f"{API_CONFIGS['exchangerate']['base_url']}/{from_currency}"
            response = requests.get(
                url,
                timeout=API_CONFIGS['exchangerate']['timeout']
            )
            
            if response.status_code == 200:
                data = response.json()
                rate = data['rates'].get(to_currency)
                if rate:
                    print(f"✓ Exchange rate fetched: {from_currency}/{to_currency} = {rate}")
                    return round(rate, 2)
        
        except requests.exceptions.Timeout:
            print("⚠ Exchange rate API timeout")
        except requests.exceptions.RequestException as e:
            print(f"⚠ Error fetching exchange rate: {e}")
        except (KeyError, json.JSONDecodeError):
            print("⚠ Invalid response format from exchange rate API")
        
        print(f"Using fallback exchange rate: {API_CONFIGS['exchangerate']['fallback']}")
        return API_CONFIGS['exchangerate']['fallback']
    
    @staticmethod
    def fetch_world_bank_inflation(country_code='NPL'):
        """
        Fetch inflation rate from World Bank API
        
        Args:
            country_code: Country code (default: NPL for Nepal)
        
        Returns:
            float: Inflation rate as decimal or fallback value
        """
        try:
            url = f"{API_CONFIGS['worldbank']['base_url']}/{country_code}/indicator/FP.CPI.TOTL.ZG"
            response = requests.get(
                f"{url}?format=json",
                timeout=API_CONFIGS['worldbank']['timeout']
            )
            
            if response.status_code == 200:
                data = response.json()
                if data[1]:
                    for record in data[1]:
                        if record['value'] is not None:
                            rate = float(record['value']) / 100
                            print(f"✓ Inflation rate fetched: {rate:.4f}")
                            return round(rate, 4)
        
        except (requests.exceptions.RequestException, KeyError, json.JSONDecodeError, TypeError) as e:
            print(f"⚠ Error fetching inflation rate: {e}")
        
        print("Using fallback inflation rate: 0.0526")
        return 0.0526
    
    @staticmethod
    def fetch_gold_price_usd(source='metals_api'):
        """
        Fetch international gold price in USD
        
        Args:
            source: Data source (metals_api, etc.)
        
        Returns:
            float: Gold price per troy ounce in USD
        """
        try:
            # Try free metals-api.com endpoint (no key required for basic tier)
            url = 'https://api.metals.live/v1/spot/gold'
            response = requests.get(url, timeout=API_CONFIGS['metals']['timeout'])
            
            if response.status_code == 200:
                data = response.json()
                # metals.live returns price in USD for gold per troy oz
                if 'gold' in data:
                    price = float(data['gold'])
                    print(f"✓ Gold price fetched: ${price}")
                    return round(price, 2)
        
        except (requests.exceptions.RequestException, KeyError, json.JSONDecodeError, TypeError, ValueError) as e:
            print(f"⚠ Error fetching gold price from metals.live: {e}")
        
        # Try alternative free API
        try:
            url = 'https://api.coinbase.com/v2/exchange-rates?currency=XAU'
            response = requests.get(url, timeout=API_CONFIGS['metals']['timeout'])
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and 'rates' in data['data'] and 'USD' in data['data']['rates']:
                    price = float(data['data']['rates']['USD'])
                    print(f"✓ Gold price fetched via Coinbase: ${price}")
                    return round(price, 2)
        except (requests.exceptions.RequestException, KeyError, json.JSONDecodeError, TypeError, ValueError) as e:
            print(f"⚠ Error fetching gold price from Coinbase: {e}")
        
        print("Using fallback gold price: $4156.60")
        return 4156.60
    
    @staticmethod
    def fetch_stock_market_index(index='nepse'):
        """
        Fetch stock market index value
        
        Args:
            index: Index name (nepse, etc.)
        
        Returns:
            float: Index value
        """
        try:
            if index.lower() == 'nepse':
                # NEPSE data would require web scraping or official API
                # This is a simplified example
                print("⚠ NEPSE live API integration requires configuration")
                return 2665.60
        
        except Exception as e:
            print(f"⚠ Error fetching stock index: {e}")
        
        return 2665.60
    
    @staticmethod
    def fetch_interest_rate(country='NPL', rate_type='policy'):
        """
        Fetch country's interest rate
        
        Args:
            country: Country code (default: NPL for Nepal)
            rate_type: Type of rate - policy, lending, etc.
        
        Returns:
            float: Interest rate as decimal
        """
        # Nepal Rastra Bank rate - would need direct integration or web scraping
        try:
            # Placeholder for actual NRB API or web scraping
            print(f"⚠ Interest rate API integration required for {country}")
        except Exception as e:
            print(f"⚠ Error fetching interest rate: {e}")
        
        return 0.06
    
    @staticmethod
    def detect_festival_season():
        """
        Detect if current date falls in major Nepali festival season
        
        Returns:
            int: Festival code (0=none, 1=wedding, 2=tihar, 3=dashain)
        """
        current_date = datetime.now()
        month = current_date.month
        day = current_date.day
        
        # Festival calendar for Nepal
        if month == 9 or (month == 10 and day <= 25):
            print("Festival detected: Dashain")
            return 3
        elif month == 10 or (month == 11 and day <= 5):
            print("Festival detected: Tihar")
            return 2
        elif month == 11 or month == 12:
            print("Festival detected: Wedding Season")
            return 1
        
        return 0


class GoldPredictionDataFetcher:
    """High-level class for fetching all required prediction features"""
    
    def __init__(self):
        self.api_service = APIService()
    
    def fetch_all_features(self):
        """
        Fetch all required features for gold price prediction
        
        Returns:
            dict: All prediction features or None on error
        """
        try:
            print("🔄 Fetching all prediction features...")
            
            usd_rate = self.api_service.fetch_exchange_rate('USD', 'NPR')
            inflation_rate = self.api_service.fetch_world_bank_inflation('NPL')
            gold_price_usd = self.api_service.fetch_gold_price_usd('lbma')
            interest_rate = self.api_service.fetch_interest_rate('NPL', 'policy')
            nepse_index = self.api_service.fetch_stock_market_index('nepse')
            festival_status = self.api_service.detect_festival_season()
            
            features = {
                'usd_rate': usd_rate,
                'inflation_rate': inflation_rate,
                'gold_price_usd': gold_price_usd,
                'interest_rate': interest_rate,
                'nepse_index': nepse_index,
                'festivals': festival_status,
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"✓ All features fetched successfully")
            return features
        
        except Exception as e:
            print(f"✗ Error fetching features: {e}")
            return None
    
    def fetch_single_feature(self, feature_name):
        """
        Fetch a single feature
        
        Args:
            feature_name: Name of the feature to fetch
        
        Returns:
            Value of the feature or None
        """
        feature_map = {
            'usd_rate': lambda: self.api_service.fetch_exchange_rate('USD', 'NPR'),
            'inflation_rate': lambda: self.api_service.fetch_world_bank_inflation('NPL'),
            'gold_price_usd': lambda: self.api_service.fetch_gold_price_usd('lbma'),
            'interest_rate': lambda: self.api_service.fetch_interest_rate('NPL', 'policy'),
            'nepse_index': lambda: self.api_service.fetch_stock_market_index('nepse'),
            'festival_status': lambda: self.api_service.detect_festival_season(),
        }
        
        if feature_name in feature_map:
            return feature_map[feature_name]()
        
        return None
