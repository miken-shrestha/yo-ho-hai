"""
API Configuration File
Store your API keys and endpoints here
"""

# Exchange Rate API Configuration
EXCHANGERATE_API = {
    'enabled': True,
    'provider': 'exchangerate-api.com',
    'endpoint': 'https://api.exchangerate-api.com/v4/latest',
    'timeout': 5
}

# World Bank API Configuration (Inflation, Economic Data)
WORLDBANK_API = {
    'enabled': True,
    'provider': 'World Bank Open Data',
    'endpoint': 'https://api.worldbank.org/v2/country',
    'timeout': 5
}

# Gold Price APIs Configuration
GOLD_PRICE_APIS = {
    'quandl': {
        'enabled': False,  # Set to True and add your API key
        'provider': 'Quandl',
        'api_key': 'YOUR_QUANDL_API_KEY',  # Replace with your key
        'endpoint': 'https://www.quandl.com/api/v3/datasets',
        'dataset': 'LBMA/GOLD'
    },
    'metals_api': {
        'enabled': False,  # Set to True and add your API key
        'provider': 'Metals API',
        'api_key': 'YOUR_METALS_API_KEY',  # Replace with your key
        'endpoint': 'https://api.metals.live/v1/spot/gold'
    },
    'fallback': {
        'price_usd': 2100.00,  # Default fallback price
        'source': 'Local Data'
    }
}

# NEPSE Stock Index Configuration
NEPSE_API = {
    'enabled': False,  # Set to True when API becomes available
    'provider': 'Nepal Stock Exchange',
    'endpoint': 'https://www.nepalstock.com.np/api/index',
    'fallback_index': 2665.60
}

# Interest Rate Configuration
INTEREST_RATE = {
    'enabled': False,  # Manual or NRB API integration
    'provider': 'Nepal Rastra Bank',
    'fallback_rate': 0.06
}

# Festival Calendar Configuration
FESTIVAL_CALENDAR = {
    'enabled': True,
    'auto_detect': True,  # Auto-detect based on current date
    'festivals': {
        'dashain': {'start_month': 9, 'start_day': 1, 'end_month': 10, 'end_day': 25},
        'tihar': {'start_month': 10, 'start_day': 15, 'end_month': 11, 'end_day': 5},
        'wedding_season': {'start_month': 11, 'start_day': 1, 'end_month': 12, 'end_day': 31}
    }
}

# Default Feature Values (Fallbacks)
DEFAULT_FEATURES = {
    'usd_rate': 142.50,
    'inflation_rate': 0.0526,
    'gold_price_usd': 2100.00,
    'interest_rate': 0.06,
    'nepse_index': 2665.60,
    'festivals': 0
}

# Cache Configuration
CACHE_SETTINGS = {
    'enabled': True,
    'ttl_seconds': 300,  # Cache for 5 minutes
    'cache_type': 'memory'  # or 'redis'
}
