from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.http import require_http_methods
from django.middleware.csrf import get_token
from django.views.decorators.csrf import csrf_exempt
import json
import joblib
import numpy as np
import os
import time
import threading
from functools import wraps
from django.conf import settings
from .api_services import GoldPredictionDataFetcher
from .models import Prediction
import pandas as pd

# --- Request spike detection decorator (MOVE THIS TO TOP) ---
_request_lock = threading.Lock()
_request_counts = {}  # path -> list[timestamps]

def detect_request_spike(max_calls: int = 8, window_seconds: float = 3.0):
    """Decorator to block a path if it's called > max_calls within window_seconds.
    Useful to detect and stop accidental request loops during debugging.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(request, *args, **kwargs):
            path = request.path
            now = time.time()
            with _request_lock:
                calls = _request_counts.get(path, [])
                # keep only recent
                calls = [t for t in calls if now - t <= window_seconds]
                calls.append(now)
                _request_counts[path] = calls
                if len(calls) > max_calls:
                    print(f"[DEBUG] Request spike detected on {path} ({len(calls)} calls in {window_seconds}s). Blocking to avoid loop.")
                    return HttpResponse("Request spike detected; aborting to prevent loop. Check logs.", status=429)
            # proceed
            try:
                return func(request, *args, **kwargs)
            finally:
                # optional: keep counts for a short time for diagnostics
                pass
        return wrapper
    return decorator

# --- Reentrancy guard for live-data fetch ---
_in_fetch_live_lock = threading.Lock()
_in_fetch_live = False

# --- Conversion constants ---
GRAM_TO_TOLA = 11.66  # 1 tola = 11.66 grams
OZ_TO_TOLA = 2.667    # 1 ounce = 31.1035g / 11.66g = ~2.667 tola

# Load the BEST model and ALL metrics
BEST_MODEL_PATH = os.path.join(settings.BASE_DIR, 'ml_models', 'linear_regression_gold_model.pkl')
ALL_METRICS_PATH = os.path.join(settings.BASE_DIR, 'ml_models', 'all_model_comparison_metrics.pkl')
SCALER_PATH = os.path.join(settings.BASE_DIR, 'ml_models', 'standard_scaler.pkl')

try:
    # Load the best model (Linear Regression)
    best_model = joblib.load(BEST_MODEL_PATH)
    all_metrics = joblib.load(ALL_METRICS_PATH)
    
    # Try to load scaler if it exists
    try:
        scaler = joblib.load(SCALER_PATH)
    except:
        scaler = None
        print("Warning: Scaler not found or failed to load")
        
except Exception as e:
    best_model = None
    all_metrics = None
    scaler = None
    print(f"Error loading models: {e}")

def get_prediction_in_npr_per_tola(features_df, use_fallback=True):
    """
    Helper function to get prediction in NPR per tola with consistent logic.
    Returns: (prediction, used_fallback)
    """
    if best_model is None:
        # Fallback calculation when model not available
        usd_rate = features_df['USD_rate'].iloc[0]
        gold_price_usd = features_df['gold_price_USD'].iloc[0]
        inflation_rate = features_df['Inflation_rate'].iloc[0]
        festivals = features_df['festivals'].iloc[0]
        
        # Calculate using consistent formula
        gold_price_usd_per_tola = gold_price_usd / OZ_TO_TOLA
        base_prediction = gold_price_usd_per_tola * usd_rate
        inflation_adjustment = 1 + float(inflation_rate)
        festival_boost = 1 + (int(festivals) * 0.01)
        fallback_pred = base_prediction * inflation_adjustment * festival_boost
        return fallback_pred, True  # True indicates fallback was used
    
    try:
        # Scale features if scaler is available
        if scaler is not None:
            features_scaled = scaler.transform(features_df)
        else:
            features_scaled = features_df.values
        
        # Model predicts NPR per gram
        prediction_per_gram = best_model.predict(features_scaled)[0]
        
        # Convert from NPR per gram to NPR per tola
        prediction_per_tola = prediction_per_gram * GRAM_TO_TOLA
        
        # Validate prediction
        gold_price_usd = features_df['gold_price_USD'].iloc[0]
        usd_rate = features_df['USD_rate'].iloc[0]
        
        # Calculate expected range (80-120% of direct conversion)
        gold_price_usd_per_tola = gold_price_usd / OZ_TO_TOLA
        direct_conversion = gold_price_usd_per_tola * usd_rate
        min_reasonable = direct_conversion * 0.8
        max_reasonable = direct_conversion * 1.2
        
        # Check if prediction is reasonable
        if not (min_reasonable <= prediction_per_tola <= max_reasonable):
            if use_fallback:
                # Use adjusted fallback
                inflation_rate = features_df['Inflation_rate'].iloc[0]
                festivals = features_df['festivals'].iloc[0]
                inflation_adjustment = 1 + float(inflation_rate)
                festival_boost = 1 + (int(festivals) * 0.01)
                fallback_pred = direct_conversion * inflation_adjustment * festival_boost
                return fallback_pred, True  # True indicates fallback was used
            else:
                # Clip to reasonable range
                prediction_per_tola = np.clip(prediction_per_tola, min_reasonable, max_reasonable)
        
        return prediction_per_tola, False  # False indicates model prediction was used
        
    except Exception as e:
        print(f"Prediction failed in helper: {e}")
        # Fallback calculation
        usd_rate = features_df['USD_rate'].iloc[0]
        gold_price_usd = features_df['gold_price_USD'].iloc[0]
        inflation_rate = features_df['Inflation_rate'].iloc[0]
        festivals = features_df['festivals'].iloc[0]
        
        gold_price_usd_per_tola = gold_price_usd / OZ_TO_TOLA
        direct_conversion = gold_price_usd_per_tola * usd_rate
        inflation_adjustment = 1 + float(inflation_rate)
        festival_boost = 1 + (int(festivals) * 0.01)
        fallback_pred = direct_conversion * inflation_adjustment * festival_boost
        return fallback_pred, True

# --- Keep the rest of your code the same from here ---
@csrf_exempt
@detect_request_spike()   # Now this will work because detect_request_spike is defined above
def fetch_live_data(request):
    """Fetch live data for the frontend - wrapper around fetch_live_data_api"""
    print(f"[DEBUG] fetch_live_data called method={request.method} path={request.path}")
    return fetch_live_data_api(request)

def index(request):
    """Home page with prediction form"""
    # Get best model metrics
    best_model_metrics = None
    model_comparison = None
    
    if all_metrics and 'Linear Regression' in all_metrics:
        best_model_metrics = all_metrics['Linear Regression']
        
        # Prepare comparison data for template
        model_comparison = [
            {
                'name': 'Linear Regression',
                'test_r2': f"{all_metrics['Linear Regression']['test_r2']:.4f}",
                'test_rmse': f"{all_metrics['Linear Regression']['test_rmse']:.2f}",
                'test_mae': f"{all_metrics['Linear Regression']['test_mae']:.2f}",
                'is_best': True
            },
            {
                'name': 'Gradient Boosting', 
                'test_r2': f"{all_metrics['Gradient Boosting']['test_r2']:.4f}",
                'test_rmse': f"{all_metrics['Gradient Boosting']['test_rmse']:.2f}",
                'test_mae': f"{all_metrics['Gradient Boosting']['test_mae']:.2f}",
                'is_best': False
            },
            {
                'name': 'Random Forest',
                'test_r2': f"{all_metrics['Random Forest']['test_r2']:.4f}",
                'test_rmse': f"{all_metrics['Random Forest']['test_rmse']:.2f}",
                'test_mae': f"{all_metrics['Random Forest']['test_mae']:.2f}",
                'is_best': False
            }
        ]
    
    context = {
        'best_model_metrics': best_model_metrics,
        'model_comparison': model_comparison,
        'model_loaded': best_model is not None,
        'scaler_loaded': scaler is not None,
        'best_model_name': 'Linear Regression'
    }
    return render(request, 'predictor/index.html', context)

@csrf_exempt
def calculate_fallback_prediction(usd_rate, gold_price_usd, inflation_rate, interest_rate, festivals, nepse_index):
    """Calculate fallback prediction when model is unavailable.

    Expects `inflation_rate` and `interest_rate` as decimals (e.g. 0.05 for 5%).
    """
    # Simple calculation: convert gold price to NPR with adjustments
    base_prediction = gold_price_usd * usd_rate * 0.9  # 90% of international conversion

    # inflation_rate is expected as decimal (e.g. 0.05 => +5%)
    inflation_adjustment = 1 + float(inflation_rate)

    # Festival boost (festivals 0-3 scale)
    festival_boost = 1 + (int(festivals) * 0.01)

    final_prediction = base_prediction * inflation_adjustment * festival_boost
    return final_prediction

@csrf_exempt
def predict(request):
    """Handle prediction requests using the BEST model"""
    if request.method == 'POST':
        try:
            # Get input data from POST request
            data = json.loads(request.body)
            
            print("=== PREDICTION DEBUG INFO ===")
            print(f"Raw input data: {data}")
            
            # Extract features with validation
            try:
                usd_rate = float(data.get('usd_rate', 0))
                inflation_rate = float(data.get('inflation_rate', 0))
                gold_price_usd = float(data.get('gold_price_usd', 0))  # USD per ounce
                interest_rate = float(data.get('interest_rate', 0))
                festivals = int(data.get('festivals', 0))
                nepse_index = float(data.get('nepse_index', 0))
            except (TypeError, ValueError) as e:
                return JsonResponse({
                    'success': False,
                    'error': f'Invalid input format: {str(e)}'
                })
            
            # Validate input ranges
            if inflation_rate > 1:
                inflation_rate = inflation_rate / 100.0
            if interest_rate > 1:
                interest_rate = interest_rate / 100.0

            if festivals < 0:
                festivals = 0
            if festivals > 3:
                festivals = 3

            # Prepare features for prediction
            features_df = pd.DataFrame([{
                'USD_rate': usd_rate,
                'Inflation_rate': inflation_rate,
                'gold_price_USD': gold_price_usd,  # USD per ounce
                'interest_rate': interest_rate,
                'festivals': festivals,
                'nepse_index': nepse_index
            }])
            
            print(f"Features for prediction:\n{features_df}")
            
            # Use the helper function for consistent prediction
            prediction_per_tola, used_fallback = get_prediction_in_npr_per_tola(features_df)
            
            # Ensure minimum reasonable price (based on direct conversion)
            gold_price_usd_per_tola = gold_price_usd / OZ_TO_TOLA
            direct_conversion = gold_price_usd_per_tola * usd_rate
            min_reasonable = direct_conversion * 0.7  # Allow down to 70% of direct conversion
            
            if prediction_per_tola < min_reasonable:
                print(f"WARNING: Low prediction: {prediction_per_tola}. Clipping to {min_reasonable}")
                prediction_per_tola = min_reasonable
                used_fallback = True
            
            final_prediction = round(float(prediction_per_tola), 2)
            print(f"Final prediction: {final_prediction} NPR per tola")
            
            # Save prediction to database
            try:
                pred_obj = Prediction.objects.create(
                    predicted_price=final_prediction,
                    usd_rate=usd_rate,
                    inflation_rate=inflation_rate,
                    gold_price_usd=gold_price_usd,
                    interest_rate=interest_rate,
                    festivals=festivals,
                    nepse_index=nepse_index,
                    model_used='Linear Regression (Best Model)',
                    model_accuracy=float(all_metrics['Linear Regression']['test_r2']) if all_metrics else None
                )
                print(f"Prediction saved to DB with ID: {pred_obj.id}")
            except Exception as e:
                print(f"Warning: Failed to save prediction to DB: {e}")
            
            # Calculate gold price in USD per tola for reference
            gold_price_usd_per_tola = gold_price_usd / OZ_TO_TOLA
            
            return JsonResponse({
                'success': True,
                'prediction': final_prediction,
                'prediction_unit': 'NPR per tola',
                'model_used': 'Linear Regression (Best Model)',
                'model_accuracy': f"{all_metrics['Linear Regression']['test_r2']:.4f}" if all_metrics else 'N/A',
                'features': {
                    'USD Rate': usd_rate,
                    'Inflation Rate': f"{inflation_rate:.4f}",
                    'Gold Price (USD/oz)': gold_price_usd,
                    'Gold Price (USD/tola)': round(gold_price_usd_per_tola, 2),
                    'Interest Rate': f"{interest_rate:.3f}",
                    'Festival': ['No', 'Wedding', 'Tihar', 'Dashain'][festivals],
                    'NEPSE Index': nepse_index
                },
                'debug_info': {
                    'direct_conversion_npr_tola': round(gold_price_usd_per_tola * usd_rate, 2),
                    'fallback_used': used_fallback
                }
            })
                
        except Exception as e:
            print(f"General prediction error: {e}")
            return JsonResponse({
                'success': False,
                'error': f'Prediction error: {str(e)}'
            })
    
    return JsonResponse({
        'success': False,
        'error': 'Invalid request method'
    })

@csrf_exempt
@detect_request_spike()
def fetch_live_data_api(request):
    """Fetch live features using `GoldPredictionDataFetcher` and return JSON."""
    global _in_fetch_live
    # Reentrancy protection: if another thread is already performing fetch_live_data_api,
    # abort this request to avoid recursion/looping into the same handler.
    with _in_fetch_live_lock:
        if _in_fetch_live:
            print("[DEBUG] Re-entrant call to fetch_live_data_api detected; aborting this call to avoid loop.")
            return JsonResponse({'success': False, 'error': 'Re-entrant fetch detected'}, status=429)
        _in_fetch_live = True

    try:
        print(f"[DEBUG] fetch_live_data_api called method={request.method} path={request.path}")
        if request.method in ('GET', 'POST'):
            try:
                fetcher = GoldPredictionDataFetcher()
                features = fetcher.fetch_all_features()
                if features:
                    return JsonResponse({'success': True, 'data': features})
                else:
                    print("[DEBUG] fetch_live_data_api: fetcher returned no features")
                    return JsonResponse({'success': False, 'error': 'Failed to fetch live features'}, status=502)
            except Exception as e:
                print(f"Error in fetch_live_data_api: {e}")
                return JsonResponse({'success': False, 'error': str(e)}, status=500)

        return JsonResponse({'success': False, 'error': 'Invalid request method'}, status=405)
    finally:
        # Clear reentrancy flag so future requests can proceed
        with _in_fetch_live_lock:
            _in_fetch_live = False



@csrf_exempt
def auto_fetch_features(request):
    """Alias endpoint to automatically fetch features.

    For now this simply returns the same payload as `fetch_live_data_api`.
    """
    return fetch_live_data_api(request)

# ... rest of your functions (model_info, model_comparison, model_details) remain the same
def model_info(request):
    """Display model information and metrics"""
    best_model_metrics = None
    if all_metrics and 'Linear Regression' in all_metrics:
        best_model_metrics = all_metrics['Linear Regression']
    
    context = {
        'best_model_metrics': best_model_metrics,
        'model_loaded': best_model is not None,
        'best_model_name': 'Linear Regression'
    }
    return render(request, 'predictor/index.html', context)

def model_comparison(request):
    """Display detailed model comparison"""
    if all_metrics:
        context = {
            'model_comparison': all_metrics,
            'best_model': 'Linear Regression',
            'best_model_metrics': all_metrics['Linear Regression']
        }
    else:
        context = {
            'model_comparison': None,
            'best_model': None,
            'best_model_metrics': None
        }
    
    return render(request, 'predictor/model_comparison.html', context)

def model_details(request):
    """Display detailed information about the best model"""
    best_model_metrics = None
    if all_metrics and 'Linear Regression' in all_metrics:
        best_model_metrics = all_metrics['Linear Regression']
    
    context = {
        'model_name': 'Linear Regression',
        'metrics': best_model_metrics,
        'model_loaded': best_model is not None,
        'model_description': 'Best performing model with near-perfect accuracy (R²: 0.9999)',
        'advantages': [
            'Highest accuracy among all tested models',
            'Fast prediction times',
            'Simple and interpretable',
            'Less prone to overfitting'
        ]
    }
    return render(request, 'predictor/model_details.html', context)

@csrf_exempt
@detect_request_spike()
def predict_7_days(request):
    """Make predictions for the next 7 days using current market conditions"""
    print(f"[DEBUG] predict_7_days called method={request.method} path={request.path}")
    if request.method == 'POST':
        try:
            from datetime import datetime, timedelta
            print("=== 7-DAY PREDICTION REQUEST ===")

            # Get input data
            data = json.loads(request.body or '{}')
            
            # Extract features
            usd_rate = float(data.get('usd_rate', 142))
            inflation_rate = float(data.get('inflation_rate', 0.05))
            gold_price_usd_per_oz = float(data.get('gold_price_usd', 4192))  # USD per ounce
            interest_rate = float(data.get('interest_rate', 0.08))
            festivals = int(data.get('festivals', 0))
            nepse_index = float(data.get('nepse_index', 2500))

            # Normalize percentages
            if inflation_rate > 1:
                inflation_rate = inflation_rate / 100.0
            if interest_rate > 1:
                interest_rate = interest_rate / 100.0
            if festivals < 0:
                festivals = 0
            if festivals > 3:
                festivals = 3
            
            predictions_7_days = []
            
            # Generate predictions for next 7 days using CURRENT values
            # (Don't apply random changes unless you want forecasting with volatility)
            
            for day in range(1, 8):
                # Use the SAME values for all days (or apply small predictable changes if desired)
                # For a simple 7-day forecast based on current conditions:
                day_gold = gold_price_usd_per_oz
                day_usd = usd_rate
                
                # OPTIONAL: Add small trend-based changes (not random)
                # Remove or adjust these if you want completely static predictions
                if day > 1:
                    # Small predictable daily changes (e.g., +0.1% per day for gold, +0.05% for USD)
                    gold_trend = 0.001  # 0.1% daily increase
                    usd_trend = 0.0005  # 0.05% daily increase
                    day_gold = gold_price_usd_per_oz * (1 + (gold_trend * (day-1)))
                    day_usd = usd_rate * (1 + (usd_trend * (day-1)))

                # Prepare features - model expects USD per ounce
                features_df = pd.DataFrame([{
                    'USD_rate': day_usd,
                    'Inflation_rate': inflation_rate,
                    'gold_price_USD': day_gold,  # USD per ounce
                    'interest_rate': interest_rate,
                    'festivals': festivals,
                    'nepse_index': nepse_index
                }])

                # Use the SAME helper function as single prediction
                day_prediction, used_fallback = get_prediction_in_npr_per_tola(features_df)
                
                # Debug: Show what we're calculating
                gold_price_usd_per_tola = day_gold * 1.26
                direct_conversion = gold_price_usd_per_tola * day_usd
                min_reasonable = direct_conversion * 0.7
                
                print(f"Day {day}: Gold=${day_gold:.2f}/oz, USD={day_usd:.2f}, "
                      f"Direct={direct_conversion:.2f}, Model={day_prediction:.2f}")
                
                if day_prediction < min_reasonable:
                    print(f"  -> Clipping from {day_prediction:.2f} to {min_reasonable:.2f}")
                    day_prediction = min_reasonable
                    used_fallback = True
                
                future_date = datetime.now() + timedelta(days=day)
                predictions_7_days.append({
                    'day': day,
                    'prediction': round(float(day_prediction), 2),  # NPR per tola
                    'date': future_date.strftime('%Y-%m-%d'),
                    'gold_price_usd': round(day_gold, 2),
                    'usd_rate': round(day_usd, 2),
                    'used_fallback': used_fallback
                })
            
            print(f"Generated 7-day predictions: {predictions_7_days}")
            
            # Also calculate what the single prediction would be for comparison
            single_features_df = pd.DataFrame([{
                'USD_rate': usd_rate,
                'Inflation_rate': inflation_rate,
                'gold_price_USD': gold_price_usd_per_oz,
                'interest_rate': interest_rate,
                'festivals': festivals,
                'nepse_index': nepse_index
            }])
            single_prediction, _ = get_prediction_in_npr_per_tola(single_features_df)
            gold_price_usd_per_tola = gold_price_usd_per_oz / OZ_TO_TOLA
            direct_conversion = gold_price_usd_per_tola * usd_rate
            
            return JsonResponse({
                'success': True,
                'predictions_7_days': predictions_7_days,
                'unit': 'NPR per tola',
                'comparison': {
                    'single_prediction_today': round(single_prediction, 2),
                    'direct_conversion': round(direct_conversion, 2),
                    'initial_gold_usd_oz': gold_price_usd_per_oz,
                    'initial_usd_rate': usd_rate
                }
            })
        except Exception as e:
            print(f"7-day prediction error: {e}")
            return JsonResponse({
                'success': False,
                'error': f'Prediction error: {str(e)}'
            })
    return JsonResponse({
        'success': False,
        'error': 'Invalid request method'
    })

@csrf_exempt
@detect_request_spike()
def prediction_history(request):
    """Fetch prediction history for charting (last N predictions)"""
    print(f"[DEBUG] prediction_history called method={request.method} path={request.path}")
    if request.method == 'GET':
        try:
            # Get number of predictions to return (default 10, max 100)
            limit = int(request.GET.get('limit', 10))
            limit = min(limit, 100)
            
            # Fetch latest predictions ordered by timestamp
            predictions = Prediction.objects.all()[:limit]
            
            history_data = {
                'success': True,
                'predictions': []
            }
            
            for pred in reversed(list(predictions)):  # Reverse to show oldest first
                # Use date (YYYY-MM-DD) for chart labels instead of time
                history_data['predictions'].append({
                    'timestamp': pred.timestamp.strftime('%Y-%m-%d'),
                    'predicted_price': pred.predicted_price,
                    'usd_rate': pred.usd_rate,
                    'gold_price_usd': pred.gold_price_usd,
                    'model_used': pred.model_used
                })
            
            return JsonResponse(history_data)
        
        except Exception as e:
            print(f"Error fetching prediction history: {e}")
            return JsonResponse({'success': False, 'error': str(e)}, status=500)
    return JsonResponse({'success': False, 'error': 'Invalid request method'}, status=405)

# Dashboard Views
@detect_request_spike()
def prediction_dashboard(request):
    """Render the prediction dashboard"""
    print(f"[DEBUG] prediction_dashboard called method={request.method} path={request.path}")
    get_token(request)  # Ensure CSRF token is set
    return render(request, 'predictor/prediction.html', {
        'active_tab': 'prediction'
    })

@detect_request_spike()
def analytics_dashboard(request):
    """Render the analytics dashboard"""
    print(f"[DEBUG] analytics_dashboard called method={request.method} path={request.path}")
    return render(request, 'predictor/analytics.html', {
        'active_tab': 'analytics'
    })

@detect_request_spike()
def forecast_dashboard(request):
    """Render the forecast dashboard"""
    print(f"[DEBUG] forecast_dashboard called method={request.method} path={request.path}")
    get_token(request)
    return render(request, 'predictor/forecast.html', {
        'active_tab': 'forecast'
    })

@detect_request_spike()
def history_dashboard(request):
    """Render the history dashboard"""
    print(f"[DEBUG] history_dashboard called method={request.method} path={request.path}")
    return render(request, 'predictor/history.html', {
        'active_tab': 'history'
    })