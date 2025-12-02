from django.urls import path
from . import views

app_name = 'predictor'

urlpatterns = [
    # Pages / dashboards
    path('', views.prediction_dashboard, name='prediction'),
    path('index/', views.index, name='index'),
    path('analytics/', views.analytics_dashboard, name='analytics'),
    path('forecast/', views.forecast_dashboard, name='forecast'),
    path('history/', views.history_dashboard, name='history'),

    # API endpoints used by frontend JS
    path('predict/', views.predict, name='predict'),
    path('prediction-history/', views.prediction_history, name='prediction_history'),
    # Map the frontend URL to the server API that actually fetches live data
    path('fetch-live-data/', views.fetch_live_data_api, name='fetch_live_data'),
    path('predict-7-days/', views.predict_7_days, name='predict_7_days'),
    path('auto-fetch-features/', views.fetch_live_data_api, name='auto_fetch_features'),

    # Model info pages (if used)
    path('model-info/', views.model_info, name='model_info'),
    path('model-comparison/', views.model_comparison, name='model_comparison'),
    path('model-details/', views.model_details, name='model_details'),
]