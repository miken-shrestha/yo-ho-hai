from django.db import models
import json

class Prediction(models.Model):
    """Store gold price predictions with input features"""
    timestamp = models.DateTimeField(auto_now_add=True)
    predicted_price = models.FloatField()
    
    # Input features
    usd_rate = models.FloatField()
    inflation_rate = models.FloatField()
    gold_price_usd = models.FloatField()
    interest_rate = models.FloatField()
    festivals = models.IntegerField()
    nepse_index = models.FloatField()
    
    # Model metadata
    model_used = models.CharField(max_length=100, default='Linear Regression')
    model_accuracy = models.FloatField(null=True, blank=True)
    
    class Meta:
        ordering = ['-timestamp']
        verbose_name_plural = "Predictions"
    
    def __str__(self):
        return f"Prediction: NPR {self.predicted_price:.2f} at {self.timestamp}"
    
    def to_dict(self):
        """Convert prediction to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'predicted_price': self.predicted_price,
            'usd_rate': self.usd_rate,
            'inflation_rate': self.inflation_rate,
            'gold_price_usd': self.gold_price_usd,
            'interest_rate': self.interest_rate,
            'festivals': self.festivals,
            'nepse_index': self.nepse_index,
            'model_used': self.model_used,
            'model_accuracy': self.model_accuracy
        }
