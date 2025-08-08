# monitoring/prometheus_metrics.py

from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter

# Define custom Prometheus metrics
prediction_counter = Counter(
    "prediction_requests_total",
    "Total number of prediction requests received"
)

def setup_prometheus(app):
    Instrumentator().instrument(app).expose(app)
