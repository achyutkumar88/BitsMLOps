from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter

prediction_counter = Counter("prediction_requests_total", "Total prediction requests")

def setup_prometheus(app):
    Instrumentator().instrument(app).expose(app)
