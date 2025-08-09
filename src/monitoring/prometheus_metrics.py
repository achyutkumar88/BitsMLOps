from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter


name = "prediction_requests_total"
description = "Total prediction requests"
prediction_counter = Counter(name, description)


def setup_prometheus(app):
    Instrumentator().instrument(app).expose(app)
