import time

from flask import request
from prometheus_client import CollectorRegistry, Counter, Histogram, multiprocess

registry = CollectorRegistry()
multiprocess.MultiProcessCollector(registry=registry)

REQUEST_COUNT = Counter('request_count', 'Total number of requests', ['method', 'endpoint', 'http_status'],
                        registry=registry)
REQUEST_LATENCY = Histogram('request_latency_seconds', 'Request latency in seconds',
                            ['method', 'endpoint', 'http_status'], registry=registry)

# Create a histogram metric to track the duration of function executions
FUNCTION_DURATION = Histogram('function_duration_seconds', 'Time spent processing the function', ['function_name'])


def before_request():
    request.start_time = time.time()


def after_request(response):
    latency = time.time() - request.start_time
    path = request.url_rule.rule
    REQUEST_COUNT.labels(method=request.method, endpoint=path, http_status=response.status_code).inc()
    REQUEST_LATENCY.labels(method=request.method, endpoint=path, http_status=response.status_code).observe(latency)
    return response


def init_metrics(app):
    with app.app_context():
        app.before_request(before_request)

        app.after_request(after_request)