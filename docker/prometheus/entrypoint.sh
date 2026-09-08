#!/bin/sh
set -e

# GRAFANA_REMOTE_WRITE_URL, GRAFANA_CLOUD_USER, GRAFANA_CLOUD_API_KEY,
# API_METRICS_TARGET are injected at deploy time via the ECS task definition
# (same convention as docker/mlflow-server). Rendered to /tmp rather than
# overwriting the file under /etc/prometheus, which the image's non-root
# user doesn't own.
sed \
  -e "s|__GRAFANA_REMOTE_WRITE_URL__|${GRAFANA_REMOTE_WRITE_URL}|g" \
  -e "s|__GRAFANA_CLOUD_USER__|${GRAFANA_CLOUD_USER}|g" \
  -e "s|__GRAFANA_CLOUD_API_KEY__|${GRAFANA_CLOUD_API_KEY}|g" \
  -e "s|__API_METRICS_TARGET__|${API_METRICS_TARGET}|g" \
  /etc/prometheus/prometheus.yml.template > /tmp/prometheus.yml

exec /bin/prometheus \
  --config.file=/tmp/prometheus.yml \
  --storage.tsdb.path=/prometheus \
  --web.listen-address=0.0.0.0:9090
