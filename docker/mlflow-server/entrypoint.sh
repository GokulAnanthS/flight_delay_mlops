#!/bin/sh
set -e

exec mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri "postgresql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}" \
  --default-artifact-root "s3://${ARTIFACT_BUCKET}" \
  --allowed-hosts "*"
