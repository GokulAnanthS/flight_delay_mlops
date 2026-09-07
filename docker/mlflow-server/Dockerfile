FROM python:3.10-slim

# psycopg2-binary: talk to the RDS Postgres backend store.
# boto3: talk to S3 for artifact storage.
RUN pip install --no-cache-dir mlflow==3.15.1 psycopg2-binary boto3

EXPOSE 5000

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, ARTIFACT_BUCKET are
# injected at deploy time via the ECS task definition.
ENTRYPOINT ["/entrypoint.sh"]
