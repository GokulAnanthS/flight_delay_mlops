FROM prom/prometheus:v3.0.1

COPY --chmod=755 entrypoint.sh /entrypoint.sh
COPY --chmod=644 prometheus.yml.template /etc/prometheus/prometheus.yml.template

EXPOSE 9090

ENTRYPOINT ["/entrypoint.sh"]
