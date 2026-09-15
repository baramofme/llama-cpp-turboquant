# Standalone gate-proxy image (no volume mount).
# Build from repo root with beellama-boosts/gate_proxy_v2.py staged in context:
#   mkdir -p /tmp/opencode/gatectx && cp beellama-boosts/gate_proxy_v2.py /tmp/opencode/gatectx/
#   docker build -f beellama-boosts/gate-proxy.Dockerfile \
#     -t localhost:5000/baramofme/gate-proxy:v2 /tmp/opencode/gatectx
FROM docker.io/library/python:3.12-slim

ENV BONSAI_BASE=http://bonsai:8080
ENV GATE_PORT=1709
ENV GATE_MAX_RETRY=2
ENV GATE_BACKEND_RETRY=1

COPY gate_proxy_v2.py /app/gate_proxy_v2.py
COPY glossary_ko_en.json /app/glossary_ko_en.json
WORKDIR /app
EXPOSE 1709
ENTRYPOINT ["python3", "/app/gate_proxy_v2.py"]