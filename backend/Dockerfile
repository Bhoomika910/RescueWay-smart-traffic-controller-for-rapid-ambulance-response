# RescueWay backend â€” FastAPI + Double DQN / tabular Q-learning inference server.
#
# Scope note: this image runs the live API server (main.py) only. It does
# NOT include the SUMO binary â€” main.py never imports traci at module load
# time (only backend/sumo/*.py, which are standalone training/evaluation
# scripts, do), so the production inference server has no SUMO dependency.
# See docker-compose.yml's comment for how to run SUMO-based training
# separately if needed.
FROM python:3.13-slim AS base

# Pyserial's dependency chain and torch's CPU wheel both need a C toolchain
# available for a subset of transitive builds even though the primary
# wheels are prebuilt; kept minimal and removed from the final layer isn't
# attempted here since this is a single-stage image â€” the size cost is
# accepted in exchange for build simplicity (this is an inference server
# image, not a size-optimized edge deployment).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first so `docker build` can cache this layer across code
# changes that don't touch requirements.txt.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# rl/, logs/, data/ hold generated artifacts (checkpoints, DB, log files)
# that .gitignore deliberately excludes from source control â€” the running
# container needs writable versions of these directories even when they're
# empty in the image.
RUN mkdir -p rl logs data sumo/checkpoints results

EXPOSE 10000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.getenv('PORT','10000') + '/health', timeout=4)" || exit 1

# Start FastAPI using Render's assigned port
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]
