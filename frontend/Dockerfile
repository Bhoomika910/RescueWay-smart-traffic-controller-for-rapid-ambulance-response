# RescueWay dashboard — static build served by nginx.
# Build-time VITE_API_BASE_URL bakes the backend URL into the compiled JS
# (Vite env vars are resolved at build time, not runtime) — pass it via
# `docker build --build-arg VITE_API_BASE_URL=...` or docker-compose.yml's
# build.args, matching whatever hostname the backend service is reachable
# at from the browser (not from inside the Docker network).
FROM node:20-slim AS build
WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
ARG VITE_API_BASE_URL=http://localhost:8000
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}
RUN npm run build

FROM nginx:1.27-alpine AS serve
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget -qO- http://localhost:80/ >/dev/null 2>&1 || exit 1

CMD ["nginx", "-g", "daemon off;"]
