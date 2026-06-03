FROM node:20-alpine AS base
WORKDIR /app
COPY package*.json ./
RUN npm ci

# ── development ───────────────────────────────────────────────
FROM base AS development
COPY . .
CMD ["npm", "run", "dev"]

# ── production ────────────────────────────────────────────────
FROM base AS builder
ARG NEXT_PUBLIC_API_URL
ARG NEXT_PUBLIC_ML_URL
ARG NEXT_PUBLIC_WS_URL
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL
ENV NEXT_PUBLIC_ML_URL=$NEXT_PUBLIC_ML_URL
ENV NEXT_PUBLIC_WS_URL=$NEXT_PUBLIC_WS_URL
COPY . .
RUN npm run build

FROM node:20-alpine AS production
WORKDIR /app
ENV NODE_ENV=production
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public
CMD ["node", "server.js"]
