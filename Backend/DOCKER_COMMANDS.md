# Docker Commands for Backend

## Restart Backend Service

### Option 1: Restart only backend service
```bash
docker compose restart backend
```

### Option 2: Stop and start backend
```bash
docker compose stop backend
docker compose start backend
```

### Option 3: Rebuild and restart (if code changed)
```bash
docker compose up -d --build backend
```

### Option 4: Full restart (stop, rebuild, start)
```bash
docker compose down backend
docker compose up -d --build backend
```

## View Backend Logs

```bash
# View logs
docker compose logs backend

# Follow logs (real-time)
docker compose logs -f backend

# View last 100 lines
docker compose logs --tail=100 backend
```

## Check Backend Status

```bash
# Check if backend is running
docker compose ps backend

# Check backend health
curl http://localhost:8080/api/health
```

## Common Commands

```bash
# Restart all services
docker compose restart

# Stop all services
docker compose down

# Start all services
docker compose up -d

# View all logs
docker compose logs -f
```

