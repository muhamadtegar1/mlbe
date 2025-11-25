# Docker Deployment Guide for Bank Marketing API

## Prerequisites
- Docker installed on your system
- Docker Compose (optional, but recommended)

## Build and Run with Docker

### Option 1: Using Docker directly

1. **Build the Docker image:**
   ```bash
   docker build -t bank-marketing-api .
   ```

2. **Run the container:**
   ```bash
   docker run -d -p 8000:8000 --name bank-marketing-api bank-marketing-api
   ```

3. **Check container status:**
   ```bash
   docker ps
   ```

4. **View logs:**
   ```bash
   docker logs bank-marketing-api
   ```

5. **Stop the container:**
   ```bash
   docker stop bank-marketing-api
   ```

6. **Remove the container:**
   ```bash
   docker rm bank-marketing-api
   ```

### Option 2: Using Docker Compose (Recommended)

1. **Start the service:**
   ```bash
   docker-compose up -d
   ```

2. **View logs:**
   ```bash
   docker-compose logs -f
   ```

3. **Stop the service:**
   ```bash
   docker-compose down
   ```

4. **Rebuild and restart:**
   ```bash
   docker-compose up -d --build
   ```

## PowerShell Commands

For Windows users using PowerShell:

```powershell
# Build image
docker build -t bank-marketing-api .

# Run container
docker run -d -p 8000:8000 --name bank-marketing-api bank-marketing-api

# View logs
docker logs bank-marketing-api -f

# Stop and remove
docker stop bank-marketing-api; docker rm bank-marketing-api

# Using Docker Compose
docker-compose up -d
docker-compose logs -f
docker-compose down
```

## Test the API

Once the container is running, test the endpoints:

1. **Health check:**
   ```bash
   curl http://localhost:8000/health
   ```

2. **API Documentation:**
   Open in browser: http://localhost:8000/docs

3. **Test prediction (Random Forest):**
   ```bash
   curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{
       "model": "random_forest",
       "data": [{
         "age": 35,
         "job": "technician",
         "marital": "married",
         "education": "university.degree",
         "default": "no",
         "housing": "yes",
         "loan": "no",
         "contact": "cellular",
         "month": "may",
         "day_of_week": "mon",
         "duration": 200,
         "campaign": 2,
         "pdays": 999,
         "previous": 0,
         "poutcome": "nonexistent",
         "emp_var_rate": 1.1,
         "cons_price_idx": 93.994,
         "cons_conf_idx": -36.4,
         "euribor3m": 4.857,
         "nr_employed": 5191.0
       }]
     }'
   ```

4. **PowerShell test:**
   ```powershell
   Invoke-WebRequest -Uri "http://localhost:8000/health" -Method GET
   ```

## Troubleshooting

### Container won't start
```bash
docker logs bank-marketing-api
```

### Port already in use
```bash
# Use a different port
docker run -d -p 9000:8000 --name bank-marketing-api bank-marketing-api
```

### Rebuild after code changes
```bash
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Check container resource usage
```bash
docker stats bank-marketing-api
```

## Environment Variables

You can pass environment variables to customize behavior:

```bash
docker run -d -p 8000:8000 \
  -e WORKERS=4 \
  --name bank-marketing-api \
  bank-marketing-api
```

## Production Considerations

1. **Use production ASGI server settings in Dockerfile:**
   - Adjust workers based on CPU cores
   - Enable access logs for monitoring
   - Set appropriate timeout values

2. **Add volume for logs:**
   ```yaml
   volumes:
     - ./logs:/app/logs
   ```

3. **Use environment-specific configurations:**
   - Create `.env` file for sensitive data
   - Use Docker secrets for production

4. **Set resource limits:**
   ```yaml
   deploy:
     resources:
       limits:
         cpus: '2'
         memory: 2G
   ```
