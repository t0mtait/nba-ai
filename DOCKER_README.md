# Celtic AI - Docker Compose Setup

This project is now configured to run with Docker Compose.

## Prerequisites

- Docker
- Docker Compose

## Running the Application

### Start the application
```bash
docker-compose up -d
```

### View logs
```bash
docker-compose logs -f app
```

### Stop the application
```bash
docker-compose down
```

### Rebuild the image
```bash
docker-compose up -d --build
```

## Access the Application

The application will be available at:
- Web UI: http://localhost:8000
- API: http://localhost:8000/docs (Swagger documentation)

## Volumes

The following directories are mounted as volumes for persistence:
- `./data` - Training data
- `./models` - Trained models
- `./uploaded_games` - User-uploaded game files

## Environment Variables

The application runs with `PYTHONUNBUFFERED=1` to ensure real-time log output.

## Development

For development with hot-reload, you can modify the docker-compose.yml to use:
```yaml
command: ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
```

And ensure the entire project directory is mounted as a volume.
