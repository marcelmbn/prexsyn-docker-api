# prexsyn-docker-api

## Build the Docker image

```bash
docker buildx build -f Dockerfile.prexsyn -t prexsyn-api .
```

## Run the container (CPU)

```bash
docker run --rm -p 8011:8011 prexsyn-api
```

If the container exposes a different internal port (e.g. 8000), adapt:

```bash
docker run --rm -p 8011:8000 prexsyn-api
```

## Run with GPU support

Make sure NVIDIA Container Toolkit is installed on the host.

```bash
docker run --rm --gpus all -p 8011:8011 prexsyn-api
```

If using a different internal port:

```bash
docker run --rm --gpus all -p 8011:8000 prexsyn-api
```

## Test the API

From another terminal:

```bash
python test_api.py --base-url http://localhost:8011
```

The `--base-url` argument is optional (it defaults to `http://localhost:8011`).
