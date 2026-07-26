# Security Notes

## Dockerfile misconfiguration: container running as root (DS-0002)

### How it was found

Ran Trivy's config scanner against the Dockerfile to check for security
misconfigurations (separate from the image vulnerability/CVE scan):

```bash
docker run --rm -v $(pwd):/workspace aquasec/trivy:0.70.0 config /workspace
```

### Result

![Trivy Dockerfile scan result](../screenshots/trivy.png)

```
Dockerfile (dockerfile)
=======================
Tests: 27 (SUCCESSES: 26, FAILURES: 1)
Failures: 1 (UNKNOWN: 0, LOW: 0, MEDIUM: 0, HIGH: 1, CRITICAL: 0)

DS-0002 (HIGH): Specify at least 1 USER command in Dockerfile with non-root user as argument
════════════════════════════════════════
Running containers with 'root' user can lead to a container escape situation.
It is a best practice to run containers as non-root users, which can be done
by adding a 'USER' statement to the Dockerfile.
See https://avd.aquasec.com/misconfig/ds-0002
```

**Why it matters:** without a `USER` instruction, the container process runs
as root by default. If the app is ever compromised (e.g. via a dependency
vulnerability) or a container-escape bug is exploited, the attacker gets
root inside the container — and potentially root on the host, depending on
the escape.

### The fix

Added a dedicated non-root system user in the Dockerfile, granted it
ownership of `/app` (since the app needs to read/write `data/` and
`models/`), and switched to it before the container's runtime process
starts:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc curl libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/data /app/models

# Create a non-root user and switch to it
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && chown -R appuser:appuser /app

USER appuser

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```


### Verification

Rebuilt the image and re-ran the same scan:

```bash
docker build -t wine-mlops-test .
docker run --rm -v $(pwd):/workspace aquasec/trivy:0.70.0 config /workspace
```

result: `Tests: 27 (SUCCESSES: 27, FAILURES: 0)`.

Also confirmed the app still starts and runs correctly as the non-root user:

```bash
docker run --rm -p 8000:8000 wine-mlops-test
```