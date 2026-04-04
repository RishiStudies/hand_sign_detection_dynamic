# Secrets Management Guide

This guide covers secure handling of secrets for the Hand Sign Detection system.

## Overview

**Never commit secrets to version control.** This includes:
- API keys
- Database passwords
- JWT secrets
- Cloud credentials
- Private certificates

## Environment Variables

### Required Secrets

| Variable | Description | Required In |
|----------|-------------|-------------|
| `TRAINING_API_KEY` | API key for training endpoints | Production |
| `REDIS_URL` | Redis connection string (may include password) | If using Redis |

### Local Development

1. Copy the example file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your local values:
   ```bash
   # Safe for local development only
   TRAINING_API_KEY=dev-key-not-for-production
   REDIS_URL=redis://localhost:6379/0
   ```

3. Never commit `.env` (already in `.gitignore`)

### Production Deployment

#### Option 1: Environment Variables (Recommended for containers)

```bash
# Docker run
docker run -e TRAINING_API_KEY="your-secure-key" \
           -e REDIS_URL="redis://:password@redis-host:6379/0" \
           hand-sign-backend

# Docker Compose
# Use .env file that's NOT in version control, or:
docker compose up -e TRAINING_API_KEY="your-secure-key"
```

#### Option 2: Docker Secrets (Swarm/Compose)

```yaml
# docker-compose.yml
services:
  backend:
    secrets:
      - training_api_key
    environment:
      TRAINING_API_KEY_FILE: /run/secrets/training_api_key

secrets:
  training_api_key:
    external: true
```

```bash
# Create the secret
echo "your-secure-key" | docker secret create training_api_key -
```

#### Option 3: Cloud Secret Managers

**AWS Secrets Manager:**
```python
import boto3
import json

def get_secret(secret_name: str) -> dict:
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response['SecretString'])

# Usage
secrets = get_secret('hand-sign-detection/prod')
os.environ['TRAINING_API_KEY'] = secrets['training_api_key']
```

**Azure Key Vault:**
```python
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

credential = DefaultAzureCredential()
client = SecretClient(vault_url="https://your-vault.vault.azure.net/", credential=credential)

secret = client.get_secret("training-api-key")
os.environ['TRAINING_API_KEY'] = secret.value
```

**Google Secret Manager:**
```python
from google.cloud import secretmanager

client = secretmanager.SecretManagerServiceClient()
name = "projects/your-project/secrets/training-api-key/versions/latest"
response = client.access_secret_version(name=name)
os.environ['TRAINING_API_KEY'] = response.payload.data.decode('UTF-8')
```

**HashiCorp Vault:**
```python
import hvac

client = hvac.Client(url='https://vault.example.com')
client.token = os.environ.get('VAULT_TOKEN')

secret = client.secrets.kv.v2.read_secret_version(path='hand-sign-detection')
os.environ['TRAINING_API_KEY'] = secret['data']['data']['training_api_key']
```

## GitHub Actions Secrets

For CI/CD pipelines, use GitHub repository secrets:

1. Go to Repository → Settings → Secrets and variables → Actions
2. Add secrets:
   - `TRAINING_API_KEY`
   - `DOCKER_REGISTRY_TOKEN`

3. Reference in workflows:
   ```yaml
   jobs:
     deploy:
       steps:
         - name: Deploy
           env:
             TRAINING_API_KEY: ${{ secrets.TRAINING_API_KEY }}
           run: ./deploy.sh
   ```

## Generating Secure Keys

```bash
# Generate a 32-character random key
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Or using openssl
openssl rand -base64 32

# Or using /dev/urandom
head -c 32 /dev/urandom | base64
```

## Security Checklist

### Before Committing

- [ ] No secrets in `.env` files that are tracked
- [ ] No hardcoded keys in source code
- [ ] No secrets in Docker build arguments
- [ ] `.env` is in `.gitignore`

### For Production

- [ ] `TRAINING_API_KEY` is set and strong (32+ chars)
- [ ] Redis password is set if exposed
- [ ] CORS origins are restricted to your domains
- [ ] Rate limiting is enabled
- [ ] TLS/HTTPS is enabled at the load balancer

### Scanning for Secrets

Use tools to detect accidental commits:

```bash
# Install git-secrets
brew install git-secrets  # macOS
# or
pip install detect-secrets

# Scan repository
git secrets --scan
# or
detect-secrets scan
```

Add pre-commit hook:
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.4.0
    hooks:
      - id: detect-secrets
```

## Rotating Secrets

### Procedure

1. Generate new secret
2. Update in secret manager/environment
3. Deploy with new secret
4. Invalidate old secret
5. Monitor for authentication failures

### Zero-Downtime Rotation

For APIs that support multiple keys:

```python
# Support both old and new keys during transition
VALID_KEYS = [
    os.environ.get('TRAINING_API_KEY'),
    os.environ.get('TRAINING_API_KEY_OLD'),  # Temporary
]

def verify_api_key(key: str) -> bool:
    return key in [k for k in VALID_KEYS if k]
```

## Incident Response

If a secret is exposed:

1. **Immediately rotate** the compromised secret
2. **Revoke** any issued tokens/sessions
3. **Audit** access logs for unauthorized use
4. **Remove** from git history if committed:
   ```bash
   # Use BFG Repo-Cleaner
   bfg --delete-files .env
   git push --force
   ```
5. **Notify** affected parties if data accessed

## Quick Reference

```bash
# Check for secrets in staged files
git diff --staged | grep -i -E "(key|secret|password|token)" 

# Verify .env is ignored
git check-ignore .env

# List environment variables (filter sensitive)
env | grep -v KEY | grep -v SECRET | grep -v PASSWORD
```
