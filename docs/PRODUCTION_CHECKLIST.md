# Production Readiness Checklist

Use this checklist to verify the system is ready for production deployment.

## Security ✓

- [ ] `TRAINING_API_KEY` is set to a strong value (minimum 32 characters, random)
- [ ] `TRAINING_API_KEY` is NOT stored in version control (use .env or secrets manager)
- [ ] `CORS_ORIGINS` is configured to match your frontend domain(s)
- [ ] CORS_ORIGINS does NOT include `*` (wildcard)
- [ ] All sensitive environment variables are injected at runtime (not hardcoded)
- [ ] `.env` file is in `.gitignore` and NOT committed
- [ ] HTTPS is enforced at load balancer / reverse proxy level
- [ ] API endpoints requiring training use API key validation (all `/train*` endpoints)

## API Functionality ✓

- [ ] `/health/live` returns 200 OK
- [ ] `/health/ready` returns 200 OK (all models available)
- [ ] `/health/details` shows expected backend status
- [ ] `/predict` inference works with sample image
- [ ] `/predict_sequence` works (or returns 501 if LSTM disabled)
- [ ] Rate limiting is active (test by sending >100 requests/min to `/predict`)
- [ ] CSV schema validation catches malformed CSVs
- [ ] Invalid API key returns 403 on training endpoints
- [ ] File upload size limits are enforced

## Model & Data ✓

- [ ] `models/shared_backend_state.json` is valid JSON
- [ ] `models/shared_backend_state.json` points to existing model files
- [ ] RandomForest model loads successfully
- [ ] RandomForest labels file is present and correct
- [ ] LSTM model loads successfully (or TensorFlow not available is expected)
- [ ] All data files referenced in `shared_backend_state.json` exist
- [ ] Data files are not world-writable (permissions set correctly)
- [ ] `.gitignore` includes `*.npy`, `*.h5`, `data/videos/`

## Infrastructure ✓

- [ ] Redis is running and accessible (if using distributed backend)
- [ ] Redis connection URL is configured correctly (`REDIS_URL`)
- [ ] Job queue (RQ) workers are running
- [ ] Sufficient disk space for models and logs
- [ ] Log rotation is configured (50MB per file, 5 backups)
- [ ] Logs directory exists and is writable
- [ ] Container has resource limits set (CPU, memory)

## Monitoring & Logging ✓

- [ ] `LOG_LEVEL` is set to `INFO` (or `WARNING` for production)
- [ ] `LOG_TO_FILE` is enabled and logs are being written
- [ ] Logs include timestamps and log levels
- [ ] Log files are rotated (not growing unbounded)
- [ ] Error logs are monitored/alerted (integration with logging service)
- [ ] API startup logs show "API Server initialized successfully"
- [ ] LSTM availability is logged at startup

## Performance ✓

- [ ] API response time for `/predict` is <200ms (p99)
- [ ] Memory usage is stable (not growing over time)
- [ ] CPU usage under load is <80%
- [ ] Worker/job queue is not backed up (job processing time <expected baseline)
- [ ] Rate limiter is not blocking legitimate traffic

## Database/State Management ✓

- [ ] Session/combo state persists correctly across requests
- [ ] Redis is backing session storage (if distributed)
- [ ] Combo predictions are cleared properly via `/clear_combos`
- [ ] Job status can be queried via `/jobs/{job_id}`

## Deployment Process ✓

- [ ] Environment variables are documented in `.env.example`
- [ ] Deployment process is documented (runbook exists)
- [ ] Manual tests pass before deploying
- [ ] Automated tests pass (if CI/CD configured)
- [ ] Deployment can be rolled back if needed
- [ ] Models are backed up before deployment

## Failure Scenarios ✓

- [ ] If Redis unavailable: in-memory fallback works
- [ ] If LSTM model missing: API gracefully returns 501
- [ ] If RF model missing: API returns 503 and refuses to start
- [ ] If training job fails: error is logged and user is notified (via job status)
- [ ] If file upload fails: clear error message returned to client

## Compliance ✓

- [ ] Privacy: Uploaded training data is not shared/logged inappropriately
- [ ] Audit: Unauthorized access attempts are logged (missing/invalid API keys)
- [ ] Data retention: Old model checkpoints and logs are archived/deleted as per policy
- [ ] Configuration: All security settings are reproducible and documented

---

## Sign-Off

| Role | Name | Date | Notes |
|------|------|------|-------|
| Engineer | | | |
| Security | | | |
| Operations | | | |

---

## Production Issues Discovered

| Date | Issue | Resolution | Root Cause |
|------|-------|-----------|-----------|
| | | | |

---

## Notes for Future Deployments

- 
- 
-
