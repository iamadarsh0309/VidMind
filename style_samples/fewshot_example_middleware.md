## Middleware — request/response pipeline

- **Middleware** = code that runs *before* and *after* every request
- Stack order matters: first added = outermost layer (*onion model*)

### ASGI middleware vs FastAPI middleware
- `@app.middleware("http")` — simple, per-route-app level
- Starlette `BaseHTTPMiddleware` — class-based, good for logging/timing

### Typical uses
- Request ID / correlation header
- CORS (*Cross-Origin Resource Sharing* — browser security)
- Auth token check before route handler
- Timing: `start = time.time()` → log duration

```python
@app.middleware("http")
async def add_process_time(request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time"] = str(time.perf_counter() - start)
    return response
```

- `call_next(request)` = hand off to inner layers + route
- Mutate `response` on the way back out

**Key point:** middleware sees every request; keep it fast, no heavy DB work here
