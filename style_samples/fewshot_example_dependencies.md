## Dependency Injection (DI) in FastAPI

- **DI** = FastAPI calls your function and *injects* shared objects (DB session, auth user, config)
- Replaces manual `db = get_db()` in every route → DRY (*Don't Repeat Yourself*)

### How it works
1. Write a dependency function: `def get_db(): ... yield session`
2. Declare in route: `def route(db: Session = Depends(get_db))`
3. FastAPI resolves the graph before your handler runs

### `Depends()` 
- Takes any callable
- Can depend on other dependencies (nested tree)
- `yield` pattern → cleanup after request (*like try/finally*)

### Common pattern — DB session
```python
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

| Concept | Meaning |
|---------|---------|
| `Depends(fn)` | "call fn and pass result here" |
| `yield` | setup before yield, teardown after |
| Sub-deps | `def auth(user=Depends(get_user))` |

**Recap:** DI = reusable setup injected per-request; `Depends` is the wiring syntax
