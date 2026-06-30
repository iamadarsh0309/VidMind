## FastAPI — Routing basics

- **Route** = URL path + HTTP method the server listens on
- `@app.get("/")` → decorator (*wraps a function to register it as a handler*)
- Handler function must be `async def` or `def` — FastAPI supports both

### Path params
- `/items/{item_id}` — curly braces = dynamic segment
- Type hint `item_id: int` → auto validation + conversion (*Pydantic under the hood*)

### Query params
- Optional filters: `/items/?skip=0&limit=10`
- Default values in function signature = query param

> Pitfall: path param names must match function arg names exactly

### Mini example (learning snippet only)
```python
from fastapi import FastAPI
app = FastAPI()

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}
```

**Takeaway:** decorator → path → typed params = full route definition in 4 lines
