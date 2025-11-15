import uvicorn
from fastapi import FastAPI
from api.predict import router

app = FastAPI()
app.include_router(router)

@app.get("/api/healthcheck")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
