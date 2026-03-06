import uvicorn
from fastapi import FastAPI
from core.config import settings

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Research Agent API"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=settings.PORT)
