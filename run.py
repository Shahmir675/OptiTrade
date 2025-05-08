import uvicorn

if __name__ == "__main__":
    print("Starting OptiTrade server...")
    uvicorn.run("app.main:app", host="127.0.0.1", port=8001, reload=True)