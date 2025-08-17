import uvicorn

if __name__ == "__main__":
    print("Starting OptiTrade server...")
    uvicorn.run("app.main:app", host="0.0.0.0", port=2931)
