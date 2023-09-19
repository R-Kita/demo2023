from fastapi import FastAPI
# from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/hello")
async def hello():
    return {"message": "hello world!"}

# @app.post("/upload/")
# async def upload_file(file: UploadFile):
#     try:
#         image_data = await file.read()
#         random_number = (100, 200)
#         return JSONResponse(content={"random_number": random_number})
#     except Exception as e:
#         return JSONResponse(content={"error": str(e)}, status_code=500)


