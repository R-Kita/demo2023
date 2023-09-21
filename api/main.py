from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ml_model import retinanet

app = FastAPI()

@app.get("/hello")
async def hello():
    return {"message": "hello world!"}

@app.post("/obj_detection/")
async def obj_detection(file: UploadFile):
    try:
        image_data = await file.read()
        pred = retinanet.prediction(image_data)
        return JSONResponse(content=pred)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


