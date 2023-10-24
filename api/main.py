from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ml_model import retinanet

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://spectrum-animated-jasper.glitch.me"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/hello")
async def hello():
    return {"message": "hello RetinaNet!"}

@app.get("/obj_detection_mock")
async def hello():
    return \
        {
          "boxes": [
            [
              270.7120666503906,
              144.43141174316406,
              367.4526062011719,
              227.78172302246094
            ]
          ],
          "labels": [
              "person"
          ]
        }

@app.post("/obj_detection/")
async def obj_detection(file: UploadFile):
    try:
        image_data = await file.read()
        pred = retinanet.prediction(image_data)
        return JSONResponse(content=pred)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


