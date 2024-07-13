from fastapi import FastAPI, UploadFile, File, HTTPException
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tomatopredictor = tf.keras.models.load_model("./model_inception.h5")
potatopredictor=tf.keras.models.load_model("./potatoes.h5")
bananapredictor=tf.keras.models.load_model("./BananaLeaf_classifier.h5")
cornpredictor=tf.keras.models.load_model("./cornmodel.h5")
class_names_tomato = ["Bacterial_spot",
 "Early_blight","Late_blight","Leaf_Mold",
 "Target_Spot","Septoria_leaf_spot",
 "Spider_mites Two-spotted_spider_mite","Tomato_Yellow_Leaf_Curl_Virus",
 "Tomato_mosaic_virus","Healthy"]
class_names_potato=["Potato__Early_blight","Potato_Late_blight","Potato__healthy"]
class_names_banana=['Banana Bacterial Wilt','Black sigatoka disease','Healthy']
class_names_corn=['corn_Blight', 'Common_Rust', 'corn_Gray_Leaf_Spot', 'Healthy']
@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict/tomato")
async def predict_All_plantdisease(file: UploadFile = File(...)):
    try:
        img = Image.open(BytesIO(await file.read()))
        img = img.resize((224, 224)) 
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        plant_disease_predictions = tomatopredictor.predict(img_array)
        predicted_disease_class = class_names_tomato[np.argmax(plant_disease_predictions)]

        return {"class": predicted_disease_class, "predictions": plant_disease_predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during plant disease prediction: {e}")
@app.post("/predict/potato")
async def potato_predictor(file: UploadFile = File(...)):
    try:
        img = Image.open(BytesIO(await file.read()))
        img = img.resize((256, 256))  
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict plant disease
        plant_disease_predictions = potatopredictor.predict(img_array)
        predicted_disease_class = class_names_potato[np.argmax(plant_disease_predictions)]

        return {"class": predicted_disease_class, "predictions": plant_disease_predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during plant disease prediction: {e}")
@app.post("/predict/banana")
async def potato_predictor(file: UploadFile = File(...)):
    try:
        img = Image.open(BytesIO(await file.read()))
        img = img.resize((100,100))  
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict plant disease
        plant_disease_predictions = bananapredictor.predict(img_array)
        predicted_disease_class = class_names_banana[np.argmax(plant_disease_predictions)]

        return {"class": predicted_disease_class, "predictions": plant_disease_predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during plant disease prediction: {e}")

@app.post("/predict/corn")
async def potato_predictor(file: UploadFile = File(...)):
    try:
        img = Image.open(BytesIO(await file.read()))
        img = img.resize((256,256))  
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        plant_disease_predictions = cornpredictor.predict(img_array)
        predicted_disease_class = class_names_corn[np.argmax(plant_disease_predictions)]

        return {"class": predicted_disease_class, "predictions": plant_disease_predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during plant disease prediction: {e}")

if _name_ == "_main_":
    import uvicorn
    uvicorn.run(app, host='localhost', port=1000)
    uvicorn.run(app,host='localhost',port=1000)
    uvicorn.run(app,host='localhost',port=1000)
    uvicorn.run(app,host='localhost',port=1000)