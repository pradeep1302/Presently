from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import numpy as np
import cv2
import face_recognition
import face_recognition_models

app = FastAPI()

# Request model
class ImageURLRequest(BaseModel):
    url: str

@app.get("/")
async def hello():
    return {"message":"Hello fast api"}

# Function to download image from URL
def download_image_from_url(url: str) -> np.ndarray:
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            img_arr = np.asarray(bytearray(response.content), dtype=np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

            return img
        else:
            raise HTTPException(status_code=400, detail="Failed to download image")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error downloading image: {str(e)}")
    
    

# API endpoint for face encoding generation
@app.post("/generateEncoding/")
async def generate_face_encoding(request: ImageURLRequest):
    print('aa gya')

    image_url = request.url
    
    # Download image
    image = download_image_from_url(image_url)
    print(image.dtype)
    
    # Convert image from BGR (OpenCV default) to RGB (face_recognition requirement)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(rgb_image.dtype)
    
    # Find face locations and generate face encodings
    try:
        face_locations = face_recognition.face_locations(rgb_image)
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        if len(face_encodings) == 0:
            raise HTTPException(status_code=404, detail="No faces found in the image")
        
        # Return the encodings as a list of lists (convert NumPy arrays to lists)
        return {"face_encodings": [encoding.tolist() for encoding in face_encodings]}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the image: {str(e)}")

# Example: To run the server, use the following command in the terminal:
# uvicorn main:app --reload