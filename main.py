from fastapi import FastAPI, HTTPException, File, UploadFile, Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import os
import google.generativeai as genai
from dotenv import load_dotenv
import logging
from logging.handlers import RotatingFileHandler
from gtts import gTTS
import cv2
import numpy as np
from paddleocr import PaddleOCR
import paddle

# Remove any existing log handlers
logger = logging.getLogger(__name__)
for handler in logger.handlers:
    logger.removeHandler(handler)

# Configure console logging
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Configure file logging
file_handler = RotatingFileHandler('app.log', maxBytes=10 * 1024 * 1024, backupCount=10)
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Set the log level for the application
logging.basicConfig(level=logging.DEBUG)

# Load environment variables from .env
load_dotenv()

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Get API key from environment variable
api_key = os.getenv("API_KEY")
if not api_key:
    raise ValueError("API_KEY environment variable is not set")

# Configure the API key
genai.configure(api_key=api_key)

ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)

def perform_ocr(image_bytes: bytes) -> str:
    try:
        # Read the image using OpenCV
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

        # Perform OCR on the image
        ocr_result = ocr.ocr(image, cls=True)

        # Extract text from OCR results
        extracted_text = ""
        for line in ocr_result:
            for char_info in line:
                extracted_text += char_info[1][0]

        return extracted_text
    except Exception as e:
        logger.error(f"Error during OCR: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred during OCR")

@app.post("/ocr/")
async def ocr_single_image(file: UploadFile = File(...)):
    try:
        # Read the uploaded image
        image_bytes = await file.read()

        # Perform OCR on the image
        extracted_text = perform_ocr(image_bytes)
        logger.info(f"OCR completed. Extracted text: {extracted_text}")

        return {"extracted_text": extracted_text}
    except Exception as e:
        logger.error(f"Error during image processing: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred during image processing")

def call_generative_model(image_data):
    try:
        # Load the GenerativeModel instance with the desired model - Gemini Pro Vision

        model = genai.GenerativeModel('gemini-1.5-flash')


        # Define the prompt for model generation
        prompt = "Give direct instructions for a blind person whose camera is sending you these images about the objects and their location. The image is taken from first person perspective of the blind. Give instructions by addressing the blind using 'You'. Keep it very short, possibly a single line with 4 or 5 words. The instructions should be in nepali. "

        # Generate content using the model, prompt, and image data
        response = model.generate_content(
            [prompt, {"mime_type": "image/png", "data": image_data}]
        )

        # Process and return the generated content
        generated_text = response.text
        return generated_text
    except Exception as e:
        logger.error(f"Error calling generative model: {e}", exc_info=True)
        raise

@app.post("/navigate/")
async def navigate(file: UploadFile = File(...)):
    try:
        # Read the uploaded image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # Call the generative model with the image bytes
        response = call_generative_model(image_bytes)
        logger.info("Generated instructions successfully.")
        return {"instructions": response}
    except ValueError as e:
        logger.error(f"Value error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Image.UnidentifiedImageError:
        logger.error("Unidentified image format", exc_info=True)
        raise HTTPException(status_code=415, detail="Unsupported image format")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred")

@app.post("/navigate/tts")
async def navigate_tts(file: UploadFile = File(...)):
    try:
        # Read the uploaded image
        image_bytes = await file.read()
        logger.debug("Received image for TTS processing.")

        # Call the generative model with the image bytes
        instructions = call_generative_model(image_bytes)
        logger.info("Generated instructions for TTS.")

        # Generate TTS audio file
        tts = gTTS(text=instructions, lang='en')
        audio_bytes_io = io.BytesIO()
        tts.write_to_fp(audio_bytes_io)

        # Convert the audio file to bytes
        audio_bytes = audio_bytes_io.getvalue()

        # Return the audio file as a response with download link
        logger.debug("Returning audio file.")
        return Response(
            audio_bytes,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "attachment; filename=output_audio.mp3"
            }
        )
    except ValueError as e:
        logger.error(f"Value error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Image.UnidentifiedImageError:
        logger.error("Unidentified image format", exc_info=True)
        raise HTTPException(status_code=415, detail="Unsupported image format")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)