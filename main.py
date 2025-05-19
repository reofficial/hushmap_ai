import os
import tempfile
import uvicorn

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from google import genai

# Load API key
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI-API-KEY not found in environment")

client = genai.Client(api_key=api_key)

app = FastAPI()


@app.post("/describe")
async def describe_audio(file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        # Upload to Gemini
        gemini_file = client.files.upload(file=tmp_path)

        # Ask Gemini to describe
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                "You are auxiliary to a noise level measurement device. The device is constantly recording audio and the noise level is calculated and displayed in an interface displayed to the user. 30 seconds of audio will be sent to you. The device is placed in a typical urban environment. Analyze the audio files with respect to this setting. Your role is to describe the audio recording. Your description will be displayed in the interface. Do not answer in complete sentences. Keep it as concise as possible and explain the audio recording.",
                gemini_file,
            ],
        )

        return JSONResponse(content={"description": response.text})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(app,host="0.0.0.0", port=port)
