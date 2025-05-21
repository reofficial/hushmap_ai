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
                "You are an AI assistant for a sound level measurement device. A 10-second audio recording from a typical urban environment is being described. Your role is to generate a concise description of the variety of distinct sound events likely present in this recording. This description will be displayed in a user interface. Instructions for Description: Identify multiple distinct sound sources. If it is traffic, you will hear engine noise. Otherwise, do not mention traffic. Go beyond just general traffic. Consider specific sounds like: Vehicles (cars, buses, motorcycles, sirens, horns) Human activity (voices, footsteps, laughter, distant shouts) Construction (drilling, hammering) Environmental (wind, distant PA announcements, occasional birds) Other urban sounds (shop doors, distant music) Focus on audible events, not just constant background hum (though a brief mention of background is okay if other distinct sounds are also listed). Output Format: Use keywords and very short phrases. Separate distinct sound observations with periods. NO complete sentences. Be extremely concise. Example Output (for guidance, do not copy verbatim): Road traffic. Distant siren. Voices nearby. Bus braking. Occasional horn. OR Piano sounds. Traffic hum. Construction sounds. Pedestrian chatter. Dog bark. Now, describe the audio.",
                gemini_file,
            ],
        )

        return JSONResponse(content={"description": response.text})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
@app.post("/summarize")
async def summarize(descriptions: str):
    try:
        prompt = (
                "The following text are descriptions of 30 second audio files collected in a stationary spot over about 9 hours. Summarize the following descriptions into 2-3 concise sentences to summarize the day:\n" + descriptions
                )

        response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[prompt],
            )

        return JSONResponse(content={"summary": response.text})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})




if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(app,host="0.0.0.0", port=port)
