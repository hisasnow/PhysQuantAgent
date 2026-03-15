import json
import os
import time
from google import genai
from google.genai import types
from PIL import Image
import io
from dotenv import load_dotenv
import json

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
load_dotenv()

# Specify an image-editing-capable model such as Nano Banana Pro (Gemini 3 Pro Image)
# Note: Model IDs may change depending on release status (e.g. gemini-2.0-flash-exp)
MODEL_NAME = "gemini-3-pro-image-preview" 

# Root directory for image paths (change if needed)
# Use an empty string if the paths in JSON are absolute,
# or specify the parent directory if they are relative.
BASE_DIR = "" 
JSON_DATA_PATH = "./Visphysquant/output.json"

# Output directory
OUTPUT_DIR = "./Visual_prompt_images/internal_image"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Wait time between API requests (seconds)
WAIT_TIME_SECONDS = 1

# ---------------------------------------------------------
# Main process
# ---------------------------------------------------------
def main():
    # Initialize client
    client = genai.Client(api_key=os.environ.get("GEMINI_API"))

    # Load user JSON data (stored in a file in this case)
    print(f"Loading dataset from '{JSON_DATA_PATH}'...")

    with open(JSON_DATA_PATH, encoding="utf-8") as f:
        json_data = json.load(f)

    for i, item in enumerate(json_data):
        # Get image paths (0th, 3rd, and 7th)
        if not item.get("images"):
            continue

        target_indices = [0, 3, 7]
        image_paths = []
        for idx in target_indices:
            if idx < len(item["images"]):
                image_paths.append(item["images"][idx])

        if not image_paths:
            continue
            
        # Use the first image to determine the output filename
        first_image_path = image_paths[0]
        full_first_path = os.path.join(BASE_DIR, first_image_path)

        if "/processed_data/" in full_first_path:
            extracted_id = full_first_path.split("/processed_data/")[1].split("/")[0]
        else:
            raise ValueError(f"Unsupported image path format: {full_first_path}")

        # Precompute save path and check for duplicates
        filename = os.path.basename(full_first_path)
        
        save_name = f"{extracted_id}_sliced.png"
        save_path = os.path.join(OUTPUT_DIR, save_name)

        if os.path.exists(save_path):
            print(f"Skipping (already exists): {save_path}")
            continue

        print(f"Processing: {image_paths}")

        try:
            contents = []
            for img_path in image_paths:
                full_path = os.path.join(BASE_DIR, img_path)
                # Read image file as bytes
                with open(full_path, "rb") as f:
                    image_bytes = f.read()
                
                # Simple MIME type detection
                mime_type = "image/jpeg" if full_path.lower().endswith(('.jpg', '.jpeg')) else "image/png"
                contents.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))

            # Create prompt: instruct the model to slice the object and reveal internal structure
            prompt = (
                "Based on the input image, create a technical split-screen diagram arranged in a 2x2 grid layout.\n"
                "The top-right panel must show the original full appearance of the object from an isometric angle.\n" 
                "The other three panels (top-left, bottom-left, bottom-right) should display precise cross-section slices of the same object from the front, side, and top views, revealing its internal structure.\n" 
                "Output image should be the same realistic style as the input image."
                )
            contents.append(prompt)

            # Send API request
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"], # Specify image output
                    safety_settings=[ # Adjust safety settings as needed
                         types.SafetySetting(
                             category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                             threshold="BLOCK_ONLY_HIGH"
                         )
                    ]
                )
            )

            # Save generated image
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if part.inline_data:
                        # Decode and save image data
                        generated_image = Image.open(io.BytesIO(part.inline_data.data))
                        
                        generated_image.save(save_path)
                        print(f"Saved: {save_path}")
                        
        except Exception as e:
            print(f"Error processing {full_path}: {e}")
        
        # Wait before next request
        #print(f"Waiting for {WAIT_TIME_SECONDS} seconds before next request...")
        time.sleep(WAIT_TIME_SECONDS)

if __name__ == "__main__":
    main()
