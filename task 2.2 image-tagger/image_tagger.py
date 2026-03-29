import os
import base64
import json
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

IMAGE_FOLDER = "images"
OUTPUT_FILE = "tags_output.json"

SUPPORTED_FORMATS = (".png", ".jpg", ".jpeg")


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


results = []

for filename in os.listdir(IMAGE_FOLDER):

    if not filename.lower().endswith(SUPPORTED_FORMATS):
        print(f"Skipping unsupported file: {filename}")
        continue

    path = os.path.join(IMAGE_FOLDER, filename)

    try:
        base64_image = encode_image(path)

        prompt = """
Analyze this advertising image and return JSON:
{
  "alt_text": "...",
  "tags": ["..."],
  "brand_safety_score": 1-10,
  "use_cases": ["..."]
}
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    ]
                }
            ]
        )

        output = response.choices[0].message.content
        data = json.loads(output)

        data["filename"] = filename
        results.append(data)

    except Exception as e:
        print(f"Error processing {filename}: {e}")


with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)

print("✅ Done! Results saved to tags_output.json")