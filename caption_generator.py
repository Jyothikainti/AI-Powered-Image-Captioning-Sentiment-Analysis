import os
from dotenv import load_dotenv
from transformers import BlipProcessor, BlipForConditionalGeneration
import openai
import torch
from PIL import Image

# ✅ Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# ✅ Set device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Load models (cached for efficiency)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# ✅ Function to generate captions using BLIP
def generate_caption(image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

# ✅ Function to enhance captions with OpenAI
def enhance_caption_with_openai(caption):
    if not openai_api_key:
        return caption  # Return original caption if API key is missing

    try:
        openai.api_key = openai_api_key
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a professional AI caption enhancer."},
                {"role": "user", "content": f"Make this caption at least 5 lines long, rich in details, and highly descriptive:\n\n'{caption}'"}
            ]
        )
        return response['choices'][0]['message']['content'].strip()
    except openai.OpenAIError as e:
        return f"OpenAI API Error: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"
