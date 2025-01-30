import os
import streamlit as st
from dotenv import load_dotenv
from transformers import BlipProcessor, BlipForConditionalGeneration
import openai
import torch
from PIL import Image
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ✅ Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# ✅ Set device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Load models (cached for efficiency)
@st.cache_resource
def load_models():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    return processor, model

processor, model = load_models()

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

# ✅ Custom Sentiment Analysis using VADER (Adjusts for Aggression)
def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()

    # Custom word weighting for aggression
    aggression_words = ["fight", "battle", "attack", "claws", "ferocity", "roar", "dominance", "aggressive", "wild", "brutal"]
    for word in aggression_words:
        if word in text.lower():
            return "NEGATIVE"  # Force negative sentiment for aggressive descriptions

    sentiment_score = analyzer.polarity_scores(text)['compound']
    
    if sentiment_score >= 0.05:
        return "POSITIVE"
    elif sentiment_score <= -0.05:
        return "NEGATIVE"
    else:
        return "NEUTRAL"

# ✅ Streamlit UI
st.title("📸 AI-Powered Image Captioning & Sentiment Analysis")
st.write("Upload an image, and the AI will generate a caption, enhance it, and analyze its sentiment.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Generate caption
    with st.spinner("🔍 Generating caption..."):
        caption = generate_caption(image)

    # Enhance caption (at least 5 lines)
    with st.spinner("✨ Enhancing caption..."):
        enhanced_caption = enhance_caption_with_openai(caption)

    # Perform sentiment analysis using VADER
    with st.spinner("📊 Analyzing sentiment..."):
        sentiment_label = analyze_sentiment(enhanced_caption)

    # Display results
    st.subheader("📌 Results")
    st.write(f"**Original Caption:** {caption}")
    st.write(f"**Enhanced Caption:**\n{enhanced_caption}")

    # Display sentiment label with emoji (without score)
    sentiment_color = "🟢" if sentiment_label == "POSITIVE" else "🔴" if sentiment_label == "NEGATIVE" else "⚪"
    st.write(f"**Sentiment:** {sentiment_color} {sentiment_label}")

    # Style enhancements using Markdown
    st.markdown(
        """
        <style>
        .stApp { background-color: #f4f4f4; }
        .stMarkdown { font-size: 18px; font-weight: bold; color: #333; }
        </style>
        """,
        unsafe_allow_html=True,
    )
