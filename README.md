📸 AI-Powered Image Captioning & Sentiment Analysis

🚀 Overview

This project utilizes AI models to generate captions for uploaded images and analyze their sentiment. It leverages:

BLIP (Bootstrapped Language-Image Pretraining) for caption generation.

OpenAI's GPT for enhancing the captions with more details.

VADER (Valence Aware Dictionary and sEntiment Reasoner) for sentiment analysis.

Streamlit & Flask for an interactive and user-friendly web UI.

✨ Features

✅ Upload images to generate AI-based captions.
✅ Enhance captions using OpenAI's GPT model.
✅ Perform sentiment analysis using VADER.
✅ Web UI built with Flask and Streamlit.
✅ Supports both CPU and GPU processing.

📂 Directory Structure

static/                  # Contains static files (CSS, JS, images)
__pycache__/             # Python cache files
.env                     # Environment variables (API keys)
base.html                # Base template for web pages
general.html             # Main webpage for uploading images
caption_generator.py     # Image caption generation logic
combined_app.py          # Main application entry point
requirements.txt         # Dependencies list
sentiment_ai.py          # Sentiment analysis logic
styles.css               # CSS for UI styling

🛠 Installation & Setup

1️⃣ Clone the Repository

git clone https://github.com/Jyothikainti/AI-Powered-Image-Captioning-Sentiment-Analysis.git
cd AI-Powered-Image-Captioning-Sentiment-Analysis

2️⃣ Set Up a Virtual Environment

python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows

3️⃣ Install Dependencies

pip install -r requirements.txt

4️⃣ Set Up Environment Variables

Create a .env file and add your OpenAI API key:

OPENAI_API_KEY=your_openai_api_key_here

▶️ Usage

Streamlit UI

streamlit run combined_app.py

Open your browser and go to http://localhost:8501/.

Flask UI

flask run

Access the app at http://127.0.0.1:5000/.

🛠 Troubleshooting

Long startup time? Ensure all dependencies are installed correctly.

CUDA issues? If using a GPU, ensure PyTorch is installed with CUDA support.

API key errors? Verify that your .env file contains a valid OpenAI API key.

Streamlit not running? Make sure you activated the virtual environment.

📜 License

This project is licensed under the MIT License.

📌 Repository: GitHub

