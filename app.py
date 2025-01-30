import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image
from caption_generator import generate_caption, enhance_caption_with_openai
from sentiment_ai import analyze_sentiment

app = Flask(__name__)

# ✅ Set upload folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        file = request.files["image"]
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Process the image
            image = Image.open(filepath).convert("RGB")
            initial_caption = generate_caption(image)
            enhanced_caption = enhance_caption_with_openai(initial_caption)
            sentiment_label = analyze_sentiment(enhanced_caption)

            # Send results to the template
            result = {
                "image_url": filepath,
                "initial_caption": initial_caption,
                "enhanced_caption": enhanced_caption,
                "sentiment": {"label": sentiment_label}
            }

            return render_template("general.html", result=result)
    
    return render_template("general.html", result=None)

if __name__ == "__main__":
    app.run(debug=True,use_reloader=False)
