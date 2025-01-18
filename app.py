from flask import Flask, render_template, request, redirect, url_for
import os
from dotenv import load_dotenv
import cv2
from ultralytics import YOLO
import google.generativeai as genai

# Load environment variables from .env
load_dotenv()

# Set up Google API Key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize the Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def get_gemini_response(image_data, prompt):
    """
    Get a response from the Gemini model.
    Updated to use the 'gemini-1.5-flash' model.
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content([image_data[0], prompt])
        return response.text
    except Exception as e:
        return f"Error in generating content: {e}"

def detect_acne(image_path):
    # Load the YOLOv8 model (replace with your own model path)
    model = YOLO(r'D:\Generative AI Preparation\langchain-crash-course-main\Genai Project\best.pt')

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    # Perform inference
    results = model(image)

    # Process results
    if isinstance(results, list) and len(results) > 0:
        for result in results:
            boxes = result.boxes.xyxy  # Get bounding boxes
            confidences = result.boxes.conf  # Get confidence scores
            classes = result.boxes.cls  # Get class indices

            for box, conf, cls in zip(boxes, confidences, classes):
                x1, y1, x2, y2 = box.tolist()  # Convert to list
                # Draw bounding box on the image
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

    # Save the output image
    output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output_image.jpg')
    cv2.imwrite(output_image_path, image)
    print(f"Image saved to {output_image_path}")
    return output_image_path

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files['image']
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Detect acne
            output_image_path = detect_acne(file_path)

            # Prepare image data for AI model
            with open(output_image_path, 'rb') as img_file:
                bytes_data = img_file.read()
                image_data = [{"mime_type": "image/jpeg", "data": bytes_data}]

            # Sample input prompt
            input_prompt = """
                You are an expert Dermatologist (for a college project prototype).
                You will receive input images of faces with marked acne.
                Based on the acne face map, provide insights into potential causes of the acne,
                along with information about possible deficiencies and related skin issues.
                Please note that this is a prototype project and does not provide medical advice.
                Offer general precautions and potential solutions for improving skin health 
                without implying any formal diagnosis or treatment.
            """

            response_text = get_gemini_response(image_data, input_prompt)

            return render_template("index.html", output_image=output_image_path, response=response_text)

    return render_template("index.html", output_image=None, response=None)

if __name__ == "__main__":
    app.run(debug=True)


