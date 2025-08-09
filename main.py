# main.py

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import matplotlib.pyplot as plt
from textblob import TextBlob
import os
import uuid

# --- Core Logic from Phase 2 (reused) ---

# Create a directory to store the generated plot images
if not os.path.exists("plots"):
    os.makedirs("plots")

def analyze_sentiment(text: str) -> list:
    """
    Analyzes the sentiment of each sentence in a given text.
    Returns a list of polarity scores.
    """
    blob = TextBlob(text)
    polarities = [sentence.sentiment.polarity for sentence in blob.sentences]
    return polarities

def create_sentiment_plot(polarities: list, filename: str) -> str:
    """
    Generates a plot of the sentiment arc and saves it as a PNG file.
    Returns the path to the saved image.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(polarities, marker='o', linestyle='-', color='b')
    plt.title('Emotional Arc of the Text', fontsize=16)
    plt.xlabel('Text Segment (Sentence Number)', fontsize=12)
    plt.ylabel('Sentiment Polarity (-1 = Negative, 1 = Positive)', fontsize=12)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.grid(True)
    plt.ylim(-1.1, 1.1)
    plot_path = os.path.join("plots", filename)
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def get_structural_feedback(polarities: list) -> list:
    """
    Provides simple, rule-based feedback on the text's structure.
    """
    feedback = []
    if len(polarities) > 2:
        max_pol = max(polarities)
        min_pol = min(polarities)
        if (max_pol - min_pol) < 0.2:
            feedback.append("The emotional arc is a bit flat. Try adding more variation in tone to make it more engaging.")
            
    if len(polarities) > 0 and polarities[-1] < 0.1:
        feedback.append("The sentiment at the end is low. A stronger, more positive conclusion could make a better impression.")

    if not feedback:
        feedback.append("Great job! The narrative flow seems well-structured.")

    return feedback

# --- FastAPI Application ---

# Create a FastAPI instance
app = FastAPI(title="Narrative Flow AI", version="1.0.0")

# Define a Pydantic model for the request body
# This helps FastAPI validate the input automatically
class TextRequest(BaseModel):
    text: str

# Define the API endpoint to analyze narrative
@app.post("/analyze-narrative")
def analyze_narrative_endpoint(request: TextRequest):
    """
    Analyzes the sentiment and structure of a given text.
    Returns a JSON object with feedback and a link to the emotional arc plot.
    """
    text_to_analyze = request.text
    polarities = analyze_sentiment(text_to_analyze)
    feedback = get_structural_feedback(polarities)
    
    # Generate a unique filename for the plot
    filename = f"plot_{uuid.uuid4()}.png"
    plot_path = create_sentiment_plot(polarities, filename)
    
    # Construct the URL for the generated plot
    # This URL will only work when your server is running
    plot_url = f"/plots/{filename}"
    
    return {
        "analysis_summary": {
            "num_sentences": len(polarities),
            "average_polarity": sum(polarities) / len(polarities) if polarities else 0
        },
        "feedback": feedback,
        "emotional_arc_plot_url": plot_url
    }

# A separate endpoint to serve the generated plot images
# This is crucial so that the frontend can display the image
@app.get("/plots/{filename}")
def serve_plot(filename: str):
    return FileResponse(os.path.join("plots", filename))

# A root endpoint for a simple health check
@app.get("/")
def read_root():
    return {"message": "Narrative Flow AI is up and running!"}