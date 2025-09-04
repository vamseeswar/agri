import os
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
from gtts import gTTS
import random
import string

from groq import Groq
from dotenv import load_dotenv
from flask_cors import CORS

# -------------------- Load Environment --------------------
load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")

if not api_key or api_key.strip() == "":
    raise ValueError("❌ Groq API key not found! Please set GROQ_API_KEY in your .env file.")

# -------------------- Initialize Groq --------------------
groq_client = Groq(api_key=api_key)

# -------------------- Flask App Setup --------------------
app = Flask(__name__)
CORS(app)  # Allow frontend requests from any origin
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'webm', 'wav', 'mp3', 'm4a'}

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs("static/audio", exist_ok=True)

# -------------------- Helper Functions --------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def transcribe_audio_groq(filepath):
    try:
        with open(filepath, "rb") as f:
            response = groq_client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=f,
            )
            return response.text
    except Exception as e:
        return f"❌ Audio transcription failed: {str(e)}"

def get_answer_groq(question):
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # use smaller model if 70b fails
            messages=[
                {"role": "system", "content": "You are a helpful agriculture chatbot for Indian farmers."},
                {"role": "user", "content": question}
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Groq API error: {str(e)}"

def text_to_audio(text, filename):
    try:
        tts = gTTS(text)
        audio_path = os.path.join("static/audio", f"{filename}.mp3")
        tts.save(audio_path)
        return audio_path
    except Exception as e:
        return None

# -------------------- Flask Routes --------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    # Handle audio input
    if 'audio' in request.files:
        audio = request.files['audio']
        if audio and allowed_file(audio.filename):
            filename = secure_filename(audio.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            audio.save(filepath)
            transcription = transcribe_audio_groq(filepath)
            answer = get_answer_groq(transcription)
            voice_filename = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
            audio_path = text_to_audio(answer, voice_filename)
            return jsonify({
                'text': f"🎤 Transcribed: {transcription}\n\n🤖 Answer: {answer}",
                'voice': url_for('static', filename=f'audio/{voice_filename}.mp3') if audio_path else None
            })

    # Handle text input
    elif 'text' in request.form:
        question = request.form['text']
        answer = get_answer_groq(question)
        voice_filename = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
        audio_path = text_to_audio(answer, voice_filename)
        return jsonify({
            'text': answer,
            'voice': url_for('static', filename=f'audio/{voice_filename}.mp3') if audio_path else None
        })

    return jsonify({'text': 'No valid input found'}), 400

# -------------------- Run App --------------------

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
