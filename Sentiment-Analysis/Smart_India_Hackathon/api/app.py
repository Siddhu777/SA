import os
import nltk
import speech_recognition as sr
from nltk.sentiment import SentimentIntensityAnalyzer
from flask import Flask, request, jsonify, render_template
from signalwire.rest import Client as SignalWireClient
from signalwire.voice_response import VoiceResponse

# Initialize Flask app
app = Flask(__name__)

# Initialize NLTK and SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# SignalWire credentials
SIGNALWIRE_PROJECT_ID = '7348f7d7-fb13-46e3-b822-c6d7a1522fe6'
SIGNALWIRE_API_TOKEN = 'PT569bdfdd0e264f8b20fc69c15d54fc573ab5702561513cbc'
SIGNALWIRE_SPACE_URL = 'phcet.signalwire.com'
client = SignalWireClient(SIGNALWIRE_PROJECT_ID, SIGNALWIRE_API_TOKEN, signalwire_space_url=SIGNALWIRE_SPACE_URL)
# Function to convert audio to text and get sentiment score
def analyze_sentiment(audio_file):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
        text = recognizer.recognize_google(audio)
        sentiment_scores = sia.polarity_scores(text)
        return text, sentiment_scores
    except sr.UnknownValueError:
        return None, {'compound': 0}
    except sr.RequestError:
        return None, {'compound': 0}

# SignalWire webhook to handle incoming calls
@app.route('/incoming_call', methods=['POST'])
def incoming_call():
    response = VoiceResponse()
    response.say("Please leave a message after the beep. Press any key when finished.")
    response.record(max_length=60, action='/handle_recording', transcribe=False)
    return str(response)

# Handle the recording, download the file, and analyze sentiment
@app.route('/handle_recording', methods=['POST'])
def handle_recording():
    recording_url = request.form['RecordingUrl']

    # Download the recorded audio file from SignalWire
    audio_file = download_audio(recording_url)

    # Analyze the sentiment and transcribe text
    transcribed_text, sentiment_scores = analyze_sentiment(audio_file)

    sentiment = None
    sentiment_score = sentiment_scores['compound'] if sentiment_scores else 0

    if sentiment_score >= 0.05:
        sentiment = 'positive'
    elif sentiment_score <= -0.05:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'

    # Return the results
    return render_template('index.html', sentiment_score=sentiment, transcribed_text=transcribed_text)

# Function to download the audio from SignalWire
def download_audio(recording_url):
    audio_response = client.recordings(recording_url).fetch()
    temp_audio_file = 'temp_audio.wav'

    with open(temp_audio_file, 'wb') as f:
        f.write(audio_response.media)

    return temp_audio_file

# Define route for the homepage
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
