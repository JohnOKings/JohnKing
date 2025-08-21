import json
import random
import numpy as np
import os
# Disable oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Now import TensorFlow
import tensorflow as tf
from tensorflow.keras.models import load_model
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash
from textblob import TextBlob
import os
import subprocess
import threading
import time
import shutil
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with your own secret key

# Global variables with thread safety
model_lock = threading.Lock()
model = None
words = None
classes = None
intents = None
model_version = "1.0"
last_trained = "Not trained yet"

# Initialize the model
def load_chatbot_model():
    global model, words, classes, intents, model_version, last_trained
    
    with model_lock:
        try:
            model = load_model('chatbot_model.h5')
            words = pickle.load(open('words.pkl', 'rb'))
            classes = pickle.load(open('classes.pkl', 'rb'))
            intents = json.loads(open('intents.json').read())
            
            # Try to load version info
            try:
                with open('model_version.txt', 'r') as f:
                    version_info = f.read().splitlines()
                    model_version = version_info[0] if version_info else "1.0"
                    last_trained = version_info[1] if len(version_info) > 1 else "Unknown"
            except:
                model_version = "1.0"
                last_trained = "Unknown"
                
            print(f"Model loaded successfully. Version: {model_version}, Last trained: {last_trained}")
        except Exception as e:
            print(f"Error loading model: {e}")
            # Try to load backup if available
            try:
                print("Attempting to load backup model...")
                model = load_model('backup/chatbot_model.h5')
                words = pickle.load(open('backup/words.pkl', 'rb'))
                classes = pickle.load(open('backup/classes.pkl', 'rb'))
                intents = json.loads(open('backup/intents.json').read())
                print("Backup model loaded successfully")
            except Exception as backup_e:
                print(f"Backup load failed: {backup_e}")
                model = None
                words = None
                classes = None
                intents = None

# Load model on startup
load_chatbot_model()

lemmatizer = WordNetLemmatizer()

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"found in bag: {w}")
    return np.array(bag)

def predict_class(sentence, model, words, classes):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    result = "I'm not sure how to respond to that."
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'admin' and password == 'password':  # Replace with your own credentials
            session['logged_in'] = True
            return redirect(url_for('admin'))
        else:
            flash('Invalid credentials. Please try again.')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

@app.route('/admin')
def admin():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    # Get stats for admin dashboard
    intent_count = len(intents['intents']) if intents else 0
    
    # Count unknown intents
    unknown_count = 0
    try:
        if os.path.exists('unrecognized_messages.json'):
            with open('unrecognized_messages.json', 'r') as f:
                unknown_data = json.load(f)
                unknown_count = len(unknown_data['intents'])
    except:
        pass
    
    return render_template('admin.html', 
                           intent_count=intent_count,
                           unknown_count=unknown_count,
                           model_version=model_version,
                           last_trained=last_trained)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get("message")
        if not user_message:
            return jsonify({"response": "Message not provided."}), 400

        with model_lock:
            if model is None or intents is None or words is None or classes is None:
                return jsonify({"response": "Server resources not properly loaded. Please try again later."}), 500

            # Sentiment analysis
            sentiment = TextBlob(user_message).sentiment.polarity

            if sentiment > 0:
                sentiment_response = "I'm glad you're feeling positive!"
            elif sentiment < 0:
                sentiment_response = "I'm sorry to hear that you're feeling negative."
            else:
                sentiment_response = "."

            ints = predict_class(user_message, model, words, classes)
            if not ints:
                handle_unrecognized_message(user_message)
                return jsonify({"response": "I'm not sure how to respond to that. Could you rephrase?"})

            response = get_response(ints, intents)
            full_response = f"{sentiment_response} {response}"
            return jsonify({"response": full_response})
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({"response": "An internal error occurred. Please try again later."}), 500

def handle_unrecognized_message(message):
    try:
        file_path = 'unrecognized_messages.json'
        print(f"Handling unrecognized message: {message}")
        
        # Create file if doesn't exist
        if not os.path.exists(file_path):
            with open(file_path, 'w') as file:
                json.dump({"intents": []}, file)
        
        with open(file_path, 'r+') as file:
            data = json.load(file)
            
            # Check if we already have this pattern
            pattern_exists = False
            for intent in data['intents']:
                if message in intent['patterns']:
                    pattern_exists = True
                    break
            
            if not pattern_exists:
                # Add to existing unrecognized intent or create new
                found = False
                for intent in data['intents']:
                    if intent['tag'] == 'unrecognized':
                        intent['patterns'].append(message)
                        found = True
                        break
                
                if not found:
                    new_entry = {
                        "tag": "unrecognized",
                        "patterns": [message],
                        "responses": [""]
                    }
                    data['intents'].append(new_entry)
                
                file.seek(0)
                json.dump(data, file, indent=4)
                file.truncate()
                print(f"Message '{message}' added to unrecognized intents.")
            else:
                print(f"Message '{message}' already exists in unrecognized intents.")
    except Exception as e:
        print(f"Error in handle_unrecognized_message: {e}")

@app.route('/log_unrecognized', methods=['POST'])
def log_unrecognized():
    try:
        message = request.json.get('message')
        if not message:
            return jsonify({"status": "Message not provided."}), 400
        handle_unrecognized_message(message)
        return jsonify({"status": "logged"})
    except Exception as e:
        print(f"Error in log_unrecognized endpoint: {e}")
        return jsonify({"status": "An internal error occurred. Please try again later."}), 500

@app.route('/get_unknown_intents', methods=['GET'])
def get_unknown_intents():
    try:
        file_path = 'unrecognized_messages.json'
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                data = json.load(file)
                return jsonify(data)
        else:
            return jsonify({"intents": []})
    except Exception as e:
        print(f"Error in get_unknown_intents endpoint: {e}")
        return jsonify({"status": "An internal error occurred. Please try again later."}), 500

@app.route('/update_intents', methods=['POST'])
def update_intents():
    try:
        new_intents = request.json.get('intents')
        if not new_intents:
            return jsonify({"status": "No intents provided."}), 400

        with open('intents.json', 'r+') as file:
            data = json.load(file)
            data['intents'].extend(new_intents)
            file.seek(0)
            json.dump(data, file, indent=4)
            file.truncate()

        # Remove the updated intents from unrecognized_messages.json
        if os.path.exists('unrecognized_messages.json'):
            with open('unrecognized_messages.json', 'r+') as file:
                unrecognized_data = json.load(file)
                updated_patterns = [pattern for intent in new_intents for pattern in intent['patterns']]
                
                # Filter out updated patterns
                for intent in unrecognized_data['intents']:
                    if intent['tag'] == 'unrecognized':
                        intent['patterns'] = [p for p in intent['patterns'] if p not in updated_patterns]
                
                # Remove empty intents
                unrecognized_data['intents'] = [
                    intent for intent in unrecognized_data['intents'] 
                    if intent['tag'] != 'unrecognized' or intent['patterns']
                ]
                
                file.seek(0)
                json.dump(unrecognized_data, file, indent=4)
                file.truncate()

        return jsonify({"status": "Intents updated and unrecognized messages cleaned."})
    except Exception as e:
        print(f"Error in update_intents endpoint: {e}")
        return jsonify({"status": "An internal error occurred. Please try again later."}), 500

def create_backup():
    """Create a backup of model and data files"""
    backup_dir = "backup"
    os.makedirs(backup_dir, exist_ok=True)
    
    files_to_backup = [
        'chatbot_model.h5',
        'words.pkl',
        'classes.pkl',
        'intents.json',
        'unrecognized_messages.json'
    ]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_folder = os.path.join(backup_dir, f"backup_{timestamp}")
    os.makedirs(backup_folder, exist_ok=True)
    
    for file in files_to_backup:
        if os.path.exists(file):
            shutil.copy(file, os.path.join(backup_folder, file))
    
    return backup_folder

@app.route('/backup_data', methods=['POST'])
def backup_data():
    try:
        backup_path = create_backup()
        return jsonify({
            "status": "success",
            "message": f"Backup created successfully at {backup_path}"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Backup failed: {str(e)}"
        }), 500

# Route 4: Retrain chatbot
@app.route("/retrain", methods=["POST"])
def retrain():
    try:
        # Create backup before retraining
        backup_path = create_backup()
        print(f"Created backup at: {backup_path}")
        
        # Run training
        result = subprocess.run(
            ['python', 'train_chatbot.py'], 
            capture_output=True, 
            text=True
        )
        
        if result.returncode != 0:
            error_msg = f"Training failed with exit code {result.returncode}\n{result.stderr}"
            print(error_msg)
            
            # Try to restore from backup
            print("Attempting to restore from backup...")
            for file in os.listdir(backup_path):
                shutil.copy(os.path.join(backup_path, file), file)
            
            return jsonify({
                "status": "error",
                "message": error_msg,
                "restored_backup": True
            }), 500
        
        # Update model version
        new_version = float(model_version) + 0.1
        with open('model_version.txt', 'w') as f:
            f.write(f"{new_version:.1f}\n")
            f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # Reload the model
        load_chatbot_model()
        
        return jsonify({
            "status": "success",
            "message": "Chatbot retrained successfully",
            "new_version": f"{new_version:.1f}",
            "output": result.stdout
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Retraining failed: {str(e)}"
        }), 500

if __name__ == '__main__':
    app.run(debug=True)