# %%writefile command se ek nayi file create hogi
%%writefile image_module.py

import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import os

# 1. Pre-trained AI Model (MobileNetV2) load karna
# Ye model Google ka hai aur 1000+ objects pehchan sakta hai
model = MobileNetV2(weights='imagenet')

def classify_now(img_path):
    try:
        # Image ko AI ke format (224x224) mein resize karna
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # AI Prediction start
        preds = model.predict(x)
        
        # Top 3 results print karna
        print('\n--- 🤖 AI ne kya pehchana? ---')
        for i, (imagenet_id, label, prob) in enumerate(decode_predictions(preds, top=3)[0]):
            print(f"{i+1}: {label} ({prob*100:.2f}%)")
    except Exception as e:
        print(f"Error: Photo process nahi ho payi. Check karo ki path sahi hai?")

if __name__ == "__main__":
    print("--- 📸 VIBETHON Image Recognition Module ---")
    file_path = input("Apni image ka naam/path likho (e.g., test.jpg): ")
    
    if os.path.exists(file_path):
        classify_now(file_path)
    else:
        print("File nahi mili! Pehle Colab mein photo upload karo.")

### 4 Mini-Games for Concept Learning
To simplify complex AI topics for beginners, the platform includes interactive mini-games:
* **File:** `ai_games.html`
* **Concept 1: Decision Trees:** Users navigate a 4-level branching logic to classify fruits based on features like shape, color, and origin.
* **Concept 2: Neural Networks:** Simulates pattern recognition where weights are adjusted and a **Softmax Activation** function is applied to give a final prediction.
* **Feature:** Includes a "Neural Inference" bar that shows the real-time confidence score (e.g., 92% match) for each classification.
  # 🤖 AI Integrated Learning Lab

An interactive educational platform designed to simplify complex Artificial Intelligence concepts through gamification, real-time logic visualization, and assessment modules.
### 5. Leaderboard & Gamification
Engaging users through professional reward systems:
* **XP System:** Earn Experience Points for every correct logic path and quiz answer.
* **Streaks:** Tracks consecutive days of learning to encourage consistency.
* **Digital Badges:** Unlock badges like `AI Mechanic`, `Scholar`, and `Expert` based on performance milestones.


import pandas as pd
import requests, zipfile, io
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# 1. Dataset Loading (Section 3.6 Requirement)
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
r = requests.get(url)
z = zipfile.ZipFile(io.BytesIO(r.content))
df = pd.read_csv(z.open('SMSSpamCollection'), sep='\t', names=['label', 'message'])

# 2. Pre-processing & Training
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Pipeline for structured execution
model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])
model.fit(X_train, y_train)

# 3. Functional Simulation (Interaction Flow)
def main():
    print("--- 📱 VIBETHON AI SPAM DETECTOR ---")
    msg = input("Check your message here: ")
    prediction = model.predict([msg])
    result = "⚠️ SPAM" if prediction[0] == 1 else "✅ HAM (SAFE)"
    print(f"Prediction: {result}")

if __name__ == "__main__":
    main()


