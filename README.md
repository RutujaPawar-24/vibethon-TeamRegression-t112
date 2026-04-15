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
