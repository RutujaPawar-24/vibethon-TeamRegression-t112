
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
