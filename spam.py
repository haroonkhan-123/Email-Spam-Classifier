import pickle
import os

# Define the correct file paths
base_dir = "C:/Users/MEGA COMPUTER/Desktop/Machine Learning Projects By Campus X/2nd Ml project/"
vectorizer_path = os.path.join(base_dir, "vectorizer.pkl")
model_path = os.path.join(base_dir, "spam_classifier.pkl")

# ✅ Load the saved TF-IDF vectorizer and trained model
with open(vectorizer_path, "rb") as f:
    tfidf = pickle.load(f)  # Load TF-IDF vectorizer

with open(model_path, "rb") as f:
    best_nb = pickle.load(f)  # Load trained model

def predict_message(text, vectorizer, model):
    text_vector = vectorizer.transform([text])  
    prediction = model.predict(text_vector)[0]  
    return "SPAM" if prediction == 1 else "HAM"

# ✅ Get user input for message classification
user_text = input("Enter a message to check: ")  
result = predict_message(user_text, tfidf, best_nb)  
print("Prediction:", result)
