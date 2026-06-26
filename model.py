import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from data_loader import load_and_preprocess_data

def train_and_save_model():
    """Runs data loading pipeline, trains a Random Forest Classifier, and saves it."""
    # 1. Load data
    X, y = load_and_preprocess_data()
    
    if len(X) == 0:
        print("No data found. Ensure your dataset paths match your Kaggle extraction directory structure.")
        return
        
    # 2. Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_split=0.2, random_state=42, stratify=y)
    
    # 3. Initialize and train Model
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 4. Evaluate performance
    predictions = model.predict(X_test)
    print(f"Model Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, predictions))
    
    # 5. Save model to disk
    model_filename = 'voice_deepfake_model.pkl'
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model successfully saved to {model_filename}")

if __name__ == "__main__":
    train_and_save_model()