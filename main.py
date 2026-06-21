import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_and_prep_data():
    """
    Loads the medical dataset, splits it into training and testing sets, 
    and applies feature scaling.
    """
    # 1. Load Breast Cancer dataset (Binary Classification: Malignant vs. Benign)
    # To use Diabetes or Heart Disease, you would load your CSV via pd.read_csv() here.
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    # 2. Split the data (80% training, 20% testing)
    # stratify=y ensures both sets have the same proportion of disease/no-disease cases
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Scale the features
    # Standardization is CRITICAL for SVM and Logistic Regression to perform well.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Only transform the test set to prevent data leakage
    
    return X_train_scaled, X_test_scaled, y_train, y_test, data.target_names

def train_and_evaluate_models(X_train, X_test, y_train, y_test, target_names):
    """
    Initializes, trains, and evaluates multiple classification algorithms.
    """
    # Initialize the four requested algorithms
    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Support Vector Machine (SVM)": SVC(kernel='linear', random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    results = {}
    
    # Iterate through each model, train, and print results
    for name, model in models.items():
        print(f"--- Training {name} ---")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Predict on unseen test data
        y_pred = model.predict(X_test)
        
        # Evaluate performance
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=target_names)
        
        results[name] = accuracy
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Classification Report:\n{report}\n")
        
    return results

if __name__ == "__main__":
    print("Starting Disease Prediction Pipeline...\n")
    
    # Run the pipeline
    X_train, X_test, y_train, y_test, target_names = load_and_prep_data()
    model_results = train_and_evaluate_models(X_train, X_test, y_train, y_test, target_names)
    
    # Display a final leaderboard of the models
    print("--- Final Model Leaderboard ---")
    for name, acc in sorted(model_results.items(), key=lambda x: x[1], reverse=True):
        print(f"{name}: {acc:.4f} Accuracy")