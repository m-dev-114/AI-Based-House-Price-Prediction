import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def preprocess_data(csv_file):
    # Load dataset
    df = pd.read_csv(csv_file)
    
    # Data Preprocessing
    X = df.drop(columns=['price'])  # Features
    y = df['price']  # Target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler for deployment
    joblib.dump(scaler, 'scaler.pkl')
    
    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data('house_prices.csv')
    print("Preprocessing complete. Scaler saved as scaler.pkl")
