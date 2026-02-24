import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables relative to the script location
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=env_path)

def train_eta_model():
    """
    Extracts incident data from Supabase and trains a RandomForest model 
     to predict response_time_sec based on distance and transport time.
    """
    url = os.environ.get("VITE_SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    
    if not url or not key:
        print("❌ Missing Supabase credentials in environment.")
        return

    supabase: Client = create_client(url, key)
    
    print("📊 Extracting training data from 'incidents' table...")
    try:
        # Part 3: Extract training data
        response = supabase.table("incidents").select("*").execute()
        data = response.data
        
        if not data:
            print("⚠️ No incident data found to train on.")
            return
            
        df = pd.DataFrame(data)
        
        # Features: distance_km, transport_time_sec
        # Target: response_time_sec
        X = df[["distance_km", "transport_time_sec"]]
        y = df["response_time_sec"]
        
        print(f"🧠 Training model on {len(df)} records...")
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Part 4: Save model
        model_path = os.path.join(os.path.dirname(__file__), "..", "eta_model.pkl")
        joblib.dump(model, model_path)
        print(f"✅ Model saved to {model_path}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")

if __name__ == "__main__":
    train_eta_model()
