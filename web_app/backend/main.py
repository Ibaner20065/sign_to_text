from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from typing import List, Optional
import joblib
import numpy as np
import os
import pytesseract
from PIL import Image
import io

# --- CONFIGURATION ---
SECRET_KEY = "aura_secret_key_change_me"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Database connection string
DB_USER = "postgres"
DB_PASSWORD = "[Ibanerjee@20065]"
DB_HOST = "db.hwjetcypauinpgrnlnse.supabase.co"
DB_PORT = "5432"
DB_NAME = "postgres"

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# Database connection pool (None if DB unavailable)
db_pool = None
# In-memory fallback when DB is unavailable
USE_MEMORY_DB = False
memory_users = {}  # email -> {name, email, hashed_password}

# --- DATABASE HELPERS ---
async def get_db_pool():
    global db_pool, USE_MEMORY_DB
    if USE_MEMORY_DB:
        return None
    if db_pool is None:
        try:
            import asyncpg
            db_pool = await asyncpg.create_pool(
                host=DB_HOST,
                port=int(DB_PORT),
                user=DB_USER,
                password=DB_PASSWORD,
                database=DB_NAME,
                min_size=1,
                max_size=10,
                timeout=10,
                command_timeout=10
            )
            async with db_pool.acquire() as conn:
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id SERIAL PRIMARY KEY,
                        email VARCHAR(255) UNIQUE NOT NULL,
                        name VARCHAR(255) NOT NULL,
                        hashed_password VARCHAR(255) NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
            print("✅ Database connected and users table ready")
        except Exception as e:
            print(f"⚠️  Database connection failed: {e}")
            print("⚠️  Falling back to in-memory user storage (development mode)")
            USE_MEMORY_DB = True
            db_pool = None
    return db_pool

async def close_db_pool():
    global db_pool
    if db_pool:
        await db_pool.close()
        db_pool = None

# --- LIFESPAN (replaces deprecated on_event) ---
@asynccontextmanager
async def lifespan(app):
    # Startup
    await get_db_pool()
    yield
    # Shutdown
    await close_db_pool()

app = FastAPI(title="Inclusive Healthcare Platform API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODELS ---
class User(BaseModel):
    name: str
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    name: str

class LandmarkData(BaseModel):
    landmarks: list[float]

class ChangePassword(BaseModel):
    old_password: str
    new_password: str

# --- HELPERS ---
def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        name: str = payload.get("name", "User")
        if email is None:
            raise credentials_exception

        if USE_MEMORY_DB:
            if email not in memory_users:
                raise credentials_exception
            return {"email": email, "name": memory_users[email]["name"]}
        else:
            pool = await get_db_pool()
            async with pool.acquire() as conn:
                user = await conn.fetchrow(
                    "SELECT email, name FROM users WHERE email = $1", email
                )
                if not user:
                    raise credentials_exception
            return {"email": user["email"], "name": user["name"]}
    except JWTError:
        raise credentials_exception
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in get_current_user: {e}")
        raise credentials_exception

# --- ML MODEL LOAD ---
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "sign_model.pkl")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "../../sign_model.pkl"
model = None
try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully")
except Exception as e:
    print(f"⚠️  Error loading model: {e}")

# --- ROUTES ---

@app.post("/auth/register")
async def register(user: User):
    if USE_MEMORY_DB:
        if user.email in memory_users:
            raise HTTPException(status_code=400, detail="Email already registered")
        memory_users[user.email] = {
            "name": user.name,
            "email": user.email,
            "hashed_password": get_password_hash(user.password),
        }
        print(f"✅ User registered (in-memory): {user.email}")
        return {"message": "User registered successfully"}
    else:
        try:
            pool = await get_db_pool()
            async with pool.acquire() as conn:
                existing_user = await conn.fetchrow(
                    "SELECT email FROM users WHERE email = $1", user.email
                )
                if existing_user:
                    raise HTTPException(status_code=400, detail="Email already registered")

                hashed_password = get_password_hash(user.password)
                await conn.execute(
                    "INSERT INTO users (email, name, hashed_password) VALUES ($1, $2, $3)",
                    user.email, user.name, hashed_password
                )
            return {"message": "User registered successfully"}
        except HTTPException:
            raise
        except Exception as e:
            print(f"Database error in register: {e}")
            raise HTTPException(status_code=500, detail="Registration failed. Please try again.")

@app.post("/auth/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    if USE_MEMORY_DB:
        user = memory_users.get(form_data.username)
        if not user or not verify_password(form_data.password, user["hashed_password"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        access_token = create_access_token(data={"sub": user["email"], "name": user["name"]})
        return {"access_token": access_token, "token_type": "bearer", "name": user["name"]}
    else:
        try:
            pool = await get_db_pool()
            async with pool.acquire() as conn:
                user = await conn.fetchrow(
                    "SELECT email, name, hashed_password FROM users WHERE email = $1",
                    form_data.username
                )

                if not user or not verify_password(form_data.password, user["hashed_password"]):
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Incorrect email or password",
                        headers={"WWW-Authenticate": "Bearer"},
                    )

                access_token = create_access_token(data={"sub": user["email"], "name": user["name"]})
                return {"access_token": access_token, "token_type": "bearer", "name": user["name"]}
        except HTTPException:
            raise
        except Exception as e:
            print(f"Database error in login: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Login failed. Please try again.",
            )

@app.get("/auth/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    return {"email": current_user["email"], "name": current_user["name"]}

@app.post("/auth/change-password")
async def change_password(data: ChangePassword, current_user: dict = Depends(get_current_user)):
    if len(data.new_password) < 6:
        raise HTTPException(status_code=400, detail="New password must be at least 6 characters")

    if USE_MEMORY_DB:
        user = memory_users.get(current_user["email"])
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        if not verify_password(data.old_password, user["hashed_password"]):
            raise HTTPException(status_code=400, detail="Current password is incorrect")
        memory_users[current_user["email"]]["hashed_password"] = get_password_hash(data.new_password)
        return {"message": "Password changed successfully"}
    else:
        try:
            pool = await get_db_pool()
            async with pool.acquire() as conn:
                user = await conn.fetchrow(
                    "SELECT hashed_password FROM users WHERE email = $1", current_user["email"]
                )
                if not user:
                    raise HTTPException(status_code=404, detail="User not found")

                if not verify_password(data.old_password, user["hashed_password"]):
                    raise HTTPException(status_code=400, detail="Current password is incorrect")

                new_hashed = get_password_hash(data.new_password)
                await conn.execute(
                    "UPDATE users SET hashed_password = $1 WHERE email = $2",
                    new_hashed, current_user["email"]
                )
            return {"message": "Password changed successfully"}
        except HTTPException:
            raise
        except Exception as e:
            print(f"Database error in change_password: {e}")
            raise HTTPException(status_code=500, detail="Password change failed. Please try again.")

@app.post("/predict")
async def predict_sign(data: LandmarkData, current_user: dict = Depends(get_current_user)):
    if model is None:
        return {"prediction": "I", "confidence": 0.95, "mock": True}
    try:
        features = np.array(data.landmarks).reshape(1, -1)
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0]
        confidence = float(max(proba))
        return {"prediction": prediction, "confidence": confidence, "mock": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/scan-document")
async def scan_document(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        try:
            text = pytesseract.image_to_string(image)
        except Exception:
            text = "Mock Report: Total Amount 50000. Admin Fee 5000. Drug: XyzFakeDrug. Surcharge applied."

        # Enhanced scam detection
        scam_keywords = [
            "admin fee", "surcharge", "processing fee", "inflated",
            "XyzFakeDrug", "hidden charge", "service tax", "convenience fee",
            "duplicate", "overcharge"
        ]
        flagged_items = [word for word in scam_keywords if word.lower() in text.lower()]

        # Check for suspiciously high amounts
        import re
        amounts = re.findall(r'\b\d{5,}\b', text)
        high_amounts = [f"High amount: ₹{amt}" for amt in amounts if int(amt) > 10000]

        return {
            "text": text,
            "flagged": flagged_items + high_amounts,
            "is_suspicious": len(flagged_items) > 0 or len(high_amounts) > 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ambulance-status")
async def get_ambulance_status(current_user: dict = Depends(get_current_user)):
    import random
    return {
        "ambulances": [
            {
                "id": 1,
                "lat_offset": 0.01 + random.uniform(-0.002, 0.002),
                "lng_offset": 0.01 + random.uniform(-0.002, 0.002),
                "status": "Moving",
                "eta": "5 mins",
                "type": "ALS"
            },
            {
                "id": 2,
                "lat_offset": -0.01 + random.uniform(-0.002, 0.002),
                "lng_offset": 0.02 + random.uniform(-0.002, 0.002),
                "status": "Stationary",
                "eta": "12 mins",
                "type": "BLS"
            },
            {
                "id": 3,
                "lat_offset": 0.02 + random.uniform(-0.002, 0.002),
                "lng_offset": -0.01 + random.uniform(-0.002, 0.002),
                "status": "Moving",
                "eta": "8 mins",
                "type": "Neonatal"
            }
        ]
    }

@app.get("/hospitals")
async def get_hospitals(current_user: dict = Depends(get_current_user)):
    return {
        "hospitals": [
            {"id": 101, "name": "City General Hospital", "lat_offset": 0.005, "lng_offset": 0.005, "specialty": "General", "phone": "+91-11-2345-6789"},
            {"id": 102, "name": "Heart Care Center", "lat_offset": -0.008, "lng_offset": 0.01, "specialty": "Cardiology", "phone": "+91-11-2345-6790"},
            {"id": 103, "name": "Neuro Institute", "lat_offset": 0.012, "lng_offset": -0.005, "specialty": "Neurology", "phone": "+91-11-2345-6791"},
            {"id": 104, "name": "Kids Health Hospital", "lat_offset": -0.003, "lng_offset": -0.01, "specialty": "Pediatrics", "phone": "+91-11-2345-6792"},
            {"id": 105, "name": "Emergency Trauma Center", "lat_offset": 0.009, "lng_offset": 0.015, "specialty": "Emergency", "phone": "+91-11-2345-6793"},
            {"id": 106, "name": "Apollo Medical Center", "lat_offset": -0.015, "lng_offset": 0.003, "specialty": "General", "phone": "+91-11-2345-6794"},
            {"id": 107, "name": "Max Heart Hospital", "lat_offset": 0.018, "lng_offset": 0.008, "specialty": "Cardiology", "phone": "+91-11-2345-6795"},
            {"id": 108, "name": "Brain & Spine Clinic", "lat_offset": -0.006, "lng_offset": 0.018, "specialty": "Neurology", "phone": "+91-11-2345-6796"},
            {"id": 109, "name": "Rainbow Children's Hospital", "lat_offset": 0.004, "lng_offset": -0.016, "specialty": "Pediatrics", "phone": "+91-11-2345-6797"},
            {"id": 110, "name": "24x7 Emergency Care", "lat_offset": -0.012, "lng_offset": -0.008, "specialty": "Emergency", "phone": "+91-11-2345-6798"},
            {"id": 111, "name": "Fortis Healthcare", "lat_offset": 0.014, "lng_offset": 0.012, "specialty": "General", "phone": "+91-11-2345-6799"},
            {"id": 112, "name": "Medanta Cardiac Unit", "lat_offset": -0.016, "lng_offset": 0.014, "specialty": "Cardiology", "phone": "+91-11-2345-6800"},
            {"id": 113, "name": "NIMHANS Neuro Center", "lat_offset": 0.008, "lng_offset": -0.012, "specialty": "Neurology", "phone": "+91-11-2345-6801"},
            {"id": 114, "name": "Cloudnine Kids Hospital", "lat_offset": -0.01, "lng_offset": 0.007, "specialty": "Pediatrics", "phone": "+91-11-2345-6802"},
            {"id": 115, "name": "Metro Emergency Hospital", "lat_offset": 0.002, "lng_offset": 0.02, "specialty": "Emergency", "phone": "+91-11-2345-6803"},
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
