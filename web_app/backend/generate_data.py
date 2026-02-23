import random
import uuid
from datetime import datetime, timedelta

districts = [
    {"id": "5163c916-1a20-4822-aab0-890e39dd843d", "name": "Hooghly", "lat": 22.9, "lon": 88.4},
    {"id": "0574e98c-8a1e-4442-b365-62b1e7f3dda3", "name": "Bardhaman", "lat": 23.2, "lon": 87.9},
]

hospitals = []
ambulances = []
incidents = []

sql_output = []

# Generate Hospitals
for d in districts:
    for i in range(5):
        hid = str(uuid.uuid4())
        name = f"{d['name']} General Hospital {i+1}"
        lat = d["lat"] + random.uniform(-0.1, 0.1)
        lon = d["lon"] + random.uniform(-0.1, 0.1)
        beds = random.randint(50, 200)
        icu = random.randint(5, 20)
        hospitals.append({"id": hid, "district_id": d["id"]})
        sql_output.append(
            f"INSERT INTO hospitals (id, name, district_id, beds_total, icu_beds, lat, lon, geom) VALUES ('{hid}', '{name}', '{d['id']}', {beds}, {icu}, {lat}, {lon}, ST_SetSRID(ST_Point({lon}, {lat}), 4326)::geography);"
        )

# Generate Ambulances
for d in districts:
    district_hospitals = [h for h in hospitals if h["district_id"] == d["id"]]
    for i in range(10):
        aid = str(uuid.uuid4())
        v_num = f"WB-{random.randint(10, 99)}-{random.randint(1000, 9999)}"
        atype = random.choice(["ALS", "BLS", "PTV"])
        provider = random.choice(["NHM", "Private"])
        hid = random.choice(district_hospitals)["id"]
        lat = d["lat"] + random.uniform(-0.1, 0.1)
        lon = d["lon"] + random.uniform(-0.1, 0.1)
        ambulances.append({"id": aid, "district_id": d["id"]})
        sql_output.append(
            f"INSERT INTO ambulances (id, vehicle_number, type, provider, district_id, base_hospital_id, status, lat, lon, geom) VALUES ('{aid}', '{v_num}', '{atype}', '{provider}', '{d['id']}', '{hid}', 'active', {lat}, {lon}, ST_SetSRID(ST_Point({lon}, {lat}), 4326)::geography);"
        )

# Generate Incidents
for d in districts:
    district_hospitals = [h for h in hospitals if h["district_id"] == d["id"]]
    district_ambulances = [a for a in ambulances if a["district_id"] == d["id"]]
    for i in range(100):
        iid = str(uuid.uuid4())
        aid = random.choice(district_ambulances)["id"]
        hid = random.choice(district_hospitals)["id"]
        call_time = datetime(2026, 1, 1) + timedelta(minutes=random.randint(0, 43200)) # Jan 2026
        resp_sec = random.randint(300, 1800)
        trans_sec = random.randint(600, 2400)
        itype = random.choice(["cardiac", "trauma", "pregnancy", "stroke", "accident"])
        triage = random.choice(["Critical", "Non-Critical", "Routine"])
        dist = random.uniform(2, 15)
        
        sql_output.append(
            f"INSERT INTO incidents (id, call_time, district_id, ambulance_id, hospital_id, response_time_sec, transport_time_sec, incident_type, triage_level, distance_km) VALUES ('{iid}', '{call_time.isoformat()}', '{d['id']}', '{aid}', '{hid}', {resp_sec}, {trans_sec}, '{itype}', '{triage}', {dist});"
        )

with open("seed_data.sql", "w") as f:
    f.write("\n".join(sql_output))
