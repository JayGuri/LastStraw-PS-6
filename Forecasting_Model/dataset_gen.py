import requests
import pandas as pd
import time

# ==========================================
# 1. THE ULTIMATE GLOBAL MATRIX (45 Cities)
# ==========================================
CITIES = [
    {"name": "Navi_Mumbai", "lat": 19.0330, "lon": 73.0297},
    {"name": "Mumbai_South", "lat": 18.9388, "lon": 72.8258},
    {"name": "Pune", "lat": 18.5204, "lon": 73.8567},
    {"name": "Chennai", "lat": 13.0827, "lon": 80.2707},
    {"name": "Kolkata", "lat": 22.5726, "lon": 88.3639},
    {"name": "Delhi", "lat": 28.7041, "lon": 77.1025},
    {"name": "Dhaka", "lat": 23.8103, "lon": 90.4125},
    {"name": "Kathmandu", "lat": 27.7172, "lon": 85.3240},
    {"name": "Colombo", "lat": 6.9271, "lon": 79.8612},
    {"name": "Cherrapunji", "lat": 25.2702, "lon": 91.7323},
    {"name": "Jakarta", "lat": -6.2088, "lon": 106.8456},
    {"name": "Manila", "lat": 14.5995, "lon": 120.9842},
    {"name": "Tokyo", "lat": 35.6762, "lon": 139.6503},
    {"name": "Seoul", "lat": 37.5665, "lon": 126.9780},
    {"name": "Beijing", "lat": 39.9042, "lon": 116.4074},
    {"name": "Bangkok", "lat": 13.7563, "lon": 100.5018},
    {"name": "Sydney", "lat": -33.8688, "lon": 151.2093},
    {"name": "Auckland", "lat": -36.8485, "lon": 174.7633},
    {"name": "Dubai", "lat": 25.2048, "lon": 55.2708},
    {"name": "Riyadh", "lat": 24.7136, "lon": 46.6753},
    {"name": "Cairo", "lat": 30.0444, "lon": 31.2357},
    {"name": "Lagos", "lat": 6.5244, "lon": 3.3792},
    {"name": "Nairobi", "lat": -1.2921, "lon": 36.8219},
    {"name": "Cape_Town", "lat": -33.9249, "lon": 18.4241},
    {"name": "London", "lat": 51.5074, "lon": -0.1278},
    {"name": "Amsterdam", "lat": 52.3676, "lon": 4.9041},
    {"name": "Valencia", "lat": 39.4699, "lon": -0.3774},
    {"name": "Istanbul", "lat": 41.0082, "lon": 28.9784},
    {"name": "Rome", "lat": 41.9028, "lon": 12.4964},
    {"name": "Moscow", "lat": 55.7558, "lon": 37.6173},
    {"name": "New_York", "lat": 40.7128, "lon": -74.0060},
    {"name": "Miami", "lat": 25.7617, "lon": -80.1918},
    {"name": "Houston", "lat": 29.7604, "lon": -95.3698},
    {"name": "Los_Angeles", "lat": 34.0522, "lon": -118.2437},
    {"name": "Denver", "lat": 39.7392, "lon": -104.9903},
    {"name": "Las_Vegas", "lat": 36.1699, "lon": -115.1398},
    {"name": "Seattle", "lat": 47.6062, "lon": -122.3321},
    {"name": "Anchorage", "lat": 61.2181, "lon": -149.9003},
    {"name": "Mexico_City", "lat": 19.4326, "lon": -99.1332},
    {"name": "Toronto", "lat": 43.6510, "lon": -79.3470},
    {"name": "Sao_Paulo", "lat": -23.5505, "lon": -46.6333},
    {"name": "Rio_de_Janeiro", "lat": -22.9068, "lon": -43.1729},
    {"name": "Manaus", "lat": -3.1190, "lon": -60.0217},
    {"name": "Bogota", "lat": 4.7110, "lon": -74.0721},
    {"name": "Buenos_Aires", "lat": -34.6037, "lon": -58.3816},
    {"name": "Lima", "lat": -12.0464, "lon": -77.0428}
]

# Pulling 16+ years of data, from 2010 right up to today
START_DATE = "2010-01-01"
END_DATE = "2026-02-28" 

all_city_data = []
print("Initiating 16-Year Global Data Extraction (2010 - Present)...")

for city in CITIES:
    print(f"Fetching 16 years of data for {city['name']}...")
    try:
        elev_url = f"https://api.open-meteo.com/v1/elevation?latitude={city['lat']}&longitude={city['lon']}"
        elev_res = requests.get(elev_url).json()
        elevation = elev_res.get('elevation', [0])[0]
        
        weather_url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={city['lat']}&longitude={city['lon']}&"
            f"start_date={START_DATE}&end_date={END_DATE}&"
            f"hourly=precipitation,soil_moisture_0_to_7cm,temperature_2m&"
            f"timezone=GMT"
        )
        
        weather_res = requests.get(weather_url)
        if weather_res.status_code != 200: 
            print(f"  -> API rejected {city['name']}. Moving to next.")
            continue
            
        data = weather_res.json()
        df = pd.DataFrame({
            'City': city['name'],
            'Elevation_m': elevation,
            'Precipitation_mm': data['hourly']['precipitation'],
            'Soil_Moisture': data['hourly']['soil_moisture_0_to_7cm'],
            'Temperature_C': data['hourly']['temperature_2m']
        })
        
        df['Rain_Last_3h'] = df['Precipitation_mm'].rolling(window=3, min_periods=1).sum()
        
        # The Proxy Physics Rule
        flood_condition = (df['Rain_Last_3h'] >= 30.0) | ((df['Rain_Last_3h'] >= 20.0) & (df['Soil_Moisture'] >= 0.35))
        df['Flash_Flood_Risk'] = flood_condition.astype(int)
        
        df = df.dropna().reset_index(drop=True)
        all_city_data.append(df)
        
        # Sleep to avoid IP ban on such a massive pull
        time.sleep(3)
        
    except Exception as e:
        print(f"Error on {city['name']}: {e}")

master_df = pd.concat(all_city_data, ignore_index=True)

# Keeping the exact same filename so your pipeline.py works without edits
master_df.to_csv("global_flash_flood_data_decade.csv", index=False)

print("\n=== MASSIVE EXTRACTION COMPLETE ===")
print(f"Total Rows Saved: {len(master_df):,}")
print(f"Total Flood Events Found: {master_df['Flash_Flood_Risk'].sum():,}")