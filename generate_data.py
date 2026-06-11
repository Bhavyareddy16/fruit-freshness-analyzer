import os
import numpy as np
import pandas as pd

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Wavelength bands (400nm to 1000nm, 100 bands)
WAVELENGTHS = np.linspace(400, 1000, 100)

def generate_spectral_curve(fruit_type, freshness, wavelengths):
    """
    Simulates a reflectance spectrum (400-1000nm) based on fruit type and freshness.
    Reflectance values range from 0.0 to 1.0.
    """
    n_bands = len(wavelengths)
    # Start with a base curve depending on fruit color and cell structure
    if fruit_type == "Apple":
        # Red fruit: high red reflectance (620-700nm), low green/blue
        base = np.zeros(n_bands)
        # Red peak
        base += 0.45 * np.exp(-((wavelengths - 650) / 40) ** 2)
        # Green bump
        base += 0.10 * np.exp(-((wavelengths - 550) / 30) ** 2)
    elif fruit_type == "Banana":
        # Yellow fruit: high green + red (530-700nm)
        base = np.zeros(n_bands)
        base += 0.55 * np.exp(-((wavelengths - 580) / 60) ** 2)
    else:  # Orange
        # Orange fruit: high red/orange (590-700nm)
        base = np.zeros(n_bands)
        base += 0.50 * np.exp(-((wavelengths - 610) / 50) ** 2)
    
    # NIR region (750-1000nm) represents cellular integrity and water content
    # Fresh fruits have high NIR reflectance (~0.6 - 0.85) due to intact cell walls
    # Spoiled fruits have low NIR reflectance due to cell collapse (browning/softening)
    if freshness == "Fresh":
        nir_level = 0.75
        water_dip_depth = 0.25 # Water absorption dip at ~970nm
        chlorophyll_dip_depth = 0.30 if fruit_type in ["Apple", "Banana"] else 0.10 # Dip at ~675nm
    elif freshness == "Moderately Fresh":
        nir_level = 0.55
        water_dip_depth = 0.18
        chlorophyll_dip_depth = 0.12
    else:  # Spoiled
        nir_level = 0.35
        water_dip_depth = 0.08
        chlorophyll_dip_depth = 0.02 # Chlorophyll degrades, browning makes it flat
        
    # NIR plateau
    nir_curve = nir_level * (1.0 / (1.0 + np.exp(-((wavelengths - 730) / 20))))
    
    # Add dips
    # 1. Water absorption dip at 970nm
    water_dip = water_dip_depth * np.exp(-((wavelengths - 970) / 40) ** 2)
    # 2. Chlorophyll absorption dip at 675nm
    chlorophyll_dip = chlorophyll_dip_depth * np.exp(-((wavelengths - 675) / 15) ** 2)
    
    # Full spectrum
    reflectance = base + nir_curve - water_dip - chlorophyll_dip
    
    # Clip and add small noise
    reflectance = np.clip(reflectance, 0.01, 0.95)
    noise = np.random.normal(0, 0.01, n_bands)
    reflectance = np.clip(reflectance + noise, 0.0, 1.0)
    
    return reflectance

def generate_datasets(num_samples=6000):
    """
    Generates spectral-chemical and E-Nose environmental datasets.
    """
    fruits = ["Apple", "Banana", "Orange"]
    freshness_levels = ["Fresh", "Moderately Fresh", "Spoiled"]
    
    spectral_data = []
    sensor_data = []
    
    for _ in range(num_samples):
        # Select random fruit and freshness state
        fruit = np.random.choice(fruits)
        freshness = np.random.choice(freshness_levels)
        
        # ------------------ 1. Spectral & Chemical Data ------------------
        # Chemical properties dependent on fruit and freshness
        if fruit == "Apple":
            # Apple sugar (Brix): 10-15%. Tends to drop slightly in spoiled
            base_brix = 13.0 if freshness == "Fresh" else (12.0 if freshness == "Moderately Fresh" else 9.5)
            brix = np.random.normal(base_brix, 0.8)
            
            # Moisture %: 83-86%
            base_moisture = 85.0 if freshness == "Fresh" else (82.5 if freshness == "Moderately Fresh" else 76.0)
            moisture = np.random.normal(base_moisture, 1.0)
            
            # Firmness (N): 60-90N fresh, drops to <30N spoiled
            base_firmness = 75.0 if freshness == "Fresh" else (50.0 if freshness == "Moderately Fresh" else 20.0)
            firmness = np.random.normal(base_firmness, 5.0)
            
        elif fruit == "Banana":
            # Banana sugar (Brix): 15-22% (rises as starch turns to sugar, then drops during decay)
            base_brix = 16.0 if freshness == "Fresh" else (20.0 if freshness == "Moderately Fresh" else 14.0)
            brix = np.random.normal(base_brix, 1.0)
            
            # Moisture %: 72-76%
            base_moisture = 75.0 if freshness == "Fresh" else (74.0 if freshness == "Moderately Fresh" else 68.0)
            moisture = np.random.normal(base_moisture, 1.2)
            
            # Firmness (N): 20-30N fresh, drops to <5N spoiled
            base_firmness = 25.0 if freshness == "Fresh" else (12.0 if freshness == "Moderately Fresh" else 3.0)
            firmness = np.random.normal(base_firmness, 2.0)
            
        else: # Orange
            # Orange sugar (Brix): 9-13%
            base_brix = 11.5 if freshness == "Fresh" else (10.5 if freshness == "Moderately Fresh" else 8.0)
            brix = np.random.normal(base_brix, 0.7)
            
            # Moisture %: 85-88%
            base_moisture = 87.0 if freshness == "Fresh" else (84.0 if freshness == "Moderately Fresh" else 78.0)
            moisture = np.random.normal(base_moisture, 1.0)
            
            # Firmness (N): 40-60N fresh, drops to <15N spoiled
            base_firmness = 50.0 if freshness == "Fresh" else (32.0 if freshness == "Moderately Fresh" else 12.0)
            firmness = np.random.normal(base_firmness, 4.0)
            
        # Clip chemical values to physical limits
        brix = max(4.0, brix)
        moisture = np.clip(moisture, 50.0, 95.0)
        firmness = max(0.5, firmness)
        
        # Generate HSI spectrum
        spectrum = generate_spectral_curve(fruit, freshness, WAVELENGTHS)
        
        # Row dict for spectral
        spec_row = {
            "fruit_type": fruit,
            "freshness": freshness,
            "sugar_brix": round(brix, 2),
            "moisture_pct": round(moisture, 2),
            "firmness_n": round(firmness, 2)
        }
        for i, val in enumerate(spectrum):
            spec_row[f"band_{i+1}"] = round(val, 4)
            
        spectral_data.append(spec_row)
        
        # ------------------ 2. E-Nose & IoT Sensor Data ------------------
        # Gas readings in ppm.
        # MQ135: General Air Quality / CO2 / Alcohol (Clean: 100-250, Moderate: 300-600, Spoiled: 800-2500)
        # MQ3: Ethanol/Alcohol (Clean: 10-30, Moderate: 40-100, Spoiled: 300-1500)
        # MQ4: Methane / Ripening organic gases (Clean: 5-15, Moderate: 20-50, Spoiled: 100-400)
        # MQ137: Ammonia (proteins breakdown) (Clean: 1-5, Moderate: 6-15, Spoiled: 30-150)
        # MQ138: Benzene / Acetone / Ethylene (Clean: 2-8, Moderate: 15-40, Spoiled: 80-300)
        
        # Environmental conditions (temp: 10-35 C, hum: 45-95%)
        # Higher temperatures and humidity accelerate spoilage.
        temp = np.random.uniform(12.0, 32.0)
        humidity = np.random.uniform(50.0, 90.0)
        
        if freshness == "Fresh":
            mq135 = np.random.normal(180, 25)
            mq3 = np.random.normal(20, 5)
            mq4 = np.random.normal(10, 2.5)
            mq137 = np.random.normal(3, 1)
            mq138 = np.random.normal(5, 1.5)
            spoilage_idx = np.random.uniform(0.0, 0.25)
        elif freshness == "Moderately Fresh":
            mq135 = np.random.normal(450, 60)
            mq3 = np.random.normal(70, 15)
            # Bananas release lots of ethylene and other organic gases when ripe
            mq4 = np.random.normal(45 if fruit == "Banana" else 25, 6)
            mq137 = np.random.normal(10, 2)
            mq138 = np.random.normal(35 if fruit == "Banana" else 20, 5)
            spoilage_idx = np.random.uniform(0.26, 0.65)
        else: # Spoiled
            mq135 = np.random.normal(1500, 250)
            mq3 = np.random.normal(850, 150)
            mq4 = np.random.normal(250, 40)
            mq137 = np.random.normal(85, 15)
            mq138 = np.random.normal(180, 30)
            spoilage_idx = np.random.uniform(0.66, 1.0)
            
        # Adjust gases slightly based on ambient temperature (heat increases volatile release)
        temp_multiplier = 1.0 + (temp - 20.0) * 0.02
        mq135 = max(50.0, mq135 * temp_multiplier)
        mq3 = max(5.0, mq3 * temp_multiplier)
        mq4 = max(2.0, mq4 * temp_multiplier)
        mq137 = max(0.5, mq137 * temp_multiplier)
        mq138 = max(0.5, mq138 * temp_multiplier)
        
        sensor_row = {
            "fruit_type": fruit,
            "freshness": freshness,
            "temp_c": round(temp, 1),
            "humidity_pct": round(humidity, 1),
            "mq135_co2_ppm": round(mq135, 1),
            "mq3_alcohol_ppm": round(mq3, 1),
            "mq4_methane_ppm": round(mq4, 1),
            "mq137_ammonia_ppm": round(mq137, 1),
            "mq138_benzene_ppm": round(mq138, 1),
            "spoilage_index": round(spoilage_idx, 3)
        }
        sensor_data.append(sensor_row)
        
    # Save to CSV
    spec_df = pd.DataFrame(spectral_data)
    sensor_df = pd.DataFrame(sensor_data)
    
    spec_df.to_csv("data/spectral_dataset.csv", index=False)
    sensor_df.to_csv("data/sensor_dataset.csv", index=False)
    
    print(f"Generated {num_samples} samples.")
    print("Saved spectral_dataset.csv shape:", spec_df.shape)
    print("Saved sensor_dataset.csv shape:", sensor_df.shape)

if __name__ == "__main__":
    generate_datasets()
