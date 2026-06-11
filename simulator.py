import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Wavelength bands (400nm to 1000nm, 100 bands)
WAVELENGTHS = np.linspace(400, 1000, 100)

def generate_telemetry_history(hours=24, anomalous=False):
    """
    Generates historical time-series data for a cold storage room.
    If anomalous=True, simulates a ventilation breakdown and temperature rise.
    """
    now = datetime.now()
    timestamps = [now - timedelta(minutes=5 * i) for i in range(hours * 12)]
    timestamps.reverse()
    
    np.random.seed(42 if not anomalous else 24)
    
    # Base conditions
    if anomalous:
        # Temperature breaks down and rises from 15C to 28C
        temp_base = np.linspace(14.0, 29.0, len(timestamps))
        # Humidity drops then spikes
        hum_base = np.linspace(75.0, 88.0, len(timestamps))
    else:
        # Normal cold storage conditions (12C - 16C)
        temp_base = 14.0 + 1.5 * np.sin(np.linspace(0, 4 * np.pi, len(timestamps)))
        hum_base = 80.0 + 3.0 * np.cos(np.linspace(0, 4 * np.pi, len(timestamps)))
        
    temp = temp_base + np.random.normal(0, 0.2, len(timestamps))
    humidity = hum_base + np.random.normal(0, 0.5, len(timestamps))
    
    # Gas concentrations build up depending on temperature and time
    # MQ135: CO2 and generic air quality
    # MQ3: Ethanol
    # MQ138: Benzene / Ethylene (ripening gas)
    
    gas_co2 = []
    gas_eth = []
    gas_ethyl = []
    gas_amm = []
    
    current_co2 = 250.0
    current_eth = 20.0
    current_ethyl = 5.0
    current_amm = 2.0
    
    for t, h in zip(temp, humidity):
        # Spoilage/ripening rate is exponential with temperature
        decay_factor = np.exp((t - 15.0) / 7.0)
        
        # Increments
        if anomalous:
            # Quick decay and accumulation
            co2_inc = np.random.uniform(1.0, 5.0) * decay_factor
            eth_inc = np.random.uniform(0.5, 3.0) * decay_factor
            ethyl_inc = np.random.uniform(0.2, 1.2) * decay_factor
            amm_inc = np.random.uniform(0.05, 0.3) * decay_factor
        else:
            # Gradual/slow accumulation, partially cleared by ventilation
            co2_inc = np.random.uniform(0.1, 0.5) * decay_factor - 0.1
            eth_inc = np.random.uniform(0.02, 0.1) * decay_factor - 0.02
            ethyl_inc = np.random.uniform(0.01, 0.05) * decay_factor - 0.01
            amm_inc = np.random.uniform(0.001, 0.01) * decay_factor
            
        current_co2 = max(180.0, min(current_co2 + co2_inc, 3500.0))
        current_eth = max(10.0, min(current_eth + eth_inc, 1800.0))
        current_ethyl = max(2.0, min(current_ethyl + ethyl_inc, 400.0))
        current_amm = max(0.5, min(current_amm + amm_inc, 200.0))
        
        # Add reading noise
        gas_co2.append(round(current_co2 + np.random.normal(0, 5), 1))
        gas_eth.append(round(current_eth + np.random.normal(0, 1), 1))
        gas_ethyl.append(round(current_ethyl + np.random.normal(0, 0.3), 1))
        gas_amm.append(round(current_amm + np.random.normal(0, 0.1), 1))
        
    df = pd.DataFrame({
        "timestamp": timestamps,
        "temp_c": np.round(temp, 1),
        "humidity_pct": np.round(humidity, 1),
        "mq135_co2_ppm": gas_co2,
        "mq3_alcohol_ppm": gas_eth,
        "mq138_benzene_ppm": gas_ethyl,
        "mq137_ammonia_ppm": gas_amm
    })
    
    return df

def simulate_hsi_scan(fruit_type, days_stored, storage_temp):
    """
    Simulates a live hyperspectral and chemical scan.
    """
    # Calculate freshness based on storage parameters
    # High temp and high days = spoiled
    freshness_score = 1.0 - (days_stored * np.exp((storage_temp - 15) / 10)) / 14.0
    freshness_score = np.clip(freshness_score, 0.0, 1.0)
    
    if freshness_score > 0.7:
        freshness = "Fresh"
    elif freshness_score > 0.35:
        freshness = "Moderately Fresh"
    else:
        freshness = "Spoiled"
        
    # Generate chemical values
    if fruit_type == "Apple":
        brix = 13.0 - (1.0 - freshness_score) * 4.0
        moisture = 85.0 - (1.0 - freshness_score) * 10.0
        firmness = 75.0 - (1.0 - freshness_score) * 55.0
    elif fruit_type == "Banana":
        # Brix rises as starch converts, then falls
        if freshness == "Fresh":
            brix = 15.0 + (1.0 - freshness_score) * 10.0
        elif freshness == "Moderately Fresh":
            brix = 20.0
        else:
            brix = 20.0 - (0.35 - freshness_score) * 18.0
            
        moisture = 75.0 - (1.0 - freshness_score) * 8.0
        firmness = 25.0 - (1.0 - freshness_score) * 23.0
    else:  # Orange
        brix = 11.5 - (1.0 - freshness_score) * 3.5
        moisture = 87.0 - (1.0 - freshness_score) * 9.0
        firmness = 50.0 - (1.0 - freshness_score) * 38.0
        
    # Standard deviation noise
    brix = max(4.0, np.random.normal(brix, 0.2))
    moisture = np.clip(np.random.normal(moisture, 0.3), 50.0, 95.0)
    firmness = max(0.5, np.random.normal(firmness, 0.5))
    
    # Generate HSI spectrum
    # Base curve color
    n_bands = len(WAVELENGTHS)
    if fruit_type == "Apple":
        base = np.zeros(n_bands)
        base += 0.45 * np.exp(-((WAVELENGTHS - 650) / 40) ** 2)
        base += 0.10 * np.exp(-((WAVELENGTHS - 550) / 30) ** 2)
    elif fruit_type == "Banana":
        base = np.zeros(n_bands)
        base += 0.55 * np.exp(-((WAVELENGTHS - 580) / 60) ** 2)
    else:  # Orange
        base = np.zeros(n_bands)
        base += 0.50 * np.exp(-((WAVELENGTHS - 610) / 50) ** 2)
        
    # Ripening changes
    if freshness == "Fresh":
        nir_level = 0.76
        water_dip_depth = 0.25
        chlorophyll_dip_depth = 0.30 if fruit_type in ["Apple", "Banana"] else 0.10
    elif freshness == "Moderately Fresh":
        nir_level = 0.56
        water_dip_depth = 0.18
        chlorophyll_dip_depth = 0.12
    else:
        nir_level = 0.34
        water_dip_depth = 0.08
        chlorophyll_dip_depth = 0.01
        
    # Apply aging to spectrum
    nir_curve = nir_level * (1.0 / (1.0 + np.exp(-((WAVELENGTHS - 730) / 20))))
    water_dip = water_dip_depth * np.exp(-((WAVELENGTHS - 970) / 40) ** 2)
    chlorophyll_dip = chlorophyll_dip_depth * np.exp(-((WAVELENGTHS - 675) / 15) ** 2)
    
    reflectance = base + nir_curve - water_dip - chlorophyll_dip
    reflectance = np.clip(reflectance + np.random.normal(0, 0.005, n_bands), 0.0, 1.0)
    
    # Generate E-Nose reading
    if freshness == "Fresh":
        mq135 = 170.0 + np.random.normal(0, 10)
        mq3 = 18.0 + np.random.normal(0, 2)
        mq4 = 8.0 + np.random.normal(0, 1)
        mq137 = 2.5 + np.random.normal(0, 0.5)
        mq138 = 4.0 + np.random.normal(0, 0.8)
    elif freshness == "Moderately Fresh":
        mq135 = 420.0 + np.random.normal(0, 30)
        mq3 = 65.0 + np.random.normal(0, 8)
        mq4 = 40.0 if fruit_type == "Banana" else 22.0 + np.random.normal(0, 2)
        mq137 = 9.0 + np.random.normal(0, 1)
        mq138 = 30.0 if fruit_type == "Banana" else 18.0 + np.random.normal(0, 1.5)
    else:
        mq135 = 1450.0 + np.random.normal(0, 100)
        mq3 = 820.0 + np.random.normal(0, 80)
        mq4 = 240.0 + np.random.normal(0, 20)
        mq137 = 80.0 + np.random.normal(0, 8)
        mq138 = 170.0 + np.random.normal(0, 15)
        
    temp_mult = 1.0 + (storage_temp - 20.0) * 0.02
    mq135 = max(50.0, mq135 * temp_mult)
    mq3 = max(5.0, mq3 * temp_mult)
    mq4 = max(2.0, mq4 * temp_mult)
    mq137 = max(0.5, mq137 * temp_mult)
    mq138 = max(0.5, mq138 * temp_mult)
    
    return {
        "fruit_type": fruit_type,
        "freshness_actual": freshness,
        "sugar_brix": round(brix, 2),
        "moisture_pct": round(moisture, 2),
        "firmness_n": round(firmness, 2),
        "spectrum": reflectance.tolist(),
        "gases": {
            "mq135": round(mq135, 1),
            "mq3": round(mq3, 1),
            "mq4": round(mq4, 1),
            "mq137": round(mq137, 1),
            "mq138": round(mq138, 1)
        }
    }
