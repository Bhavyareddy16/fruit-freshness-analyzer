# Non-Destructive Fruit Freshness Analyzer (Software-Only Implementation)

An intelligent software-based simulation and quality assessment system for fruits using simulated **Hyperspectral Imaging (HSI)**, **Near-Infrared (NIR) Spectroscopy**, **IoT Electronic Nose (E-Nose)**, and **Machine Learning** fusion.

This application is built entirely in software, meaning it does not require physical cameras, spectrometers, or gas sensors. It uses scientific ripening profiles to simulate sensor readings dynamically and classify specimens using trained machine learning models.

---

## 🚀 Features

1. **Ripening & Aging Simulation Engine**: Adjust days in storage and ambient temperatures to dynamically shift the physical, spectral, and gaseous characteristics of Apple, Banana, and Orange specimens.
2. **Interactive Hyperspectral Scanner**: Displays spatial-spectral simulation charts and maps reflectance vs. wavelength, highlighting key absorption bands like Chlorophyll (~675nm) and Water (~970nm).
3. **Electronic Nose (E-Nose) Profiler**: Visualizes volatile organic compounds (CO2, Ethanol, Methane, Ammonia, and organic volatiles) released at different decay rates.
4. **Machine Learning Fusion Pipeline**: Uses trained Scikit-learn Random Forests, SVMs, and Gradient Boosting to classify fruit freshness (`Fresh`, `Moderately Fresh`, `Spoiled`) and estimate internal traits (Sugar content/Brix, Moisture, and Firmness).
5. **Cold Storage Ambient IoT Dashboard**: Features a 24-hour environmental sensor log with a simulation toggle to trigger ventilation breakdowns and live spoilage risk alerts.
6. **Automated Quality Certification**: Dynamic generator that creates and compiles formal PDF inspection certificates with analysis values, interactive charts, and custom recommendations.

---

## 📂 Project Structure

- `app.py`: Main Streamlit application and layout design (Premium Dark Theme).
- `simulator.py`: High-fidelity simulation utilities generating environmental telemetry logs and live fruit scan sensors.
- `generate_data.py`: Data generation script that creates spectral and gas databases (6,000 samples) mapped to fruit degradation rules.
- `train_models.py`: Machine learning pipeline training the classification and regression models.
- `create_samples.py`: Helper drawing sample vector fruit images (Fresh/Spoiled versions of Apple, Banana, Orange) using OpenCV.
- `requirements.txt`: Project dependencies list.
- `data/`: CSV databases for spectral and IoT observations.
- `models/`: Serialized ML models and label encoders (`.joblib`).
- `sample_images/`: Directory containing generated testing images.

---

## 🛠️ Installation & Running

Ensure you have Python 3.11+ installed.

### 1. Set Active Workspace
Open this directory in your IDE or terminal:
```bash
cd /Users/bhavya/.gemini/antigravity/scratch/fruit-freshness-analyzer
```

### 2. Install Dependencies
Dependencies have already been pre-installed on this system, but you can verify them using:
```bash
pip install -r requirements.txt
```

### 3. Setup and Train (Pre-completed)
The training pipeline has already generated datasets and serialized the models. If you ever need to re-run the pipeline manually:
```bash
python generate_data.py
python train_models.py
python create_samples.py
```

### 4. Run the Streamlit Application
Launch the web interface by running:
```bash
streamlit run app.py
```
This will start a local server (typically at `http://localhost:8501`) and automatically open it in your browser.

---

## 🌐 Deployment to Streamlit Cloud

The application is deployed live on Streamlit Community Cloud:
👉 **[Live Demo](https://fruit-freshness-analyzer-fwzgeq3vxcjb6lagqcozsb.streamlit.app/)**

### Steps to Deploy Your Own Copy:
1. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
2. Click **"New app"**, select your repository, set the branch to `main`, and main file path to `app.py`.
3. Click **"Deploy"**.
4. **First-Time Activation**: On the first launch in the cloud environment, open the sidebar and click the **"⚙️ Setup & Train Models Now"** button. This will dynamically generate the simulated datasets and train the ML classifiers/regressors directly inside the cloud container (taking about 10 seconds).

---

## 🧠 ML Performance Metrics
- **Spectral Freshness Classification**: **100% Accuracy** (Random Forest on synthetic spectra)
- **E-Nose Freshness Classification**: **99.92% Accuracy** (Gradient Boosting on gas sensors)
- **Sugar Content (Brix) Prediction**: **93.98% R² Variance Explained**
- **Moisture Level Prediction**: **96.74% R² Variance Explained**
- **Firmness Prediction**: **97.12% R² Variance Explained**
