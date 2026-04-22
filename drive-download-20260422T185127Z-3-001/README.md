# Smart Sericulture Environment System 🐛

**VIT Vellore | Winter Semester 2025-26 Capstone Project**

> Simulation-Based IoT Environmental Control System for Sericulture

---

## Prerequisites
- Python 3.10 or higher
- VS Code (recommended)

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place your dataset
Make sure `weatherHistory.csv` is accessible.  
Default path assumed: `/mnt/user-data/uploads/weatherHistory.csv`  
*(You can change the path in the app sidebar at runtime)*

### 3. Run the app
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## Features

| Tab | Description |
|-----|-------------|
| 🔴 Live Simulation | Animates through dataset row-by-row with live metrics |
| 📊 Data Analysis   | Statistical tables: mean, std, min, max, in-range % |
| 📈 Visualizations  | 6 chart types: time series, scatter, before/after, pie |
| ⚙️ Control Results | Summary metrics + downloadable simulation CSV |
| 📋 Dataset         | Raw data preview and descriptive stats |
| 📖 About System    | Architecture diagram, algorithm, pseudocode, viva Q&A |

## Control Logic
```
IF temperature > 28°C   → Fan ON
IF temperature < 24°C   → Heater ON
IF humidity    < 70%    → Humidifier ON
Multiple conditions     → Multi-Action
ELSE                    → Stable
```

## Optimal Sericulture Conditions (adjustable in sidebar)
- Temperature: 24°C – 28°C
- Humidity: 70% – 85%
