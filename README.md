# Ridehailing-demand-dashboard

Proyecto end-to-end de ciencia de datos para analizar y predecir la demanda de servicios de ride-hailing utilizando tecnologías Big Data y un dashboard interactivo.

---

## 📊 Descripción del Proyecto

Este proyecto analiza los patrones de demanda de transporte en Chicago integrando:

- Variables temporales (hora, día de la semana)
- Variables espaciales (zonas o community areas)
- Variables meteorológicas (temperatura)
- Factores externos (festivos)

El objetivo es **optimizar la asignación de vehículos**, reducir tiempos de espera y mejorar la eficiencia en la movilidad urbana.

---

## 🧱 Arquitectura del Proyecto

El proyecto sigue una arquitectura moderna de datos:

- **Capture:** Taxi Trips, Weather (NOAA), Holidays, Zones  
- **Ingest:** Limpieza y transformación con PySpark  
- **Store:** Data Lake (Parquet + CSV)  
- **Compute:** Feature engineering y modelado de demanda  
- **Use:** Dashboard interactivo en Streamlit  

---

## 📈 Funcionalidades Principales

- 📊 Análisis de demanda por hora y día  
- 🔥 Mapa de calor (heatmap) de patrones temporales  
- 🏙️ Top zonas con mayor demanda  
- 🌍 Mapa coroplético interactivo  
- 🌦️ Impacto de clima y festivos  
- 🤖 Comparación de modelos de Machine Learning  
- 🧪 Evaluación de features (ej: lag_demand)  

---

## 🚀 Demo en Vivo

👉 (Aquí pegarás tu link de Streamlit cuando lo publiques)

---

## 🛠️ Tecnologías Utilizadas

- PySpark  
- Pandas  
- Altair  
- Streamlit  
- Folium  

---

## 📌 Principales Hallazgos

- La demanda presenta **patrones temporales claros** (horas pico)  
- Existe una **concentración espacial** en zonas centrales y O’Hare  
- El clima y los festivos impactan significativamente la movilidad  
- El modelo **Gradient Boosting** mostró mejor desempeño  
- Las variables tipo **lag_demand** mejoran la precisión predictiva  

---

## 📁 Estructura del Proyecto
Ridehailing-dashboard-Streamlit/
│
├── app.py
├── requirements.txt
├── README
├── data/
│ ├── final_dataset.csv
│ └── chicago_geo.json

---

## ⚠️ Nota

Este proyecto tiene fines educativos y demuestra habilidades en ingeniería de datos, análisis, modelado y big data.

---

## 👤 Autor

Global_Proyect_Maestria en Data Science & Big Data

ALUMNO 1: Alberto Miranda   
ALUMNO 2: María José Mangas Gutiérrez  
ALUMNO 3: Santiago Ricardo José Mendoza
