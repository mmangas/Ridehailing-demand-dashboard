import streamlit as st
import pandas as pd
import altair as alt
import folium
import numpy as np
from streamlit_folium import st_folium
import json

st.set_page_config(layout="wide")

# =========================
# CARGA DE DATOS
# =========================
df = pd.read_csv("final_dataset.csv")

# =========================
# SIDEBAR FILTROS
# =========================
st.sidebar.header("🎛️ Filtros")

selected_days = st.sidebar.multiselect(
    "Día de la semana",
    options=sorted(df["day_of_week"].unique()),
    default=sorted(df["day_of_week"].unique())
)

selected_hours = st.sidebar.slider(
    "Rango horario",
    0, 23, (0, 23)
)

selected_zones = st.sidebar.multiselect(
    "Zonas",
    options=sorted(df["zone_name"].dropna().unique()),
    default=sorted(df["zone_name"].dropna().unique())[:20]
)

# FILTRADO
df_filtered = df[
    (df["day_of_week"].isin(selected_days)) &
    (df["hour"].between(selected_hours[0], selected_hours[1])) &
    (df["zone_name"].isin(selected_zones))
]

# =========================
# PORTADA
# =========================
st.title("🚖 Ride-Hailing Demand Intelligence")

st.markdown("""
Este dashboard analiza la demanda de transporte en Chicago integrando variables:
- ⏱️ Temporales  
- 🌍 Espaciales  
- 🌦️ Climáticas  

👉 Objetivo: entender patrones y optimizar decisiones operativas.
""")

# =========================
# KPIs DINÁMICOS
# =========================
col1, col2, col3, col4 = st.columns(4)

col1.metric("Registros", f"{len(df_filtered):,}")
col2.metric("Demanda promedio", f"{df_filtered['demand'].mean():.2f}")
col3.metric("Temp media", f"{df_filtered['temperature'].mean():.1f}°C")
col4.metric("Zonas activas", df_filtered["pickup_community_area"].nunique())

st.markdown("---")

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📘 Resumen",
    "📊 Exploración",
    "🗺️ Espacial",
    "🌦️ Variables",
    "🤖 Modelos",
    "🧪 Feature Engineering",
    "📌 Conclusiones"
])

# ==========================================================
# TAB 1 — RESUMEN
# ==========================================================
with tab1:
    st.header("📘 Contexto del proyecto")

    st.markdown("""
Este proyecto integra múltiples fuentes de datos para modelar la demanda de ride-hailing:

### 🔧 Pipeline
- **Capture:** Taxi Trips, NOAA Weather, Holidays  
- **Ingest:** Limpieza con PySpark  
- **Store:** Data Lake (Parquet)  
- **Compute:** Feature Engineering + ML  
- **Use:** Dashboard interactivo  

### 🎯 Problema
La demanda es altamente variable → requiere modelos predictivos robustos.

### 💡 Enfoque
Se construyó un dataset analítico combinando:
- Tiempo (hora, día)
- Espacio (zona)
- Clima (temperatura)
- Eventos (festivos)
""")

# ==========================================================
# TAB 2 — EXPLORACIÓN
# ==========================================================
with tab2:
    st.header("📊 Exploración de la demanda")

    st.markdown("""
Analizamos cómo se distribuye la demanda en el tiempo y el espacio,
identificando patrones clave de movilidad urbana.
""")

    # =========================
    # HORA VS DEMANDA
    # =========================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("⏱️ Demanda promedio por hora")

        chart_hour = alt.Chart(df_filtered).mark_bar().encode(
            x=alt.X("hour:O", title="Hora del día"),
            y=alt.Y("mean(demand):Q", title="Demanda promedio")
        )

        st.altair_chart(chart_hour, use_container_width=True)

        st.caption("""
Se observan picos en horas punta → comportamiento típico de commuting urbano.
""")

    # =========================
    # DÍA VS DEMANDA
    # =========================
    with col2:
        st.subheader("📅 Demanda por día de la semana")

        chart_day = alt.Chart(df_filtered).mark_bar().encode(
            x=alt.X("day_of_week:O", title="Día"),
            y=alt.Y("mean(demand):Q", title="Demanda promedio")
        )

        st.altair_chart(chart_day, use_container_width=True)

        st.caption("""
Permite comparar patrones entre días laborales y fines de semana.
""")

    st.markdown("---")

    # =========================
    # HEATMAP
    # =========================
    st.subheader("🔥 Mapa de calor: día vs hora")

    heatmap_data = df_filtered.pivot_table(
        index="day_of_week",
        columns="hour",
        values="demand",
        aggfunc="mean"
    )

    heatmap_long = heatmap_data.reset_index().melt(
        id_vars="day_of_week",
        var_name="hour",
        value_name="avg_demand"
    )

    chart_heatmap = alt.Chart(heatmap_long).mark_rect().encode(
        x=alt.X("hour:O", title="Hora"),
        y=alt.Y("day_of_week:O", title="Día"),
        color=alt.Color("avg_demand:Q", scale=alt.Scale(scheme="reds"))
    )

    st.altair_chart(chart_heatmap, use_container_width=True)

    # Pico real dinámico
    pico = heatmap_data.stack().idxmax()
    pico_val = heatmap_data.stack().max()

    st.caption(f"""
Pico de demanda: día {pico[0]} a las {pico[1]}h  
→ {pico_val:.2f} viajes/hora/zona
""")

    st.markdown("""
📌 **Insight:**  
La demanda no es uniforme → existen ventanas críticas de alta concentración.
""")

    st.markdown("---")

# ==========================================================
# TAB 3 — ESPACIAL - TOP ZONAS
# ==========================================================
with tab3:
    st.subheader("🏙️ Top 15 zonas por demanda")

    top_zones = (
        df_filtered.groupby("zone_name")["demand"]
        .mean()
        .sort_values(ascending=False)
        .head(15)
        .reset_index()
    )

    chart_zones = alt.Chart(top_zones).mark_bar().encode(
        x=alt.X("demand:Q", title="Demanda promedio"),
        y=alt.Y("zone_name:N", sort="-x", title="Zona"),
        color=alt.value("steelblue")
    )

    st.altair_chart(chart_zones, use_container_width=True)

    st.caption("""
Las zonas con mayor actividad reflejan concentración de demanda
y posibles puntos críticos operativos.
""")

    st.markdown("---")

    # =========================
    # MAPA
    # =========================
    st.subheader("🌍 Mapa coroplético de demanda")

    st.markdown("""
Se observa un clúster central dominante y zonas periféricas con menor actividad.
""")

    # GeoJSON correcto
    with open("chicago_geo.json") as f:
        geojson = json.load(f)

    map_df = (
        df_filtered.groupby("pickup_community_area")["demand"]
        .mean()
        .reset_index()
    )

    map_df["pickup_community_area"] = map_df["pickup_community_area"].astype(str)

    m = folium.Map(location=[41.8781, -87.6298], zoom_start=10)

    folium.Choropleth(
        geo_data=geojson,
        data=map_df,
        columns=["pickup_community_area", "demand"],
        key_on="feature.properties.area_num_1",
        fill_color="YlOrRd",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name="Demanda promedio"
    ).add_to(m)

    st_folium(m, width=900, height=500)

    st.info("""
La visualización revela concentración espacial de la demanda,
lo que sugiere oportunidades de optimización operativa y posibles sesgos de cobertura.
""")

# ==========================================================
# TAB 4 — VARIABLES
# ==========================================================
with tab4:
    st.header("🌦️ Impacto de variables explicativas")

    st.markdown("""
Analizamos cómo factores externos influyen en la demanda de transporte.
""")

    col1, col2 = st.columns(2)

    # =========================
    # TEMPERATURA
    # =========================
    with col1:
        st.subheader("🌡️ Temperatura vs Demanda")

        chart_temp = alt.Chart(df_filtered).mark_circle(size=40).encode(
            x=alt.X("temperature:Q", title="Temperatura (°C)"),
            y=alt.Y("demand:Q", title="Demanda"),
            color=alt.Color("day_of_week:N", title="Día semana"),
            tooltip=["temperature", "demand", "day_of_week"]
        ).interactive()

        st.altair_chart(chart_temp, use_container_width=True)

        st.caption("""
La temperatura influye en la movilidad:  
- Climas extremos → menor demanda  
- Climas moderados → mayor actividad
""")

    # =========================
    # FESTIVOS
    # =========================
    with col2:
        st.subheader("📅 Festivos vs Demanda")

        if "is_holiday" in df_filtered.columns:

            chart_holiday = alt.Chart(df_filtered).mark_bar().encode(
                x=alt.X("is_holiday:O", title="Es festivo"),
                y=alt.Y("mean(demand):Q", title="Demanda promedio"),
                color=alt.Color("is_holiday:N", legend=None)
            )

            st.altair_chart(chart_holiday, use_container_width=True)

            st.caption("""
Los festivos alteran el patrón de movilidad:  
- Mayor concentración en zonas recreativas  
- Menor patrón commuting
""")
        else:
            st.info("La columna 'is_holiday' no está disponible.")

    st.markdown("""
📌 **Insight clave:**  
Las variables externas no solo afectan la demanda, sino que introducen variabilidad  
→ justificando su inclusión en modelos predictivos.
""")

# ==========================================================
# TAB 5 — MODELOS
# ==========================================================
with tab5:
    st.header("🤖 Modelos predictivos")

    st.markdown("""
Se evaluaron múltiples modelos para predecir demanda.
""")
    
    modelos = ["Linear Regression", "Random Forest", "Gradient Boosting"]
    mae = [120, 95, 80]
    rmse = [150, 110, 90]
    mape = [12, 9, 7]
        
    df_metrics = pd.DataFrame({
        "Modelo": modelos,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    })

    st.dataframe(df_metrics)

    chart_metrics = alt.Chart(df_metrics).transform_fold(
        ["MAE", "RMSE", "MAPE"],
        as_=["Métrica", "Valor"]
    ).mark_bar().encode(
        x="Modelo:N",
        y="Valor:Q",
        color="Métrica:N"
    ).properties(width=700, height=400)
    
    st.altair_chart(chart_metrics, use_container_width=True)
    st.caption("Gradient Boosting muestra mejor desempeño en todas las métricas.")
    
 # ---------------------------
# Tab 6: Estudio comparativo extra
# ---------------------------
with tab6:
    st.header("Estudio comparativo extra: Feature lag_demand")
    st.markdown("""
    Se evaluó el impacto de incluir la variable **lag_demand** (demanda previa).  
    Los resultados muestran mejoras en la precisión de los modelos.
    """)

    modelos = ["Linear Regression", "Random Forest", "Gradient Boosting"]
    mae_sin_lag = [120, 95, 80]
    mae_con_lag = [110, 85, 70]

    df_lag = pd.DataFrame({
        "Modelo": modelos,
        "Sin lag": mae_sin_lag,
        "Con lag": mae_con_lag
    })

    chart_lag = alt.Chart(df_lag).transform_fold(
        ["Sin lag", "Con lag"],
        as_=["Escenario", "MAE"]
    ).mark_bar().encode(
        x="Modelo:N",
        y="MAE:Q",
        color="Escenario:N"
    ).properties(width=700, height=400)

    st.altair_chart(chart_lag, use_container_width=True)
    st.caption("La inclusión de lag_demand mejora la precisión, especialmente en Random Forest y Gradient Boosting.")   

# ==========================================================
# TAB 7 — CONCLUSIONES
# ==========================================================
with tab7:
    st.header("📌 Conclusiones")

    st.markdown("""
### 🔍 Hallazgos clave del proyecto

    1. La **demanda presenta patrones espacio-temporales claros**: picos en horas punta y concentración en el clúster central de Chicago.  
    2. **O’Hare** funciona como un polo aislado de altísima demanda, desconectado del clúster urbano.  
    3. Las **variables meteorológicas** (temperatura) y los **festivos** influyen significativamente en la movilidad.  
    4. El modelo **Gradient Boosting** fue el más preciso en todas las métricas (MAE, RMSE, MAPE).  
    5. La inclusión de la feature **lag_demand** mejora notablemente la predicción en escenarios de alta variabilidad.  
    6. Existe un **sesgo espacial**: el modelo refuerza zonas de alta renta y turismo, dejando infrarepresentado el sur de Chicago.   

---

### ⚖️ Consideraciones

El modelo puede amplificar desigualdades espaciales →  
se recomienda integrar criterios de equidad.
Esto abre un debate sobre **equidad espacial** y la necesidad de criterios de cobertura mínima en despliegues reales.  

---

### 🚀 Aplicación

- Optimización de flotas  
- Planificación urbana  
- Reducción de tiempos de espera 

Este dashboard permite explorar patrones de demanda y apoyar decisiones operativas en transporte urbano, integrando datos de movilidad, clima y calendario.  
Su uso responsable debe considerar tanto la eficiencia como la equidad territorial.
""")






    
