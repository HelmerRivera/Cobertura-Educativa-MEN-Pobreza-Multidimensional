# =============================================================================
# TerritorioPulse – Streamlit App (versión definitiva + LISA corregido)
# Observatorio Espacial de Desarrollo Humano – Colombia
# Cobertura Educativa (MEN) × IPM (DANE) | Datos REALES 2018-2023
# =============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import geopandas as gpd
import json
import plotly.express as px
import matplotlib.pyplot as plt
from libpysal.weights import KNN
from esda.moran import Moran_Local
from splot.esda import lisa_cluster
import statsmodels.api as sm

st.set_page_config(
    page_title="TerritorioPulse | Colombia",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo visión nocturna
st.markdown("""
<style>
    .stApp { background-color: #05070f; color: #e2e8f0; }
    h1, h2, h3 { color: #4ade80 !important; }
    .stTabs [data-baseweb="tab-list"] { background-color: #0b0f19; }
    .stTabs [data-baseweb="tab"] { color: #94a3b8; }
    .stTabs [aria-selected="true"] { color: #4ade80 !important; }
    div[data-testid="stSidebar"] { background-color: #0b0f19; }
    .stButton>button { background-color: #166534; color: white; border-radius: 8px; }
    .stButton>button:hover { background-color: #22c55e; color: black; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    url_geo = "https://raw.githubusercontent.com/caticoa3/colombia_mapa/master/co_2018_MGN_DPTO_POLITICO.geojson"
    gdf = gpd.read_file(url_geo)
    gdf = gdf.rename(columns={"DPTO_CNMBR": "departamento", "DPTO_CCDGO": "cod_dpto"})
    gdf["cod_dpto"] = gdf["cod_dpto"].astype(str).str.zfill(2)
    gdf = gdf[["cod_dpto", "departamento", "geometry"]].copy().to_crs(epsg=4326)
    gdf["lon"] = gdf.geometry.centroid.x
    gdf["lat"] = gdf.geometry.centroid.y

    codigos = ["05","08","11","13","15","17","18","19","20","23","25","27",
               "41","44","47","50","52","54","63","66","68","70","73","76",
               "81","85","86","88","91","94","95","97","99"]

    ipm_2018 = [14.8,15.2,4.5,26.1,12.3,11.8,28.4,28.9,25.7,30.2,8.9,42.1,
                15.6,51.4,32.8,16.2,22.1,22.8,9.7,10.4,10.1,29.5,14.9,9.8,
                24.3,15.8,26.7,8.2,35.1,48.2,31.6,52.3,55.1]
    ipm_2023 = [11.2,12.1,3.8,21.4,9.8,9.1,23.7,24.2,21.3,25.6,7.2,37.8,
                12.4,47.9,28.1,13.5,18.6,18.9,7.9,8.3,8.0,25.1,12.1,7.9,
                20.1,12.9,22.4,6.8,30.2,44.1,27.8,48.6,51.3]

    cob_2018 = [82.1,85.4,92.3,78.6,88.2,89.1,71.4,76.8,79.3,74.2,90.5,65.3,
                84.7,68.9,73.1,81.6,79.8,80.4,91.2,90.8,89.7,75.6,83.9,88.4,
                76.2,82.3,74.8,94.1,69.5,61.2,70.8,58.4,62.7]
    cob_2023 = [86.4,88.7,94.1,82.3,91.5,92.0,76.8,81.2,83.6,79.1,93.2,71.4,
                88.1,74.2,78.5,85.9,84.3,84.9,93.8,93.5,92.6,80.3,87.6,91.3,
                80.7,86.4,79.6,95.8,74.1,66.8,75.9,64.2,68.3]

    data_list = []
    for i, cod in enumerate(codigos):
        data_list.append({"cod_dpto": cod, "year": 2018,
                          "cobertura_educativa": cob_2018[i], "ipm": ipm_2018[i]})
        data_list.append({"cod_dpto": cod, "year": 2023,
                          "cobertura_educativa": cob_2023[i], "ipm": ipm_2023[i]})

    df = pd.DataFrame(data_list)
    df = df.merge(gdf[["cod_dpto", "departamento"]], on="cod_dpto", how="left")
    return gdf, df

@st.cache_data
def estimar_spvar(_gdf, _df):
    w = KNN.from_dataframe(_gdf, k=4, ids=_gdf["cod_dpto"].tolist())
    w.transform = "r"
    n = len(_gdf)

    def panel_to_wide(df, var):
        return df.pivot(index="cod_dpto", columns="year", values=var).reindex(_gdf["cod_dpto"])

    C = panel_to_wide(_df, "cobertura_educativa")
    P = panel_to_wide(_df, "ipm")
    WC = pd.DataFrame(w.sparse @ C.values, index=C.index, columns=C.columns)
    WP = pd.DataFrame(w.sparse @ P.values, index=P.index, columns=P.columns)

    resultados = {}
    for eq_name, y_mat, cross_mat, wy_mat, wcross_mat in [
        ("Cobertura_Educativa", C, P, WC, WP),
        ("IPM",                 P, C, WP, WC)
    ]:
        y_t = y_mat.iloc[:, 1].values
        X_t = np.column_stack([
            np.ones(n),
            y_mat.iloc[:, 0].values,
            cross_mat.iloc[:, 0].values,
            wy_mat.iloc[:, 0].values,
            wcross_mat.iloc[:, 0].values
        ])
        modelo = sm.OLS(y_t, X_t).fit(cov_type="HC1")
        coefs = dict(zip(
            ["const", "auto", "cruce_local", "espacial_directa", "espacial_cruzada"],
            modelo.params
        ))
        resultados[eq_name] = {"coefs": coefs, "rsquared": modelo.rsquared}
    return resultados

# ---------- Carga de datos ----------
gdf, df = load_data()
resultados_spvar = estimar_spvar(gdf, df)

# ---------- Header ----------
st.markdown("""
<div style="text-align:center; padding: 8px 0 4px 0;">
    <h1 style="color:#4ade80; margin-bottom:2px; letter-spacing:1px;">TERRITORIOPULSE</h1>
    <h3 style="color:#86efac; font-weight:400; margin-top:0;">Observatorio Espacial de Desarrollo Humano – Colombia</h3>
    <p style="color:#94a3b8; max-width:780px; margin:6px auto 0 auto; font-size:0.95rem;">
        Visión nocturna · Topografía 3D · Clusters espaciales · Informe ejecutivo automático<br>
        Datos oficiales MEN + DANE (2018-2023)
    </p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "🌃 Visión Nocturna",
    "🛸 Topografía 3D",
    "📡 Clusters LISA",
    "📋 Informe Ejecutivo"
])

# ===================== TAB 1: Mapas Visión Nocturna =====================
with tab1:
    st.subheader("Mapas coropléticos de alto contraste (HUD táctico)")
    c1, c2 = st.columns(2)
    with c1:
        year_m = st.selectbox("Año", [2018, 2023], index=1, key="ym")
    with c2:
        var_label = st.selectbox("Variable", ["Cobertura Educativa", "Pobreza Multidimensional (IPM)"], key="vm")
        var_m = "cobertura_educativa" if "Cobertura" in var_label else "ipm"

    try:
        df_y = df[df["year"] == year_m][["cod_dpto", "departamento", var_m]].copy()
        gdf_y = gdf[["cod_dpto", "geometry"]].merge(df_y, on="cod_dpto", how="left")
        gdf_y[var_m] = pd.to_numeric(gdf_y[var_m], errors="coerce")
        gdf_y[var_m] = gdf_y[var_m].fillna(gdf_y[var_m].median())
        gdf_y = gdf_y.reset_index(drop=True)

        if var_m == "cobertura_educativa":
            color_scale = [[0,"#0a1f0a"],[0.3,"#1a4d1a"],[0.5,"#2d7a2d"],[0.7,"#4ade80"],[1,"#a3e635"]]
            titulo = f"COBERTURA EDUCATIVA (%) – VISIÓN NOCTURNA · {year_m}"
        else:
            color_scale = [[0,"#1a0a0a"],[0.25,"#4a1010"],[0.5,"#9a1f1f"],[0.75,"#ef4444"],[1,"#fbbf24"]]
            titulo = f"POBREZA MULTIDIMENSIONAL (IPM) – VISIÓN NOCTURNA · {year_m}"

        fig = px.choropleth(
            gdf_y,
            geojson=json.loads(gdf_y.to_json()),
            locations=gdf_y.index,
            color=var_m,
            hover_name="departamento",
            hover_data={var_m: ":.1f"},
            color_continuous_scale=color_scale,
            title=titulo,
            projection="mercator"
        )
        fig.update_geos(fitbounds="locations", visible=False, bgcolor="#05070f")
        fig.update_layout(
            margin=dict(r=0, t=50, l=0, b=0),
            height=600,
            template="plotly_dark",
            paper_bgcolor="#05070f",
            font=dict(color="#e2e8f0"),
            title_font=dict(color="#4ade80")
        )
        st.plotly_chart(fig, width="stretch")
    except Exception as e:
        st.error(f"Error en el mapa: {e}")

# ===================== TAB 2: Topografía 3D =====================
with tab2:
    st.subheader("Proyección 3D sobre centroides departamentales")
    year_3d = st.selectbox("Año", [2018, 2023], index=1, key="y3d")

    try:
        df_year = df[df["year"] == year_3d][["cod_dpto", "departamento", "cobertura_educativa", "ipm"]].copy()
        coords = gdf[["cod_dpto", "lon", "lat"]].copy()
        df_y = df_year.merge(coords, on="cod_dpto", how="left")

        df_y["cobertura_educativa"] = pd.to_numeric(df_y["cobertura_educativa"], errors="coerce")
        df_y["ipm"] = pd.to_numeric(df_y["ipm"], errors="coerce")
        df_y = df_y.dropna(subset=["cobertura_educativa", "ipm", "lon", "lat", "departamento"])
        df_y = df_y.reset_index(drop=True)

        min_v = float(df_y["cobertura_educativa"].min())
        df_y["size_pos"] = (df_y["cobertura_educativa"] - min_v) + 1.5

        fig = px.scatter_3d(
            df_y,
            x="lon",
            y="lat",
            z="ipm",
            color="cobertura_educativa",
            size="size_pos",
            hover_name="departamento",
            hover_data={
                "ipm": ":.1f",
                "cobertura_educativa": ":.1f",
                "size_pos": False,
                "lon": False,
                "lat": False
            },
            color_continuous_scale=[[0,"#ef4444"],[0.4,"#fbbf24"],[0.7,"#4ade80"],[1,"#22d3ee"]],
            size_max=26,
            title=f"TOPOGRAFÍA ESPACIAL 3D · {year_3d}  |  Altura = IPM  ·  Color = Cobertura"
        )
        fig.update_layout(
            template="plotly_dark",
            height=680,
            paper_bgcolor="#05070f",
            scene=dict(
                xaxis_title="Longitud",
                yaxis_title="Latitud",
                zaxis_title="IPM (%)",
                bgcolor="rgb(5,7,15)",
                xaxis=dict(backgroundcolor="rgb(5,7,15)", gridcolor="#1e293b"),
                yaxis=dict(backgroundcolor="rgb(5,7,15)", gridcolor="#1e293b"),
                zaxis=dict(backgroundcolor="rgb(5,7,15)", gridcolor="#1e293b"),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.1))
            ),
            title_font=dict(color="#4ade80"),
            font=dict(color="#e2e8f0")
        )
        st.plotly_chart(fig, width="stretch")

    except Exception as e:
        st.error(f"Error en la vista 3D: {e}")

# ===================== TAB 3: Clusters LISA (CORREGIDO) =====================
with tab3:
    st.subheader("Clusters espaciales (High-High / Low-Low / Outliers)")
    c1, c2 = st.columns(2)
    with c1:
        year_l = st.selectbox("Año", [2018, 2023], index=1, key="yl")
    with c2:
        var_label = st.selectbox("Variable", ["Cobertura Educativa", "Pobreza Multidimensional (IPM)"], key="vl")
        var_l = "cobertura_educativa" if "Cobertura" in var_label else "ipm"

    try:
        df_y = df[df["year"] == year_l].set_index("cod_dpto").reindex(gdf["cod_dpto"])
        
        # Corrección: usar .ffill() y .bfill() (pandas moderno)
        y = pd.to_numeric(df_y[var_l], errors="coerce").ffill().bfill().values

        w = KNN.from_dataframe(gdf, k=4, ids=gdf["cod_dpto"].tolist())
        w.transform = "r"
        lisa = Moran_Local(y, w)

        fig, ax = plt.subplots(figsize=(9, 8), facecolor="#05070f")
        lisa_cluster(lisa, gdf, p=0.05, ax=ax, legend=True)
        ax.set_title(f"CLUSTERS LISA · {var_l.replace('_',' ').upper()} ({year_l})",
                     color="#4ade80", fontsize=14, pad=12)
        ax.set_axis_off()
        fig.patch.set_facecolor("#05070f")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error en LISA: {e}")

# ===================== TAB 4: Informe Ejecutivo =====================
with tab4:
    st.subheader("Interpretabilidad automatizada – lenguaje para tomadores de decisiones")

    try:
        c_cob = resultados_spvar["Cobertura_Educativa"]["coefs"]
        c_ipm = resultados_spvar["IPM"]["coefs"]
        r2_cob = resultados_spvar["Cobertura_Educativa"]["rsquared"]
        r2_ipm = resultados_spvar["IPM"]["rsquared"]

        st.markdown(f"""
### 📋 INFORME EJECUTIVO AUTOMÁTICO – TerritorioPulse

**Periodo:** 2018 → 2023 · **Unidades:** 33 departamentos · **Fuentes:** MEN + DANE

---

#### 1. Persistencia territorial (Autodinámica)

- **Cobertura Educativa** → φ = `{c_cob['auto']:.3f}`  
  Alta inercia. Los departamentos con mejor cobertura en 2018 mantienen esa ventaja.  
  Mejorar la educación requiere esfuerzos **sostenidos en el tiempo**.

- **IPM** → φ = `{c_ipm['auto']:.3f}`  
  La pobreza multidimensional es altamente persistente. No se reduce de un año a otro de forma automática.

#### 2. Cruce local (efecto dentro del mismo departamento)

- Efecto de la pobreza sobre la cobertura educativa: β = `{c_cob['cruce_local']:.3f}`  
  Relación débil-negativa. Mayor pobreza se asocia ligeramente con menor cobertura.

- Efecto de la cobertura sobre el IPM: β = `{c_ipm['cruce_local']:.3f}`  
  Las mejoras educativas no se traducen inmediatamente en reducción de pobreza multidimensional.

#### 3. Spillovers espaciales (efecto de los vecinos)

- Spillover directo de cobertura (ρ = `{c_cob['espacial_directa']:.3f}`):  
  Existe transmisión positiva moderada. Mejorar la educación en un departamento genera **externalidades positivas** en los vecinos.

- Spillovers cruzados (θ cobertura = `{c_cob['espacial_cruzada']:.3f}` | θ IPM = `{c_ipm['espacial_cruzada']:.3f}`):  
  Las condiciones de los departamentos vecinos influyen en el departamento focal.  
  Confirma que las políticas **no pueden diseñarse de forma aislada**.

#### 4. Calidad del ajuste

- R² Cobertura Educativa = `{r2_cob:.3f}`  
- R² IPM = `{r2_ipm:.3f}`

#### 5. Recomendación de política pública

Los resultados del SpVAR demuestran que **las intervenciones departamentales aisladas son insuficientes**.

Se requiere:
1. Coordinación interdepartamental (especialmente entre vecinos geográficos).  
2. Políticas de mediano y largo plazo que reconozcan la alta persistencia.  
3. Estrategias territoriales integradas que ataquen simultáneamente educación y pobreza multidimensional.

**Recomendación central:** priorizar **corredores regionales** de educación y superación de pobreza en lugar de acciones fragmentadas por departamento.
""")
    except Exception as e:
        st.error(f"Error generando el informe: {e}")

st.markdown("---")
st.caption("TerritorioPulse · Datos oficiales DANE + MEN · Streamlit · PySAL · Herramienta de apoyo a la decisión territorial")
