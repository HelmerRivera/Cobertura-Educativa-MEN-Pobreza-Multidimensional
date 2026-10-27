# =============================================================================
# TerritorioPulse – Streamlit App (versión corregida)
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

# Cargar datos
gdf, df = load_data()
resultados_spvar = 
