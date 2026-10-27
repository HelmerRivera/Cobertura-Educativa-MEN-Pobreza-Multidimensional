# 🎓 TerritorioPulse – Observatorio Espacial de Desarrollo Humano (Colombia)

> **Modelado econométrico espacial (SpVAR) y análisis exploratorio (ESDA)** de la relación entre la Cobertura Educativa Bruta (MEN) y el Índice de Pobreza Multidimensional - IPM (DANE) a nivel departamental (2018–2023).

---

## 📌 Descripción general

**TerritorioPulse** es una herramienta analítica interactiva desarrollada en Python que combina ciencia de datos espaciales, modelado econométrico de vectores autoregresivos espaciales (SpVAR) y visualización inmersiva mediante **Gradio** y **Plotly**. 

El sistema aísla **cuatro mecanismos de transmisión espacial**:
1. **Autodinámica espacial:** Persistencia e inercia de la variable en el mismo departamento.
2. **Cruce local:** Efecto cruzado contemporáneo entre la pobreza y la educación dentro del territorio.
3. **Spillover espacial directo:** Externalidades e influencia de los departamentos vecinos sobre la misma variable.
4. **Spillover espacial cruzado:** Efecto de las condiciones socioeconómicas de los vecinos sobre la variable focal.

---

## 🚀 Características principales

- **Geometría oficial DANE:** Integración directa con los límites departamentales de Colombia mediante GeoJSON oficial.
- **ESDA robusto:** Evaluación automática de matrices de pesos espaciales (*Queen*, *Rook*, *K-Nearest Neighbors*), cálculo de índices globales de **Moran** y **Geary**, scatterplots y **mapas LISA** (Local Indicators of Spatial Association).
- **Visualización 3D Avanzada:** Gráficos de dispersión tridimensional y superficies interpoladas que relacionan longitud, latitud y niveles de IPM con animación temporal (2018–2023).
- **Dashboard Interactivo (Gradio):** Interfaz web completa dividida en pestañas para mapas coropléticos, experiencias 3D, análisis LISA y coeficientes SpVAR.
- **Informe Ejecutivo Automatizado:** Módulo de interpretabilidad orientado a tomadores de decisiones de política pública.

---

## 🛠️ Tecnologías y librerías utilizadas

- **Análisis Espacial:** `libpysal`, `esda`, `spreg`, `geopandas`, `splot`, `mapclassify`, `contextily`
- **Econometría y Estadística:** `statsmodels`, `scikit-learn`, `scipy`, `numpy`, `pandas`
- **Visualización:** `matplotlib`, `seaborn`, `plotly`, `folium`
- **Interfaz de Usuario:** `gradio`

---

## ⚙️ Instalación y ejecución rápida

Puedes clonar este repositorio y ejecutar el código localmente o abrirlo directamente en **Google Colab**.

### 1. Clonar el repositorio
```bash
git clone [https://github.com/tu-usuario/territoriopulse.git](https://github.com/tu-usuario/territoriopulse.git)
cd territoriopulse
