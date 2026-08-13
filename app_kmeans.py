import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler


# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Taller K-Means Interactiva",
    layout="wide"
)

st.markdown("""
# Taller Práctico: Agrupación con K-Means
### Curso IA / ML Junior

Este aplicativo demuestra el flujo de trabajo de un modelo
de clustering no supervisado.
""")


# --- PASO 1: GENERACIÓN DE DATOS ---
st.sidebar.header("1. Configuración de Datos")

n_muestras = st.sidebar.slider(
    "Número de muestras",
    100,
    1000,
    500
)

n_centros = st.sidebar.slider(
    "Centros reales (Clusters)",
    2,
    8,
    4
)


# Generamos datos sintéticos para el ejercicio
X, y = make_blobs(
    n_samples=n_muestras,
    centers=n_centros,
    cluster_std=1.0,
    random_state=42
)


# --- PASO 2: PREPROCESAMIENTO ---
# Es una buena práctica normalizar los datos antes de K-Means

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

st.markdown("## 2. Preprocesamiento de Datos")

st.write(
    "Los datos han sido normalizados para que tengan "
    "la misma escala."
)


# --- PASO 3: MODELAMIENTO ---
st.sidebar.header("2. Hiperparámetros del Modelo")

k_seleccionado = st.sidebar.number_input(
    "Selecciona el valor de K (Clusters)",
    min_value=1,
    max_value=10,
    value=3
)


# Inicializamos el modelo de scikit-learn
# Se recomienda utilizar n_init para evitar mínimos locales

kmeans = KMeans(
    n_clusters=k_seleccionado,
    init="random",
    n_init=10,
    random_state=0
)


# Simulación de estructura de persistencia
# si estuviéramos guardando resultados en BD

# -- BEGIN WORK --

y_pred = kmeans.fit_predict(X_scaled)

# -- COMMIT --


# --- PASO 4: VISUALIZACIÓN DE RESULTADOS ---

st.markdown(
    f"## 3. Visualización de Resultados (K={k_seleccionado})"
)

col1, col2 = st.columns(2)


# ============================================================
# COLUMNA 1: GRÁFICA
# ============================================================

with col1:

    fig, ax = plt.subplots()

    scatter = ax.scatter(
        X[:, 0],
        X[:, 1],
        c=y_pred,
        cmap="viridis",
        s=30
    )


    # Dibujar los centroides

    centros = scaler.inverse_transform(
        kmeans.cluster_centers_
    )

    ax.scatter(
        centros[:, 0],
        centros[:, 1],
        c="red",
        s=200,
        alpha=0.75,
        marker="X",
        label="Centroides"
    )

    ax.set_title("Agrupación Resultante")

    ax.legend()

    st.pyplot(fig)


# ============================================================
# COLUMNA 2: ANÁLISIS
# ============================================================

with col2:

    st.markdown("""
    ### Análisis del Algoritmo:

    1. **Inicialización**:
       Se seleccionaron K centroides aleatorios.

    2. **Asignación**:
       Cada punto se asignó al centroide más cercano.

    3. **Actualización**:
       Se recalcularon los centros como el promedio
       de sus puntos.

    4. **Convergencia**:
       El proceso se repitió hasta que los centros
       dejaron de moverse.
    """)


# --- PASO 5: MÉTODO DEL CODO ---

st.markdown("---")

st.markdown(
    "## 4. Determinando el valor óptimo de K"
)

st.write(
    "El **Método del Codo** nos ayuda a encontrar el punto "
    "donde añadir más clusters no aporta una mejora significativa."
)


if st.button("Calcular Método del Codo"):

    inercias = []

    K_range = range(1, 11)


    for i in K_range:

        km = KMeans(
            n_clusters=i,
            n_init=10,
            random_state=0
        ).fit(X_scaled)

        inercias.append(km.inertia_)


    # Crear gráfica del método del codo

    fig_codo, ax_codo = plt.subplots()

    ax_codo.plot(
        K_range,
        inercias,
        "bx-"
    )

    ax_codo.set_xlabel(
        "Valor de K"
    )

    ax_codo.set_ylabel(
        "Inercia (Suma de distancias al cuadrado)"
    )

    ax_codo.set_title(
        "Método del Codo"
    )

    st.pyplot(fig_codo)