# Trabajo Práctico Final — Inteligencia Artificial 2026 (UNaB)
## Desarrollo Progresivo de una Arquitectura de Red Neuronal sobre Kuzushiji-MNIST (KMNIST)

**Institución:** Universidad Nacional Guillermo Brown (UNaB)  
**Materia:** Inteligencia Artificial — 2° Cuatrimestre 2026  
**Docente:** Pablo Moreira  
**Dataset Asignado:** Kuzushiji-MNIST (`KMNIST` - 70.000 imágenes de caracteres cursivos japoneses de 28x28 píxeles en escala de grises, 10 clases)  

---

## 1. Estructura del Proyecto

El repositorio y archivo de entrega está estructurado según lo estipulado en la Sección 5.2 de la pauta del TP:

```text
TP_IA2026_APELLIDO_KMNIST/
│
├── TP_IA2026_KMNIST_Final.ipynb   # Notebook Jupyter principal con las 5 Etapas completas
├── requirements.txt               # Dependencias y versiones del entorno
├── README.md                      # Instrucciones de reproducción y ejecución
└── resultados/                    # Gráficos y visualizaciones exportadas en alta resolución (26 figuras)
    ├── 01_eda_ejemplos_clases.png
    ├── 02_eda_histograma_pixeles.png
    ├── 03_eda_boxplot_brillo.png
    ├── 04_eda_outliers_extremos.png
    ├── 05_particion_estratificada.png
    ├── 06_baseline_matriz_confusion.png
    ├── 07_baseline_comparativa_normalizacion_solvers.png
    ├── 08_etapa2_50_errores_baseline.png
    ├── 09_etapa2_resumen_categorias_errores.png
    ├── 10_etapa2_mlp_sin_regularizacion.png
    ├── 11_etapa2_mlp_regularizada_curvas.png
    ├── 12_etapa2_mlp_regularizada_cm_f1.png
    ├── 13_etapa2_heatmap_24_mlps.png
    ├── 14_etapa3_cnn_con_aumentacion_p4.png
    ├── 15_etapa3_cnn_sin_aumentacion_p4.png
    ├── 16_etapa3_cnn_mejor_wd0.png
    ├── 17_etapa3_cnn_vs_pipeline_atribucion.png
    ├── 18_etapa3_impacto_data_augmentation.png
    ├── 19_etapa4_vae_curvas_elbo.png
    ├── 20_etapa4_vae_reconstrucciones.png
    ├── 21_etapa4_vae_espacio_latente_2d.png
    ├── 22_etapa4_vae_prior_interpolacion.png
    ├── 23_etapa4_vae_manifold_grid_10x10.png
    ├── 24_etapa5_tradeoff_complejidad.png
    ├── 25_etapa5_test_matriz_confusion.png
    └── 26_etapa5_sesgo_subgrupos_dev_test.png
```

---

## 2. Requisitos y Entorno de Ejecución

- **Python:** 3.10+ (probado en Python 3.11/3.14)
- **PyTorch:** 2.x
- **Hardware:** CPU estándar o GPU con CUDA (el código detecta automáticamente `cuda` si está disponible, con fallback a `cpu`).

### Instalación de Dependencias

Se recomienda crear un entorno virtual e instalar las librerías indicadas:

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno (Windows)
.\venv\Scripts\activate

# Activar entorno (Linux/macOS)
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

---

## 3. Ejecución del Notebook

1. Abrir Jupyter Notebook o Jupyter Lab:
   ```bash
   jupyter notebook TP_IA2026_KMNIST_Final.ipynb
   ```
2. Ejecutar las celdas secuencialmente de arriba hacia abajo.
3. El dataset KMNIST se descarga automáticamente en la carpeta `./data` en la primera ejecución a través de `torchvision.datasets.KMNIST`.
4. El conjunto de **Test oficial ($10.000$ imágenes)** se mantiene estrictamente aislado y solo es consultado e inferido en la **Etapa 5 (Sección 4.5.2)**.

---

## 4. Resumen de Modelos y Progresión

| Etapa | Modelo / Arquitectura | Parámetros | F1-Macro Dev | F1-Macro Test | Observaciones Clave |
|---|---|---|---|---|---|
| **Etapa 1** | Regresión Logística (Min-Max [0, 1], C=0.1) | 7.850 | 0.8243 | — | Baseline lineal de referencia |
| **Etapa 2** | MLP 3 Capas Ocultas (Dropout=0.2, L2=1e-4, ES) | 242.762 | 0.9652 | — | Resuelve la no linealidad; regularización previene sobreajuste |
| **Etapa 3** | CNN 3 Bloques Convolucionales + Data Aug (wd=0.0) | 93.962 | 0.9911 | **0.9705** | Modelo de mejor generalización espacial |
| **Etapa 3** | Pipeline Secuencial (CNN congelada + LogReg) | 95.252 | 0.9906 | 0.9683 | Desacopla representación de clasificación |
| **Etapa 4** | ConvVAE Generativo (Espacio Latente 2D) | 266.085 | MSE: 0.0701 | — | 50 épocas de entrenamiento; reconstrucción y manifold 10x10 |
| **Etapa 5** | Evaluación Final en Test & Análisis Ético | — | 0.9911 | **0.9705** | Brecha Dev-Test de solo 2.06%; análisis de sesgo por alógrafo |

