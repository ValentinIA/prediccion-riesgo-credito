# Predicción de riesgo de crédito

Proyecto de *machine learning* para predecir si un solicitante **incumplirá el pago de un préstamo**. Cubre todo el ciclo: control de calidad de datos, análisis exploratorio, transformación, preselección de variables, comparación de seis algoritmos y empaquetado del modelo final en un `Pipeline` de scikit-learn serializado y listo para usar.

## Resultados

La métrica principal es el **ROC AUC**. Todos los modelos se entrenan sobre el mismo conjunto de entrenamiento (80 %, estratificado) y se validan con *StratifiedKFold* de 5 particiones sobre el entrenamiento, además del conjunto de test (20 %).

| Modelo | Notebook | AUC validación cruzada | AUC test |
|---|---|---|---|
| Regresión logística (L2, C=1) | `05` | 0,860 | n/d¹ |
| KNN (k=5) | `06` | 0,853 | 0,863 |
| Árbol de decisión (poda `ccp_alpha=0.0015`) | `07` | 0,868 | 0,877 |
| Random Forest | `08` | 0,922 | 0,930 |
| Gradient boosting por histogramas² | `10` | 0,933 | 0,940 |
| **XGBoost** | `09` | **0,938** | **0,945** |


### Modelo del pipeline final: regresión logística

El modelo serializado en `modelos/Regresion_Logistica.joblib` es la regresión logística. Se prioriza un modelo lineal, con coeficientes interpretables y un despliegue sencillo, aunque los modelos de boosting alcancen un AUC unos 8 puntos superior. Esa diferencia indica que hay relaciones no lineales o interacciones que el modelo lineal no captura.

Evaluación del pipeline final sobre el conjunto de test (5.699 solicitudes, umbral 0,5):

| | Precisión | Recall | F1 | Soporte |
|---|---|---|---|---|
| Cumple (0) | 0,87 | 0,95 | 0,91 | 4.462 |
| Impago (1) | 0,74 | 0,49 | 0,59 | 1.237 |

Exactitud global: **0,85**. AUC medio en validación cruzada: **0,859** (rango por partición: 0,852 – 0,866).

| | Predicho: cumple | Predicho: impago |
|---|---|---|
| **Real: cumple** | 4.245 | 217 |
| **Real: impago** | 633 | 604 |

## Datos

[Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) (Kaggle). Según su descripción, simula datos de un buró de crédito. Tiene 32.581 registros y 12 columnas, que en el proyecto se renombran al español:

| Original | En el proyecto | Descripción |
|---|---|---|
| `person_age` | `edad` | Edad del solicitante |
| `person_income` | `ingreso` | Ingreso anual |
| `person_home_ownership` | `propiedad_vivienda` | Régimen de vivienda (`RENT`, `MORTGAGE`, `OWN`, `OTHER`) |
| `person_emp_length` | `duracion_empleo` | Años de antigüedad laboral |
| `loan_intent` | `intencion_prestamo` | Finalidad del préstamo |
| `loan_grade` | `calificacion_prestamo` | Calificación del préstamo (A–G) |
| `loan_amnt` | `monto_prestamo` | Importe solicitado |
| `loan_int_rate` | `tasa_interes` | Tipo de interés |
| `loan_percent_income` | `porcentaje_ingreso` | Importe del préstamo sobre el ingreso |
| `cb_person_default_on_file` | `incumplimiento_historial` | Impago previo registrado (Y/N) |
| `cb_person_cred_hist_length` | `duracion_credito` | Años de historial crediticio |
| `loan_status` | `estado_prestamo` | **Variable objetivo**: 1 = impago, 0 = cumple |

La clase positiva (impago) representa alrededor del **22 %** de los registros. No se aplica balanceo de clases.

## Flujo de trabajo

| Notebook | Contenido |
|---|---|
| `01_Importacion_Calidad_Datos` | Renombrado, tipos, 165 duplicados eliminados, nulos (`duracion_empleo`: 887, `tasa_interes`: 3.095) y valores atípicos (edad > 100, antigüedad laboral > 60 años) |
| `02_EDA` | Frecuencias, estadísticos y distribuciones. Se descartan las 106 filas con `propiedad_vivienda = OTHER` y se aplica `log1p` a `ingreso` |
| `03_Tranformació` | *One-hot* (vivienda, finalidad), codificación ordinal de la calificación (G → A), codificación binaria del historial y escalado MinMax |
| `04_Preseleccion_Variables` | Información mutua, correlaciones fuertes, RFECV con XGBoost e importancia por permutación. Partición 80/20 estratificada |
| `05` – `10` | Regresión logística, KNN, árbol de decisión, Random Forest, XGBoost y gradient boosting por histogramas, con búsqueda de hiperparámetros y validación cruzada |
| `11_Pipeline` | Limpieza reproducible (`scripts/limpieza_datos.py`) y `Pipeline` con `ColumnTransformer` + regresión logística, serializado con `joblib` |
| `12_Test` | Evaluación del pipeline cargado desde disco: informe de clasificación, matriz de confusión y AUC por validación cruzada |

### Variables del modelo final

| Variable | Tratamiento |
|---|---|
| `ingreso` | `log1p` + MinMax |
| `tasa_interes`, `porcentaje_ingreso`, `duracion_empleo`, `duracion_credito` | MinMax |
| `propiedad_vivienda` | *One-hot* con `OWN` y `RENT` (`MORTGAGE` y `OTHER` como referencia) |
| `intencion_prestamo` | *One-hot* con `DEBTCONSOLIDATION`, `HOMEIMPROVEMENT`, `MEDICAL` y `VENTURE` (`EDUCATION` y `PERSONAL` como referencia) |

Variables descartadas en la preselección:

| Variable | Motivo |
|---|---|
| `edad`,`intencion_prestamo_EDUCATION`, `intencion_prestamo_PERSONAL` | Información mutua con el objetivo ≈ 0 |
| `calificacion_prestamo`, `propiedad_vivienda_MORTGAGE` | Correlaciones fuertes |
| `incumplimiento_historial`,`monto_prestamo`,`duracion_credito` | Descartada por Recursive Feature Elimination |

## Estructura del repositorio

```
.
├── notebooks/              # Los 12 notebooks, en orden de ejecución
├── scripts/
│   └── limpieza_datos.py   # Función de limpieza reutilizada en el pipeline
├── modelos/
│   └── Regresion_Logistica.joblib
├── datos/                  # No se versiona (ver .gitignore)
│   ├── originales/         # credit_risk_dataset.csv
│   ├── procesados/
│   └── entrenamiento/
├── environment.yml
└── README.md
```

## Cómo reproducirlo

**1. Crear el entorno con conda**

```bash
git clone https://github.com/<tu-usuario>/prediccion-riesgo-credito.git
cd prediccion-riesgo-credito
conda env create -f environment.yml
conda activate prediccion-riesgo-credito
python -m ipykernel install --user --name prediccion-riesgo-credito
```

**2. Descargar el dataset** desde [Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) y dejar `credit_risk_dataset.csv` en `datos/originales/`. Con la CLI de Kaggle:

```bash
kaggle datasets download -d laotse/credit-risk-dataset -p datos/originales --unzip
```

**3. Crear las carpetas de trabajo.** `datos/` no se versiona y los notebooks no crean carpetas al guardar.

```bash
# Linux / macOS
mkdir -p datos/originales datos/procesados datos/entrenamiento modelos
```

```powershell
# Windows (PowerShell)
mkdir datos/originales, datos/procesados, datos/entrenamiento, modelos
```

**4. Ejecutar los notebooks** desde `notebooks/` con el kernel `prediccion-riesgo-credito`.

