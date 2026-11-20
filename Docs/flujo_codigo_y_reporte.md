# Flujo del código y reporte de estado

**Proyecto:** monitoreo secuencial de traslape espectral en señales IQ  
**Base activa:** `data/raw/raw_data/32G_B2B`  
**Estado del documento:** guía para revisión y coordinación del equipo

## 1. Resumen para el equipo

El proyecto estima si una captura IQ corresponde a una separación de canales con traslape espectral. En vez de clasificar una captura usando todos sus símbolos de una vez, la idea es acumular evidencia por bloques y estudiar cuántos símbolos hacen falta para decidir.

Los módulos y scripts implementan el inventario de datos, la carga y normalización de capturas, un banco de densidades KDE por condición, la actualización bayesiana y trazas de creencia. El notebook [proyecto.ipynb](../notebooks/proyecto.ipynb) añade una primera implementación de SARSA y compara políticas de parada en un subconjunto cercano a la frontera. Esa implementación todavía debe ejecutarse y revisarse con los datos antes de tratar sus métricas como resultados.

## 2. Pregunta y formulación

La pregunta de trabajo puede expresarse así:

> ¿Cuántos símbolos IQ debe observar el monitor antes de declarar traslape o ausencia de traslape, y cómo cambia ese número al variar el costo de observación y los costos de error?

Para el escenario activo, la separación nominal es

$[B_{nom}=R_s(1+\rho)=32(1+0.1)=35.2\;\text{GHz}.]$

La etiqueta geométrica usada por el inventario es `1` si `ChSp < 35.2 GHz` y `0` si `ChSp > 35.2 GHz`. Una separación exactamente igual al umbral queda sin etiqueta dura. Esta etiqueta representa geometría espectral; por sí sola **no demuestra** degradación de BER ni calidad de servicio.

Una captura tiene una condición física fija `c = (ChSp, OSNR)` y una secuencia de observaciones complejas `y = I + jQ`. En cada bloque `B_k`, el código suma log-verosimilitudes de las KDE y calcula el posterior de las condiciones. La probabilidad agregada de traslape es

$[b_k=P(z=1\mid B_1,\ldots,B_k)=\sum_{c:z(c)=1}P(c\mid B_1,\ldots,B_k).]$

## 3. Base de datos y supuestos

La configuración selecciona `32G_B2B` dentro del árbol de datos. En el árbol observado, los archivos son Parquet, organizados por escenario, separación (`ChSp`) y adquisición `songN`. Las columnas RX son `rx_i` y `rx_q`; también aparece `tx_sym`, que el flujo actual no usa para tomar decisiones.

El inventario obtiene `ChSp` del nombre del directorio y OSNR del nombre del archivo. El cargador forma el símbolo complejo y normaliza cada captura para que su potencia media sea uno. Esta normalización facilita comparar formas de constelación, pero elimina diferencias absolutas de potencia entre capturas.

Supuestos actuales:

- Los símbolos de una captura son aproximadamente independientes condicionados a `(ChSp, OSNR)` al acumular la evidencia. La dependencia temporal puede hacer que el posterior se vuelva demasiado seguro.
- Las KDE entrenadas representan las distribuciones IQ de las condiciones conocidas.
- La frontera `35.2 GHz` genera la etiqueta geométrica; no se infiere una etiqueta de BER.
- El procedimiento reproduce archivos completos en modo offline. No solicita nuevas adquisiciones a un equipo durante la operación.
- Solo se usa `32G_B2B` para el experimento principal. Los escenarios `16G_B2B`, `32G_270km_0dBm` y `32G_270km_9dBm` no se mezclan en el banco activo.

## 4. Flujo de extremo a extremo

```text
Parquet/CSV original
        │
        ▼
scripts/build_inventory.py
  escanea, parsea escenario/ChSp/OSNR y etiqueta geométricamente
        │
        ▼
data/processed/inventory.csv
        │
        ├── scripts/explore_data.py → figuras exploratorias
        │
        ▼
scripts/build_kde_bank.py
  separa song1–2 / song3 / song4; ajusta KDE con song1–2
        │
        ▼
data/processed/kde_bank.pkl + kde_bank_meta.json
        │
        ▼
scripts/trace_bk.py
  evalúa capturas song4, actualiza el posterior por bloques
        │
        ▼
data/processed/fig_bk_trace.png
```

### Paso 1: construir el inventario

`src/data/inventory.py` recorre archivos `.parquet` y `.csv`, filtra por `scenarios` y extrae metadatos de la estructura de carpetas y nombres. Cada fila registra `capture_id`, ruta, condición, escenario, etiqueta y marcas de control/ambigüedad. `scripts/build_inventory.py` guarda el resultado en `data/processed/inventory.csv`.

### Paso 2: cargar una adquisición

`src/data/loader.py` abre Parquet o CSV, acepta columnas `rx_i`/`rx_q` o `I`/`Q`, crea `y = I + jQ` y normaliza por potencia media. `split_into_blocks` devuelve bloques consecutivos disjuntos por defecto; descarta el sobrante que no completa un bloque.

### Paso 3: separar adquisiciones y ajustar KDE

`make_split` en `scripts/build_kde_bank.py` identifica la canción a partir del comienzo del nombre de archivo. La configuración actual usa:

| Uso | Adquisiciones |
|---|---|
| Entrenamiento KDE | `song1`, `song2` |
| Validación reservada | `song3` |
| Prueba reservada | `song4` |

La función `build_bank` agrupa las capturas de entrenamiento por `(ChSp, OSNR)`, concatena sus símbolos y selecciona hasta 3.000 centros KDE por condición con semilla fija. La KDE gaussiana 2D usa ancho de banda `0.05`. El banco guarda una densidad por condición conocida; no es directamente un clasificador de traslape.

El split reserva `song3`, pero el script actual no lo usa todavía para selección de hiperparámetros ni calibración. Esa etapa falta antes de presentar `song3` como validación experimental efectiva.

### Paso 4: prior y actualización bayesiana

`src/bayes/update.py` construye un prior balanceado por clase: asigna la mitad de la probabilidad previa a cada clase y la reparte uniformemente entre las condiciones disponibles de esa clase. Para cada observación, la KDE entrega una log-densidad. El código acumula esas log-densidades por condición y normaliza el posterior con `log-sum-exp`.

Se conserva el posterior sobre las condiciones, no solo `b_k`. Esto importa porque distintas combinaciones de ChSp y OSNR pueden producir la misma probabilidad agregada de traslape, pero distintas distribuciones de evidencia futura.

### Paso 5: producir trazas de creencia

`scripts/trace_bk.py` carga el inventario y banco, obtiene las etiquetas, calcula la log-densidad de cada símbolo bajo cada KDE, acumula evidencia y guarda `b_k` cada 100 símbolos por defecto. `--all-test` selecciona las capturas que corresponden a `song4` según el split.

La gráfica muestra cómo cambia la creencia durante la captura. **No elige cuándo parar:** actualmente todas las capturas se procesan hasta el final y no se aplican costos terminales ni costo por bloque.

### Paso 6: exploración descriptiva

`scripts/explore_data.py` genera constelaciones por separación, un barrido de OSNR cerca del umbral y distribuciones de amplitud. Estas figuras sirven para inspección; no entrenan el modelo ni son resultados de decisión por sí mismas.

## 5. Archivos principales

| Archivo | Responsabilidad |
|---|---|
| `configs/experiment.yaml` | Escenario, frontera, tamaño de bloque, split y parámetros KDE |
| `src/data/inventory.py` | Recorrido del dataset y etiquetas geométricas |
| `src/data/loader.py` | Lectura IQ, normalización y partición en bloques |
| `src/kde/bank.py` | KDEs por condición y serialización del banco |
| `src/bayes/update.py` | Prior, actualización posterior y probabilidad de traslape |
| `scripts/build_inventory.py` | Genera el CSV de inventario |
| `scripts/build_kde_bank.py` | Define el split y ajusta el banco KDE |
| `scripts/trace_bk.py` | Calcula las trazas de creencia sobre capturas de prueba |
| `scripts/explore_data.py` | Figuras exploratorias |
| `Docs/viabilidad_sarsa.md` | Formulación y limitaciones de SARSA para este problema |

## 6. Cómo reproducir el flujo

Ejecutar desde la raíz del repositorio, con las dependencias de `requirements.txt` instaladas y los archivos de datos presentes:

```bash
python scripts/build_inventory.py
python scripts/build_kde_bank.py
python scripts/trace_bk.py --all-test
python scripts/explore_data.py
```

Los productos se escriben bajo `data/processed/`, que está excluido del control de versiones. El banco KDE requiere `pyarrow` para leer los Parquet; la dependencia ya está declarada en `requirements.txt`.

## 7. SARSA: cómo encajaría y qué falta

SARSA puede aprender una política de parada simulada. Un episodio sería una captura: en cada paso el agente observa el siguiente bloque o declara una clase. Un estado mínimo debería incluir una representación de la creencia posterior completa (posiblemente discretizada) y los bloques restantes. Las acciones terminales reciben una penalización `-C_FP` o `-C_FN`; observar otro bloque recibe `-c_bloque`.

Para entrenar sin fuga, SARSA puede interactuar con las secuencias `song1–2`, ajustarse con `song3` y evaluarse una sola vez con `song4`. La etiqueta verdadera puede determinar el costo terminal durante el entrenamiento offline, pero no debe estar disponible en el estado ni para seleccionar la acción. La comparación debe incluir al menos una ventana fija y un umbral de confianza, con riesgo y cantidad de símbolos observados.

**Estado de implementación:** SARSA, la política de umbral, la ventana fija y métricas de FPR/FNR/costo están implementados en el notebook. Aún faltan la ejecución reproducible completa, sensibilidad a varias semillas/costos, Brier score, calibración e intervalos por adquisición. El documento [viabilidad_sarsa.md](viabilidad_sarsa.md) desarrolla la recomendación y los límites.

## 8. Pendientes para revisar en equipo

1. Confirmar que `32G_B2B` será el único escenario del resultado principal.
2. Inspeccionar el inventario generado: número de capturas por ChSp, OSNR y `songN`; comprobar que las cuatro canciones existen en las condiciones próximas a 35.2 GHz.
3. Decidir si normalizar cada captura por potencia es compatible con la información que se quiere conservar.
4. Usar `song3` para escoger ancho de banda, tamaño de bloque y discretización; dejar `song4` intacta hasta fijar el método.
5. Implementar y comparar ventana fija, umbral y SARSA antes de afirmar ahorro de símbolos.
6. Evaluar calibración y dependencia temporal; los símbolos consecutivos pueden no ser independientes.
7. Aclarar en el informe que la etiqueta es geométrica y no equivale a una medición de BER.

## 9. Estado de verificación

Este documento describe el código inspeccionado y la configuración actual. La ejecución completa del flujo no se ha verificado en este entorno: `pyarrow` no estaba instalado al revisar el proyecto. Por tanto, no se reportan aquí métricas ni resultados experimentales; primero hay que instalar dependencias y ejecutar la generación del inventario y el banco.
