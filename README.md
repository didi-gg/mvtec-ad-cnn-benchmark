# MVTec AD CNN Benchmark

## Detección y localización de anomalías industriales mediante modelos convolucionales

Proyecto de maestría en Machine Learning orientado a comparar tres enfoques de detección de anomalías industriales, evaluando su precisión, capacidad de localización y eficiencia computacional

## Pregunta de investigación:
¿De qué manera los ajustes experimentales de tres enfoques basados en CNN influyen en la detección y localización de anomalías industriales, y cómo se relacionan sus resultados con el costo computacional?

## Objetivos
### Objetivo general

Implementar y comparar tres enfoques de detección de anomalías industriales utilizando MVTec AD, mediante experimentos controlados que permitan evaluar mejoras de desempeño y eficiencia
### Objetivos específicos

1. Preparar el dataset y establecer particiones reproducibles
2. Implementar tres métodos basados en redes convolucionales
3. Evaluar el efecto de un ajuste experimental en cada modelo
4. Comparar la capacidad de detección
5. Medir el tiempo de entrenamiento, la latencia y el consumo de memoria
6. Analizar resultados, errores y limitaciones
7. Desarrollar una demostración interactiva con Streamlit

## Dataset
**Fuente:** [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)
**Licencia:** CC BY-NC-SA 4.0.

Se utilizarán las 15 categorías del dataset
| Característica                   | Cantidad |
| -------------------------------- | -------: |
| Imágenes totales                 | 5.354    |
| Entrenamiento (normales) | 3.629    |
| Test                     | 1.725    |
| Imágenes defectuosas de test     | 1.258    |
### Particiones

Se dividió el entrenamiento por categoría utilizando una proporción 80/20 y la semilla **1926**
| Partición     | Imágenes |
| ------------- | -------: |
| Entrenamiento | 2.903    |
| Validación    | 726      |
| Test  | 1.725    |
La proporción 80/20 busca conservar suficientes ejemplos para aprender patrones normales y reservar imágenes independientes para validación

La auditoría SHA-256 no encontró duplicados exactos ni coincidencias entre entrenamiento y prueba

Las particiones se almacenan en `results/splits/`

## Modelos
### M1. Autoencoder convolucional (CNN-AE)
Red convolucional codificador–decodificador entrenada desde cero para reconstruir imágenes normales
Las diferencias entre la imagen original y la reconstrucción permiten identificar posibles anomalías.

**Ajuste experimental:** función de pérdida.
- Configuración 1: MSE (error cuadrático medio)
- Configuración 2: L1 (error absoluto medio)

### M2. PatchCore con ResNet18
Método que utiliza una CNN ResNet18 preentrenada en ImageNet para extraer características de imágenes normales

Una selección de estas características se conserva en un banco de memoria y se utiliza para detectar regiones diferentes de los patrones normales

**Ajuste experimental:** porcentaje de características conservadas
- Configuración 1: 1 %.
- Configuración 2: 5 %.

Ambas configuraciones utilizarán todas las imágenes normales de entrenamiento

### M3. Sistema de reconstrucción y segmentación con dos U-Net
Arquitectura compuesta por dos redes convolucionales entrenadas desde cero:

1. Una U-Net reconstruye imágenes normales a partir de imágenes modificadas artificialmente.
2. Otra U-Net identifica las regiones anómalas utilizando la imagen y su reconstrucción.

**Ajuste experimental:** capacidad de las redes.
- Configuración 1: 16–32–64 canales.
- Configuración 2: 16–32–64–128 canales.

Se mantendrá fijo el procedimiento de generación de anomalías sintéticas durante el entrenamiento para aislar el efecto del tamaño de la arquitectura

## Hipótesis y experimentos
| Hipótesis                                                                                            | Ajuste experimental                                                                                  | Cómo se contrasta                                                                           |
| ---------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| **H1. Autoencoder:** la función de pérdida influye en la detección de anomalías.                     | Comparar MSE y L1.                                                                                   | Diferencias de Image-AUROC y AUPRO.                                                         |
| **H2. PatchCore:** conservar más características puede mejorar la detección, pero aumenta el costo computacional. | Comparar 1 % y 5 % de características conservadas.                                                   | Diferencias de Image-AUROC, AUPRO, latencia y memoria.                                      |
| **H3. Dos U-Net:** aumentar la capacidad de las redes puede mejorar la localización, pero incrementa el costo computacional. | Comparar arquitecturas de tres y cuatro niveles.                                                     | Diferencias de AUPRO, IoU, Dice, latencia y memoria.                                        |
| **H4. General:** PatchCore con ResNet18 preentrenada ofrecerá el mejor equilibrio entre detección, localización y eficiencia computacional frente al Autoencoder CNN y al sistema de dos U-Net entrenados desde cero. | Comparar la mejor configuración de cada modelo bajo el mismo protocolo y presupuesto de ajuste local. | Comparar Image-AUROC, AUPRO, latencia p95, memoria y tiempo de ajuste en las 15 categorías. |
Cada experimento modificará una variable principal, manteniendo constantes las demás condiciones relevantes dentro del mismo modelo

Los resultados se analizarán por categoría y mediante el promedio macro de las 15 categorías

## Protocolo de validación
A partir de las 726 imágenes de validación se generarán anomalías sintéticas mediante:

- **CutPaste:** recortar y pegar fragmentos de una imagen para introducir irregularidades
- **Modificación de texturas:** generar defectos artificiales con máscaras de formas irregulares

Se conservarán las máscaras de las regiones modificadas

Los tres modelos utilizarán exactamente el mismo conjunto de validación, compuesto por imágenes normales y sus versiones con anomalías sintéticas

La validación permitirá:
- Comparar configuraciones
- Seleccionar checkpoints
- Establecer umbrales de clasificación y segmentación
- Calcular métricas sin utilizar el test

La generación de anomalías para entrenar las dos U-Net utilizará exclusivamente imágenes del conjunto de entrenamiento y será independiente de las imágenes reservadas para validación

## Métricas de evaluación

### Detección de anomalías
| Métrica             | Propósito                                                          |
| ------------------- | ------------------------------------------------------------------ |
| **Image-AUROC**     | Métrica principal para distinguir imágenes normales y defectuosas. |
| Precision           | Medir la proporción de alertas correctas.                          |
| Recall              | Medir la proporción de defectos detectados.                        |
| F1-score            | Evaluar el equilibrio entre precision y recall.                    |
| Matriz de confusión | Analizar aciertos y errores de clasificación.                      
### Localización de anomalías
| Métrica   | Propósito                                                               |
| --------- | ----------------------------------------------------------------------- |
| **AUPRO** | Métrica principal para evaluar la localización de regiones defectuosas. |
| IoU       | Medir la superposición entre máscaras predichas y reales.               |
| Dice      | Medir la similitud entre las regiones predichas y reales.               |
Los umbrales necesarios para precision, recall, F1, IoU y Dice se fijarán mediante validación, sin utilizar el test.

### Eficiencia computacional
| Métrica                | Propósito                                          |
| ---------------------- | -------------------------------------------------- |
| Tiempo de ajuste local | Medir el costo de entrenar o preparar cada modelo. |
| Latencia p95           | Evaluar el tiempo de inferencia por imagen.        |
| FPS                    | Medir la capacidad de procesamiento.               |
| Memoria máxima         | Comparar los recursos utilizados.                  |
## Comparación
Los tres enfoques se evaluarán bajo condiciones comunes:

- Las mismas 15 categorías de MVTec AD
- Los mismos conjuntos de entrenamiento y validación, con semilla 1926
- Resolución de entrada común de 256 × 256 píxeles, sujeta a verificación en el piloto
- Preprocesamiento compatible con cada arquitectura
- Las mismas imágenes normales y anomalías sintéticas de validación
- Las mismas métricas y test
- Un presupuesto máximo común de tiempo de ajuste local por modelo y categoría
- Registro del tiempo real utilizado, criterios de parada y recursos computacionales

El presupuesto definitivo se establecerá después de realizar un piloto de rendimiento en el hardware disponible

PatchCore utiliza una ResNet18 preentrenada en ImageNet. El costo de este preentrenamiento externo no forma parte del tiempo de ajuste local y se considerará al interpretar los resultados

## Instrucciones de ejecución

Los siguientes comandos deben ejecutarse desde la raíz del repositorio.

### 1. Crear el entorno e instalar dependencias

Se requiere Conda o Miniforge. El script crea el entorno `mvtec-cnn`, instala las dependencias y registra el kernel de Jupyter:

```bash
bash virtual_env.sh
conda activate mvtec-cnn
```

El dataset MVTec AD debe estar disponible en `data/mvtec_anomaly_detection/`, con una carpeta por categoría (por ejemplo, `data/mvtec_anomaly_detection/bottle/`). Las particiones reproducibles se generan con:

```bash
python scripts/prepare_data.py
```

### 2. Ejecutar un piloto de hardware

Antes de lanzar todos los experimentos, se puede medir el costo de los tres modelos en categorías representativas:

```bash
python scripts/pilot.py --category bottle
python scripts/pilot.py --category carpet
python scripts/pilot.py --category toothbrush
```

Los diagnósticos se guardan en `results/hardware/`.

### 3. Preparar la validación sintética

La validación común utiliza anomalías `cutpaste` e `irregular_texture` generadas a partir de las imágenes normales reservadas para validación:

```bash
python scripts/prepare_synthetic_validation.py --all
```

Para procesar una sola categoría:

```bash
python scripts/prepare_synthetic_validation.py --category bottle
```

### 4. Entrenar los modelos

Primero se puede comprobar el plan de ejecución sin entrenar:

```bash
python scripts/run_experiments.py --dry-run
```

Después, se ejecutan las dos variantes de cada modelo en las 15 categorías:

```bash
python scripts/run_experiments.py
```

También es posible limitar la ejecución a un modelo o a determinadas categorías:

```bash
python scripts/run_experiments.py --models patchcore
python scripts/run_experiments.py --categories cable carpet
```

Los checkpoints y las métricas de validación se almacenan en `results/runs/`. Las ejecuciones completas se omiten automáticamente; para repetirlas se puede utilizar `--force`.

### 5. Evaluar en el test

La evaluación selecciona automáticamente la mejor variante según la validación y no vuelve a entrenar ni optimiza umbrales con el test:
La evaluación selecciona automáticamente la mejor variante según la validación y no vuelve a entrenar ni optimiza umbrales con el test:

```bash
python scripts/evaluate.py --dry-run
python scripts/evaluate.py
```

Para evaluar una sola categoría o modelo:

```bash
python scripts/evaluate.py --category bottle --model patchcore
```

Los resultados finales se guardan en `results/final/test/` y los agregados en `results/final/`.

### 6. Generar visualizaciones

Después de completar la evaluación, se pueden generar ejemplos de predicciones y errores:

```bash
python -m scripts.visualize_predictions --category bottle
python -m scripts.visualize_predictions --category tile --examples-per-case 2
```

Las figuras y el índice por imagen se guardan en `results/figures/test_examples/`.

### 7. Ejecutar los notebooks

Para explorar los análisis paso a paso:

```bash
jupyter lab
```

En Jupyter se debe seleccionar el kernel `Python (MVTec CNN)` y ejecutar los notebooks de `notebooks/` en orden numérico.

## Evaluación final
Una vez seleccionadas las configuraciones mediante validación:

1. Congelar los modelos, configuraciones y umbrales
2. Evaluar sobre el test oficial de MVTec AD
3. Calcular métricas de detección y localización por categoría
4. Medir tiempo de ajuste, latencia y memoria
5. Comparar los resultados y contrastar las hipótesis
6. Analizar errores, limitaciones y relación entre desempeño y costo

## Hardware y tecnologías
### Hardware
- Apple M3 Pro
- CPU de 12 núcleos
- GPU Apple de 18 núcleos
- 36 GB de memoria unificada
- macOS ARM64
- Aceleración PyTorch MPS
## Demostración interactiva
Se desarrollará una aplicación en Streamlit que permita:
- Cargar imágenes para su análisis.
- Seleccionar una categoría y uno de los tres modelos.
- Visualizar la imagen y el mapa de anomalías.
- Consultar la puntuación de anomalía y la clasificación obtenida.
- Comparar los resultados de los modelos.
## Referencias
- [MVTec AD — Dataset oficial](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)
- [PatchCore — Implementación de referencia](https://github.com/amazon-science/patchcore-inspection)
- [ResNet18 — Torchvision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
- [U-Net — Artículo original](https://arxiv.org/abs/1505.04597)
- [CutPaste — Artículo científico](https://arxiv.org/abs/2104.04015)