# Red Neuronal Simple para Predecir el Doble de un Número

Este proyecto implementa una red neuronal de una sola neurona (regresión lineal) en Python, que aprende a predecir el doble de un número. Es ideal para fines educativos y para entender los conceptos básicos de aprendizaje automático, entrenamiento, error y ajuste de parámetros.

## Características
- Entrenamiento con datos del 1 al 20.
- Ajuste automático de pesos y sesgo usando descenso de gradiente.
- Autocorrección para mejorar la predicción si no es exacta.
- Gráfica del error cuadrático medio durante el entrenamiento.
- Interfaz por consola para ingresar el número a predecir.

## Requisitos
- Python 3.7 o superior
- Numpy
- Matplotlib

## Instalación
1. Descarga el archivo `ejemplo.py` desde este repositorio.
2. Instala las dependencias necesarias ejecutando en la terminal:
   ```bash
   pip install numpy matplotlib
   ```

## Uso
1. Ejecuta el script en la terminal:
   ```bash
   python ejemplo.py
   ```
2. Ingresa un número cuando se te solicite. El programa mostrará el resultado esperado, la predicción del modelo y la curva de error del entrenamiento.

## Estructura del Código
- **Ingreso de datos:** Solicita al usuario un número para predecir su doble.
- **Entrenamiento:** Ajusta los parámetros de la red para minimizar el error cuadrático medio.
- **Autocorrección:** Si la predicción no es exacta, realiza ciclos adicionales de ajuste.
- **Visualización:** Muestra una gráfica del error durante el entrenamiento.

## Notas
- El modelo está entrenado para números entre 1 y 20. Para mejores resultados fuera de ese rango, modifica la sección de datos de entrenamiento.
- Este proyecto es solo para fines educativos.
