#Épocas: cantidad de veces que el modelo repasa todos los datos para aprender.
#Error cuadrático medio (ECM): qué tan lejos está la predicción del resultado correcto
#Peso: qué tanto influye la entrada (x) en la salida.
#Sesgo (bias): valor constante que ayuda a ajustar la salida (como el punto de partida).


import numpy as np
import matplotlib.pyplot as plt
import json



# 1. Ingreso del número por parte del usuario
try:
    nuevo_x = float(input("🔢 Ingresa un número para predecir su doble: "))
except ValueError:
    print(" Entrada inválida. Intenta con un número.")
    exit()

# 2. Datos de entrenamiento: números del 1 al 20 y su doble como salida
x = np.array([[i] for i in range(1, 21)], dtype=float)        # Entradas
y = np.array([[2 * i] for i in range(1, 21)], dtype=float)    # Salidas esperadas (el doble)

# 3. Inicialización de parámetros
np.random.seed(42)                     
peso = np.random.normal(0, 0.1)        # Peso inicial aleatorio
sesgo = np.random.normal(0, 0.1)       # Sesgo inicial aleatorio
tasa_aprendizaje = 0.001               # Velocidad de aprendizaje
errores = []                           # Lista para almacenar errores por época

# 4. Entrenamiento del modelo por un número fijo de épocas
max_epochs = 1000                      # Número máximo de ciclos de entrenamiento
for epoch in range(max_epochs):
    y_pred = x * peso + sesgo                         # Predicción del modelo
    error = np.mean((y - y_pred) ** 2)                # Error cuadrático medio
    errores.append(error)                             # Guardar el error de esta época

    # Derivadas para actualizar peso y sesgo
    d_peso = -2 * np.mean(x * (y - y_pred))           # Derivada con respecto al peso
    d_sesgo = -2 * np.mean(y - y_pred)                # Derivada con respecto al sesgo

    # Actualización de parámetros
    peso -= tasa_aprendizaje * d_peso
    sesgo -= tasa_aprendizaje * d_sesgo




# 5. Predicción inicial con los pesos aprendidos
prediccion = nuevo_x * peso + sesgo
esperado = 2 * nuevo_x                                # Valor correcto
print(f"\n Entrenamiento completado en {max_epochs} épocas")
print(f" Peso aprendido: {peso:.4f}")
print(f" Sesgo aprendido: {sesgo:.4f}")
print(f"\n Resultado esperado: {esperado:.2f}")
print(f" Predicción inicial del modelo: {float(prediccion):.2f}")

# 6. Autocorrección: mejora la predicción si no es exacta
limite_error = 0.00          # Margen de error permitido (exactitud deseada)
auto_epochs = 0               # Contador de ciclos de corrección
max_auto = 10000              # Límite de ciclos de autocorrección para evitar bucles infinitos

while abs(prediccion - esperado) > limite_error and auto_epochs < max_auto:
    y_pred = x * peso + sesgo                     # Recalcula la predicción
    d_peso = -2 * np.mean(x * (y - y_pred))       # Nuevas derivadas
    d_sesgo = -2 * np.mean(y - y_pred)

    peso -= tasa_aprendizaje * d_peso             # Ajuste del peso
    sesgo -= tasa_aprendizaje * d_sesgo           # Ajuste del sesgo

    prediccion = nuevo_x * peso + sesgo           # Recalcula la predicción con nuevos parámetros
    auto_epochs += 1                              # Aumenta el contador

# Mostrar resultado final después de la autocorrección
if auto_epochs > 0:
    print(f"\n Autocorrección realizada en {auto_epochs} ciclos")
    print(f" Nueva predicción: {float(prediccion):.2f}")
else:
    print("\n No fue necesario corregir, el modelo ya estaba bien!")

# 7. Gráfica del error (últimas 150 épocas para ver la curva de aprendizaje final)

# Mostrar los últimos 50 errores (o todos si hay menos)
ultimos_errores = errores[-1000:] if len(errores) >= 1000 else errores

# Ajustar el eje Y con margen correcto
plt.figure(figsize=(8, 4))
plt.plot(range(len(errores)), errores, color='blue')
plt.ylim(min(errores) * 0.95, max(errores) * 1.05)
plt.title("Curva de Error del Entrenamiento (Todas las épocas)")
plt.xlabel("Épocas")
plt.ylabel("Error cuadrático medio")
plt.grid(True)
plt.tight_layout()
plt.show()

