# Ejemplo: Distribución de probabilidad para el clima
distribucion_clima = {
    "Soleado": 0.6,
    "Nublado": 0.3,
    "Lluvioso": 0.1
}

# Verificamos que las probabilidades sumen 1 (normalización)
suma_prob = sum(distribucion_clima.values())

# Normalizar si es necesario
if abs(suma_prob - 1.0) > 1e-6:
    distribucion_clima = {k: v / suma_prob for k, v in distribucion_clima.items()}

# Mostrar resultados
print("Distribución de probabilidad del clima:")
for estado, prob in distribucion_clima.items():
    print(f"P({estado}) = {prob:.2f}")

print(f"Suma total de probabilidades = {sum(distribucion_clima.values()):.2f}")
