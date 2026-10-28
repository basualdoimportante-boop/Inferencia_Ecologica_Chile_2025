import arviz as az
import pandas as pd
import numpy as np
import os

base = r"C:\Users\Martín\Desktop\Inferencia_Ecologica_Chile_2025"

# Cargar la traza guardada
ruta_traza = os.path.join(base, "resultados", "traza_modelo_real.nc")
trace = az.from_netcdf(ruta_traza)

# Extraer las probabilidades de transición (theta)
# theta tiene forma (cadenas, iteraciones, comunas, origenes, destinos)
# Promediamos sobre las comunas para ver la transferencia NACIONAL
theta_mean = trace.posterior["theta"].mean(dim=["chain", "draw", "theta_dim_0"]).values

origenes = ['Jara', 'Kast', 'Parisi', 'Kaiser', 'Matthei', 'Otros']
destinos = ['Jara', 'Kast', 'Nulos/Blancos']

# Crear tabla
df_transferencia = pd.DataFrame(theta_mean, index=origenes, columns=destinos)
df_transferencia = (df_transferencia * 100).round(1)

print("=== MATRIZ DE TRANSFERENCIA NACIONAL (2025) ===")
print(df_transferencia.to_string())

# Guardar como CSV para el portafolio
df_transferencia.to_csv(os.path.join(base, "resultados", "matriz_transferencia.csv"))
