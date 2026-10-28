import pandas as pd
import os
base = r"C:\Users\Martín\Desktop\Inferencia_Ecologica_Chile_2025"
df = pd.read_csv(os.path.join(base, "resultados", "matriz_transferencia_numerica.csv"), index_col=0)
print((df).round(1).to_string())