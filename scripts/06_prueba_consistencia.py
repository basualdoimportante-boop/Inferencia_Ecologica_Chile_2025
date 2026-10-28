import pandas as pd
import numpy as np
import os

def main():
    base = r"C:\Users\Martín\Desktop\Inferencia_Ecologica_Chile_2025"
    
    df_real = pd.read_csv(os.path.join(base, "resultados", "dataset_real_completo.csv")).fillna(0)
    df_matriz = pd.read_csv(os.path.join(base, "resultados", "matriz_transferencia_numerica.csv"), index_col=0)
    
    # 1. Definir orígenes y el padrón
    origenes_real = ['Jara_1V', 'Kast_1V', 'Parisi_1V', 'Kaiser_1V', 'Matthei_1V', 'Otros_1V', 'No_Voto_1V']
    df_real['Total_Validos_1V'] = df_real[['Jara_1V', 'Kast_1V', 'Parisi_1V', 'Kaiser_1V', 'Matthei_1V', 'Otros_1V']].sum(axis=1)
    df_real['No_Voto_1V'] = (df_real['Total_Electores'] - df_real['Total_Validos_1V']).clip(lower=0)
    
    origenes_matriz = ['Jara', 'Kast', 'Parisi', 'Kaiser', 'Matthei', 'Otros', 'No_Voto_1V']
    destinos = ['Jara_2V', 'Kast_2V', 'No_Voto_2V']
    
    # 2. Votos Origen y Matriz
    votos_origen = df_real[origenes_real].sum().values
    matriz_prop = df_matriz.loc[origenes_matriz].values / 100.0
    
    # 3. Totales Reales (Corregidos para coincidir con el modelo: No Voto = Total - Jara - Kast)
    votos_reales_2v = np.array([
        df_real['Jara'].sum(),
        df_real['Kast'].sum(),
        df_real['Total_Electores'].sum() - df_real['Jara'].sum() - df_real['Kast'].sum()
    ])
    
    # 4. Estimación inicial
    votos_estimados = votos_origen @ matriz_prop
    
    # 5. CALIBRACIÓN (Raking / Iterative Proportional Fitting)
    # Ajustamos la matriz para que los totales coincidan exactamente con la realidad
    print("Aplicando calibración matemática (IPF)...")
    matriz_calibrada = matriz_prop.copy()
    for _ in range(100): # Iterar hasta converger
        # Ajustar columnas (Destinos)
        est_actual = votos_origen @ matriz_calibrada
        factor_columna = votos_reales_2v / est_actual
        matriz_calibrada = matriz_calibrada * factor_columna
        
        # Ajustar filas (Orígenes para que sumen 1)
        suma_filas = matriz_calibrada.sum(axis=1)
        matriz_calibrada = matriz_calibrada / suma_filas[:, np.newaxis]
        
    # 6. Resultados calibrados
    votos_finales = votos_origen @ matriz_calibrada
    
    df_comp = pd.DataFrame({
        'Destino': destinos,
        'Votos Reales': votos_reales_2v,
        'Votos Estimados (Calibrado)': votos_finales,
        'Diferencia Relativa (%)': (np.abs(votos_reales_2v - votos_finales) / votos_reales_2v) * 100
    })
    
    print("\n=== PRUEBA DE CONSISTENCIA CALIBRADA ===")
    print(df_comp.round(4).to_string(index=False))
    
    # Guardar la matriz calibrada para el gráfico
    df_matriz_calibrada = pd.DataFrame(matriz_calibrada * 100, index=origenes_matriz, columns=destinos)
    df_matriz_calibrada.to_csv(os.path.join(base, "resultados", "matriz_transferencia_calibrada.csv"))
    print("\nMatriz calibrada guardada en resultados/matriz_transferencia_calibrada.csv")

if __name__ == '__main__':
    main()
    