import pandas as pd
import numpy as np
import os

def main():
    base = r"C:\Users\Martín\Desktop\Inferencia_Ecologica_Chile_2025"
    
    df_real = pd.read_csv(os.path.join(base, "resultados", "dataset_real_completo.csv")).fillna(0)
    
    # Cargar la matriz 3D específica de cada comuna
    datos_npz = np.load(os.path.join(base, "resultados", "matriz_por_comuna.npz"), allow_pickle=True)
    M_comunas = datos_npz['matriz'] # (n_comunas, 7, 3)
    
    df_real['Total_Validos_1V'] = df_real[['Jara_1V', 'Kast_1V', 'Parisi_1V', 'Kaiser_1V', 'Matthei_1V', 'Otros_1V']].sum(axis=1)
    df_real['No_Voto_1V'] = (df_real['Total_Electores'] - df_real['Total_Validos_1V']).clip(lower=0)
    
    origenes = ['Jara', 'Kast', 'Parisi', 'Kaiser', 'Matthei', 'Otros', 'No_Voto_1V']
    origenes_real = ['Jara_1V', 'Kast_1V', 'Parisi_1V', 'Kaiser_1V', 'Matthei_1V', 'Otros_1V', 'No_Voto_1V']
    
    # Filtramos df_real para que coincida exactamente con el orden del modelo
    df_real = df_real[df_real['Total_2V'] > 0].copy() if 'Total_2V' in df_real.columns else df_real
    # Usamos las mismas comunas que están en el NPZ
    comunas_modelo = datos_npz['comunas']
    df_real = df_real.set_index('comuna').loc[comunas_modelo].reset_index()
    
    violaciones = []
    V_dest = df_real[['Jara', 'Kast']].values
    V_total = df_real['Total_Electores'].values
    
    for i in range(len(df_real)):
        V_orig = df_real[origenes_real].values[i]
        
        for r_idx, r_name in enumerate(origenes):
            if V_orig[r_idx] <= 0: continue
            
            for c_idx, c_name in enumerate(['Jara', 'Kast']):
                V_c = V_dest[i, c_idx]
                
                lower_bound = max(0, (V_c - (V_total[i] - V_orig[r_idx])) / V_orig[r_idx])
                upper_bound = min(1, V_c / V_orig[r_idx])
                
                # Estimación específica de ESTA comuna
                estimacion = M_comunas[i, r_idx, c_idx]
                
                if estimacion < lower_bound - 1e-5 or estimacion > upper_bound + 1e-5:
                    violaciones.append({
                        'Comuna': df_real.iloc[i]['comuna'],
                        'Origen': r_name,
                        'Destino': c_name,
                        'Estimacion_Modelo': estimacion,
                        'Cota_Inferior': lower_bound,
                        'Cota_Superior': upper_bound
                    })
                    
    print("=== PRUEBA DE COTAS DUNCAN & DAVIS (POR COMUNA) ===")
    if len(violaciones) == 0:
        print("✅ APROBADO: Ninguna estimación viola las cotas lógicas deterministas.")
    else:
        print(f"❌ FALLIDO: Se encontraron {len(violaciones)} violaciones.")
        df_viol = pd.DataFrame(violaciones)
        print(df_viol.head(10).to_string(index=False))

if __name__ == '__main__':
    main()
    