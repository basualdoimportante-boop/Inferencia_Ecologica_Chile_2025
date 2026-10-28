import pymc as pm
import numpy as np
import pandas as pd
import os
import arviz as az

def main():
    np.random.seed(42)
    base = r"C:\Users\Martín\Desktop\Inferencia_Ecologica_Chile_2025"
    ruta_datos = os.path.join(base, "resultados", "dataset_real_completo.csv")

    print("Cargando datos reales para usar como esqueleto...")
    df = pd.read_csv(ruta_datos).fillna(0)
    
    df['Total_2V'] = df['Kast'] + df['Jara'] + df['Nulos'] + df['Blancos']
    df['Total_Electores'] = df[['Total_Electores', 'Total_2V']].max(axis=1)
    df = df[df['Total_2V'] > 0].copy()
    
    # Excluimos los mismos outliers que en el modelo real
    comunas_excluidas = ['ANTARTICA', 'COLCHANE', 'GENERAL LAGOS']
    df = df[~df['comuna'].isin(comunas_excluidas)].copy()
    
    # 1. Definir la "Verdad Oculta" (La matriz que aprobaste)
    # Filas: Jara, Kast, Parisi, Kaiser, Matthei, Otros, No_Voto_1V
    # Columnas: Jara, Kast, No_Voto_2V
    true_matrix = np.array([
        [0.90, 0.05, 0.05], # Jara
        [0.05, 0.90, 0.05], # Kast
        [0.20, 0.60, 0.20], # Parisi
        [0.10, 0.80, 0.10], # Kaiser
        [0.15, 0.75, 0.10], # Matthei
        [0.40, 0.30, 0.30], # Otros
        [0.10, 0.20, 0.70]  # No_Voto_1V
    ])

    # 2. Preparar variables de origen
    df['Total_Validos_1V'] = df[['Jara_1V', 'Kast_1V', 'Parisi_1V', 'Kaiser_1V', 'Matthei_1V', 'Otros_1V']].sum(axis=1)
    df['No_Voto_1V'] = (df['Total_Electores'] - df['Total_Validos_1V']).clip(lower=0)
    
    X_origenes = df[['Jara_1V', 'Kast_1V', 'Parisi_1V', 'Kaiser_1V', 'Matthei_1V', 'Otros_1V', 'No_Voto_1V']].values
    X_props = X_origenes / df['Total_Electores'].values[:, None]
    
    # 3. Generar los votos sintéticos (Simulando la urna)
    print("Generando votos sintéticos a partir de la Verdad Oculta...")
    y_synthetic = np.zeros((len(df), 3))
    for i in range(len(df)):
        # Probabilidades reales para esta comuna
        p_real = X_props[i] @ true_matrix
        # Simulamos la urna con una distribución Multinomial
        votos_simulados = np.random.multinomial(df['Total_Electores'].iloc[i], p_real)
        y_synthetic[i] = votos_simulados

    # Convertir a proporciones para el modelo
    y_props_synthetic = y_synthetic / df['Total_Electores'].values[:, None]
    cov_educacion_std = (df['pct_sup_adultos'].values - df['pct_sup_adultos'].mean()) / df['pct_sup_adultos'].std()

    print("Entrenando modelo ciego para descubrir la verdad...")

    with pm.Model() as model_sim:
        mu_alpha = pm.Normal("mu_alpha", mu=0, sigma=1, shape=(7, 3))
        beta_educ = pm.Normal("beta_educ", mu=0, sigma=0.1, shape=(7, 3))
        
        logits = mu_alpha[None, :, :] + beta_educ[None, :, :] * cov_educacion_std[:, None, None]
        theta = pm.Deterministic("theta", pm.math.softmax(logits, axis=2))
        
        p_destino = pm.Deterministic("p_destino", pm.math.sum(X_props[:, :, None] * theta, axis=1))
        
        alpha_dir = p_destino * 10 
        y_obs = pm.Dirichlet("y_obs", a=alpha_dir, observed=y_props_synthetic)
        
        trace_sim = pm.sample(draws=1000, tune=1000, chains=2, cores=2, 
                              target_accept=0.99, return_inferencedata=True, progressbar=True)

    print("\n=== EVALUACIÓN DE RECUPERACIÓN DE LA VERDAD ===")
    
    # Extraer la matriz promedio estimada
    theta_samples = trace_sim.posterior["theta"].values
    theta_flat = theta_samples.reshape(-1, theta_samples.shape[2], 7, 3).mean(axis=1)
    
    origenes = ['Jara', 'Kast', 'Parisi', 'Kaiser', 'Matthei', 'Otros', 'No_Voto_1V']
    destinos = ['Jara_2V', 'Kast_2V', 'No_Voto_2V']
    
    print(f"{'Celda':<25} | {'Verdad':<7} | {'Mediana':<7} | {'IC 94%':<15} | {'¿Capturada?'}")
    print("-" * 75)
    
    celdas_capturadas = 0
    total_celdas = 21
    
    for i, orig in enumerate(origenes):
        for j, dest in enumerate(destinos):
            verdad = true_matrix[i, j] * 100
            muestras = theta_flat[:, i, j] * 100
            mediana = np.percentile(muestras, 50)
            inf = np.percentile(muestras, 3)
            sup = np.percentile(muestras, 97)
            
            # Verificar si la verdad está dentro del intervalo
            capturada = inf <= verdad <= sup
            if capturada:
                celdas_capturadas += 1
                
            estado = "✅ Sí" if capturada else "❌ No"
            print(f"{orig[:6]} -> {dest:<10} | {verdad:>5.1f}% | {mediana:>5.1f}% | [{inf:>4.1f}, {sup:>4.1f}] | {estado}")

    cobertura = (celdas_capturadas / total_celdas) * 100
    print("-" * 75)
    print(f"COBERTURA TOTAL: {cobertura:.1f}% (Criterio de aprobación: >= 90%)")
    
    if cobertura >= 90:
        print("\n✅ SIMULACIÓN APROBADA: El modelo recupera exitosamente la verdad oculta.")
    else:
        print("\n❌ SIMULACIÓN FALLIDA: El modelo no logró capturar la verdad en suficientes celdas.")

if __name__ == '__main__':
    main()