import pymc as pm
import numpy as np
import pandas as pd
import os
import arviz as az

def main():
    # Semilla para reproducibilidad en Numpy y PyMC
    np.random.seed(42)
    
    base = r"C:\Users\Martín\Desktop\Inferencia_Ecologica_Chile_2025"
    ruta_datos = os.path.join(base, "resultados", "dataset_real_completo.csv")

    print("Cargando dataset real...")
    df = pd.read_csv(ruta_datos).fillna(0)
    
    # Calcular totales y ajustar electores
    df['Total_2V'] = df['Kast'] + df['Jara'] + df['Nulos'] + df['Blancos']
    df['Total_Electores'] = df[['Total_Electores', 'Total_2V']].max(axis=1)
    df = df[df['Total_2V'] > 0].copy()
    
    # Exclusión de outliers geográficos (3 comunas extremas)
    comunas_excluidas = ['ANTARTICA', 'COLCHANE', 'GENERAL LAGOS']
    df = df[~df['comuna'].isin(comunas_excluidas)].copy()
    print(f"Comunas excluidas: {comunas_excluidas}. Comunas a modelar: {len(df)}")
    
    # Separar Nulos/Blancos de Abstención (4 destinos)
    df['Nulos_Blancos_2V'] = df['Nulos'] + df['Blancos']
    df['Abstencion_2V'] = df['Total_Electores'] - df['Jara'] - df['Kast'] - df['Nulos_Blancos_2V']

    # Proporciones de destino (4 columnas)
    y_props = np.column_stack([
        df['Jara'] / df['Total_Electores'], 
        df['Kast'] / df['Total_Electores'], 
        df['Nulos_Blancos_2V'] / df['Total_Electores'],
        df['Abstencion_2V'] / df['Total_Electores']
    ])

    # Proporciones de origen (7 columnas)
    df['Total_Validos_1V'] = df[['Jara_1V', 'Kast_1V', 'Parisi_1V', 'Kaiser_1V', 'Matthei_1V', 'Otros_1V']].sum(axis=1)
    df['No_Voto_1V'] = (df['Total_Electores'] - df['Total_Validos_1V']).clip(lower=0)
    
    X_origenes = df[['Jara_1V', 'Kast_1V', 'Parisi_1V', 'Kaiser_1V', 'Matthei_1V', 'Otros_1V', 'No_Voto_1V']].values
    X_props = X_origenes / df['Total_Electores'].values[:, None]
    
    # Covariable (Educación)
    cov_educacion_std = (df['pct_sup_adultos'].values - df['pct_sup_adultos'].mean()) / df['pct_sup_adultos'].std()

    print("Iniciando modelo bayesiano jerárquico (Matriz 7x4)...")

    with pm.Model() as model_real:
        # Priors
        mu_alpha = pm.Normal("mu_alpha", mu=0, sigma=1, shape=(7, 4))
        beta_educ = pm.Normal("beta_educ", mu=0, sigma=0.1, shape=(7, 4))
        
        # Probabilidades de transición
        logits = mu_alpha[None, :, :] + beta_educ[None, :, :] * cov_educacion_std[:, None, None]
        theta = pm.Deterministic("theta", pm.math.softmax(logits, axis=2))
        
        # Probabilidad agregada esperada
        p_destino = pm.Deterministic("p_destino", pm.math.sum(X_props[:, :, None] * theta, axis=1))
        
        # Verosimilitud (Dirichlet)
        alpha_dir = p_destino * 10 
        y_obs = pm.Dirichlet("y_obs", a=alpha_dir, observed=y_props)
        
        # Muestreo (con semilla fija)
        trace_real = pm.sample(draws=2000, tune=2000, chains=2, cores=2, 
                               target_accept=0.99, random_seed=42, 
                               return_inferencedata=True, progressbar=True)

    print("\n=== DIAGNÓSTICOS DE CONVERGENCIA ===")
    summary = az.summary(trace_real, var_names=["mu_alpha", "beta_educ"])
    print(f"R-hat Máximo: {summary['r_hat'].max():.4f}")
    print(f"ESS Mínimo: {summary['ess_bulk'].min():.0f}")

    # Extraer muestras de theta
    theta_samples = trace_real.posterior["theta"].values
    # Forma: (cadenas, iteraciones, comunas, 7, 4) -> Promediamos comunas para la matriz nacional
    theta_flat = theta_samples.reshape(-1, theta_samples.shape[2], 7, 4).mean(axis=1)
    
    origenes = ['Jara', 'Kast', 'Parisi', 'Kaiser', 'Matthei', 'Otros', 'No_Voto_1V']
    destinos = ['Jara_2V', 'Kast_2V', 'Nulos_Blancos_2V', 'Abstencion_2V']
    
    # 1. Guardar matriz completa con intervalos (Usando MEDIA para que sumen 100%)
    resultados = []
    matrix_medias = np.zeros((7, 4))
    
    for i, orig in enumerate(origenes):
        fila = {'Origen': orig}
        for j, dest in enumerate(destinos):
            media = np.mean(theta_flat[:, i, j]) * 100
            inf = np.percentile(theta_flat[:, i, j], 3) * 100
            sup = np.percentile(theta_flat[:, i, j], 97) * 100
            fila[dest] = f"{media:.1f}% [{inf:.1f}, {sup:.1f}]"
            matrix_medias[i, j] = media
        resultados.append(fila)
        
    df_res = pd.DataFrame(resultados)
    df_res.to_csv(os.path.join(base, "resultados", "matriz_transferencia_intervalos.csv"), index=False)
    
    # 2. Guardar matriz numérica pura (Para el Heatmap)
    df_num = pd.DataFrame(matrix_medias, index=origenes, columns=destinos)
    df_num.to_csv(os.path.join(base, "resultados", "matriz_transferencia_numerica.csv"))
    
    # 3. Guardar matriz 3D por comuna (Para las pruebas de cotas)
    matriz_comunas = theta_samples.mean(axis=(0, 1))
    np.savez(os.path.join(base, "resultados", "matriz_por_comuna.npz"), 
             matriz=matriz_comunas, 
             comunas=df['comuna'].values)
    
    print("\n=== MATRIZ COMPLETA (Media e IC 94%) ===")
    print(df_res.to_string(index=False))
    
    # Verificación de que las filas suman 100%
    print("\n=== VERIFICACIÓN DE SUMA DE FILAS ===")
    for i, orig in enumerate(origenes):
        suma = matrix_medias[i].sum()
        print(f"Fila {orig}: {suma:.2f}%")
        
    print("\nLos archivos han sido guardados en la carpeta 'resultados'.")

if __name__ == '__main__':
    main()