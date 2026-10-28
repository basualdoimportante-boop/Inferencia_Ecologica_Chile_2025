

import pymc as pm
import numpy as np
import pandas as pd
import os
import arviz as az
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore") 

def main():
    np.random.seed(42)
    base = r"C:\Users\Martín\Desktop\Inferencia_Ecologica_Chile_2025"
    ruta_datos = os.path.join(base, "resultados", "dataset_real_completo.csv")

    print("Cargando dataset real...")
    df = pd.read_csv(ruta_datos).fillna(0)
    
    df['Total_2V'] = df['Kast'] + df['Jara'] + df['Nulos'] + df['Blancos']
    df['Total_Electores'] = df[['Total_Electores', 'Total_2V']].max(axis=1)
    df = df[df['Total_2V'] > 0].copy()
    
    comunas_excluidas = ['ANTARTICA', 'COLCHANE', 'GENERAL LAGOS']
    df = df[~df['comuna'].isin(comunas_excluidas)].copy()
    
    df['No_Voto_2V'] = df['Total_Electores'] - df['Jara'] - df['Kast']

    y_props = np.column_stack([
        df['Jara'] / df['Total_Electores'], 
        df['Kast'] / df['Total_Electores'], 
        df['No_Voto_2V'] / df['Total_Electores']
    ])

    df['Total_Validos_1V'] = df[['Jara_1V', 'Kast_1V', 'Parisi_1V', 'Kaiser_1V', 'Matthei_1V', 'Otros_1V']].sum(axis=1)
    df['No_Voto_1V'] = (df['Total_Electores'] - df['Total_Validos_1V']).clip(lower=0)
    
    X_origenes = df[['Jara_1V', 'Kast_1V', 'Parisi_1V', 'Kaiser_1V', 'Matthei_1V', 'Otros_1V', 'No_Voto_1V']].values
    X_props = X_origenes / df['Total_Electores'].values[:, None]
    
    cov_educacion_std = (df['pct_sup_adultos'].values - df['pct_sup_adultos'].mean()) / df['pct_sup_adultos'].std()

    def build_and_sample(usar_covariable):
        with pm.Model() as model:
            mu_alpha = pm.Normal("mu_alpha", mu=0, sigma=1, shape=(7, 3))
            
            if usar_covariable:
                beta_educ = pm.Normal("beta_educ", mu=0, sigma=0.1, shape=(7, 3))
                logits = mu_alpha[None, :, :] + beta_educ[None, :, :] * cov_educacion_std[:, None, None]
            else:
                logits = mu_alpha[None, :, :] + np.zeros((len(df), 7, 3))
                
            theta = pm.Deterministic("theta", pm.math.softmax(logits, axis=2))
            p_destino = pm.Deterministic("p_destino", pm.math.sum(X_props[:, :, None] * theta, axis=1))
            
            alpha_dir = p_destino * 10 
            y_obs = pm.Dirichlet("y_obs", a=alpha_dir, observed=y_props)
            
            trace = pm.sample(draws=1000, tune=1000, chains=2, cores=2, 
                              target_accept=0.9, return_inferencedata=True, progressbar=False)
            
            pm.compute_log_likelihood(trace, model=model)
            
        return trace, model

    print("\n--- Entrenando Modelo A: CON Covariable (Educación) ---")
    trace_con, model_con = build_and_sample(usar_covariable=True)
    
    print("--- Entrenando Modelo B: SIN Covariable (Solo Intercepto) ---")
    trace_sin, model_sin = build_and_sample(usar_covariable=False)

    # 1. Comparación de Modelos (LOO/WAIC) - CORREGIDO
    print("\n=== COMPARACIÓN DE AJUSTE (LOO) ===")
    comparison = az.compare({"Con_Educacion": trace_con, "Sin_Educacion": trace_sin})
    
    # Imprimir la tabla completa para evitar problemas de nombres de columnas
    print(comparison.to_string())
    
    mejor_modelo = comparison.index[0]
    print(f"\nEl modelo con mejor ajuste es: {mejor_modelo}")

    # 2. Extraer efecto de la covariable
    print("\n=== EFECTO DE LA EDUCACIÓN (Parisi -> Kast) ===")
    beta_parisi_kast = trace_con.posterior["beta_educ"].sel(beta_educ_dim_0=2, beta_educ_dim_1=1).values.flatten()
    mediana = np.percentile(beta_parisi_kast, 50)
    inf = np.percentile(beta_parisi_kast, 3)
    sup = np.percentile(beta_parisi_kast, 97)
    
    print(f"Coeficiente Beta: {mediana:.3f} [IC 94%: {inf:.3f}, {sup:.3f}]")
    if inf <= 0 <= sup:
        print("Conclusión: El intervalo cruza el cero. La educación NO tiene un efecto estadísticamente significativo en esta transferencia.")
    else:
        print("Conclusión: El intervalo NO cruza el cero. Hay una asociación contextual significativa.")

    # 3. Verificación Predictiva
    print("\nGenerando gráfico de verificación predictiva...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, dest in enumerate(['Jara_2V', 'Kast_2V', 'No_Voto_2V']):
        ppc = trace_con.posterior["p_destino"].values[..., i]
        ax = axes[i]
        ax.hist(ppc.mean(axis=(0,1)), bins=30, alpha=0.7, label="Estimación Modelo", color="blue")
        ax.hist(y_props[:, i], bins=30, alpha=0.7, label="Datos Reales", color="red")
        ax.set_title(f"Verificación Predictiva: {dest}")
        ax.legend()
    
    plt.tight_layout()
    ruta_grafico = os.path.join(base, "resultados", "verificacion_predictiva.png")
    plt.savefig(ruta_grafico)
    print(f"Gráfico guardado en: {ruta_grafico}")

if __name__ == '__main__':
    main()
    