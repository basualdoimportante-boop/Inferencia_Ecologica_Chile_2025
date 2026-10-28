import pymc as pm
import numpy as np
import pandas as pd
import os
import arviz as az

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

    origenes = ['Jara', 'Kast', 'Parisi', 'Kaiser', 'Matthei', 'Otros', 'No_Voto_1V']
    destinos = ['Jara_2V', 'Kast_2V', 'No_Voto_2V']
    
    # Definimos las 3 configuraciones de priors
    escenarios = {
        "Debil (Sigma=0.1)": {"sigma_mu": 1.0, "sigma_beta": 0.1},
        "Concentrada (Sigma=0.01)": {"sigma_mu": 0.5, "sigma_beta": 0.01},
        "Difusa (Sigma=0.5)": {"sigma_mu": 2.0, "sigma_beta": 0.5}
    }

    resultados_medianas = {}

    for nombre_esc, params in escenarios.items():
        print(f"\n--- Entrenando modelo: {nombre_esc} ---")
        with pm.Model() as model:
            mu_alpha = pm.Normal("mu_alpha", mu=0, sigma=params["sigma_mu"], shape=(7, 3))
            beta_educ = pm.Normal("beta_educ", mu=0, sigma=params["sigma_beta"], shape=(7, 3))
            
            logits = mu_alpha[None, :, :] + beta_educ[None, :, :] * cov_educacion_std[:, None, None]
            theta = pm.Deterministic("theta", pm.math.softmax(logits, axis=2))
            
            p_destino = pm.Deterministic("p_destino", pm.math.sum(X_props[:, :, None] * theta, axis=1))
            
            alpha_dir = p_destino * 10 
            y_obs = pm.Dirichlet("y_obs", a=alpha_dir, observed=y_props)
            
            # Menos iteraciones para que no tarde horas
            trace = pm.sample(draws=1000, tune=1000, chains=2, cores=2, 
                              target_accept=0.95, return_inferencedata=True, progressbar=False)
            
        # Extraer matriz promedio nacional
        theta_samples = trace.posterior["theta"].values
        theta_flat = theta_samples.reshape(-1, theta_samples.shape[2], 7, 3).mean(axis=1)
        
        mediana_nacional = np.zeros((7, 3))
        for i in range(7):
            for j in range(3):
                mediana_nacional[i, j] = np.percentile(theta_flat[:, i, j], 50) * 100
                
        resultados_medianas[nombre_esc] = mediana_nacional

    # Imprimir comparativa
    print("\n=== COMPARACIÓN DE MEDIANAS ENTRE PRIORS ===")
    for i, orig in enumerate(origenes):
        for j, dest in enumerate(destinos):
            debil = resultados_medianas["Debil (Sigma=0.1)"][i, j]
            conc = resultados_medianas["Concentrada (Sigma=0.01)"][i, j]
            difusa = resultados_medianas["Difusa (Sigma=0.5)"][i, j]
            
            # Calcular la diferencia máxima
            max_diff = max(abs(debil - conc), abs(debil - difusa))
            
            alerta = " ⚠️" if max_diff > 5.0 else ""
            
            print(f"{orig[:6]:<6} -> {dest[:6]:<6} | Debil: {debil:>5.1f}% | Conc: {conc:>5.1f}% | Difusa: {difusa:>5.1f}% | Max Dif: {max_diff:>4.1f}%{alerta}")

    print("\nCriterio de aprobación: Ninguna celda debe cambiar más de 5.0 puntos porcentuales.")

if __name__ == '__main__':
    main()