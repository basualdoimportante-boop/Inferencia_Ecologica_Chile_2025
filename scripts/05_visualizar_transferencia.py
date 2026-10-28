import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    base = r"C:\Users\Martín\Desktop\Inferencia_Ecologica_Chile_2025"
    ruta_csv = os.path.join(base, "resultados", "matriz_transferencia_numerica.csv")
    ruta_imagen = os.path.join(base, "resultados", "heatmap_transferencia.png")

    print("Cargando matriz de transferencia...")
    df = pd.read_csv(ruta_csv, index_col=0)
    
    # 1. Renombrar las filas (Orígenes)
    df.index = [
        'Jara (1V)', 'Kast (1V)', 'Parisi (1V)', 
        'Kaiser (1V)', 'Matthei (1V)', 'Otros (1V)', 
        'No Votó (1V)'
    ]
    
    # 2. Renombrar las columnas (Destinos) - AHORA SON 4
    df.columns = [
        'Jara (2V)', 'Kast (2V)', 'Nulos/Blancos', 'Abstención'
    ]

    print("Generando Heatmap de 4 columnas...")
    plt.figure(figsize=(10, 6))
    
    # Paleta de colores
    sns.heatmap(df, annot=True, fmt=".1f", cmap="Reds", 
                cbar_kws={'label': 'Porcentaje de Transferencia (%)'},
                linewidths=.5, vmin=0, vmax=100)
    
    plt.title('Matriz de Transferencia Electoral (Chile 2025)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Destino del Voto (Segunda Vuelta)', fontsize=12, fontweight='bold')
    plt.ylabel('Origen del Voto (Primera Vuelta)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(ruta_imagen, dpi=300, bbox_inches='tight')
    
    print(f"¡Éxito! Gráfico guardado en: {ruta_imagen}")
    plt.show()

if __name__ == '__main__':
    main()