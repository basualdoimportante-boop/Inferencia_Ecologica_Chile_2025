# Inferencia Ecológica y Transferencia Electoral (Chile 2025)

## 📌 Resumen Ejecutivo
Este proyecto aplica **inferencia ecológica bayesiana jerárquica** (modelo multinomial-Dirichlet) para estimar la matriz de transferencia de votos entre la primera y la segunda vuelta presidencial de Chile en 2025, a nivel comunal.

**Hallazgos principales**
- Las filas de Jara y Kast son las mejor identificadas, pero no están libres de incertidumbre: sus celdas de voto duro variaron hasta 14 puntos porcentuales al cambiar las distribuciones previas (priors).
- La fila de Parisi tiene identificación intermedia y es sensible a cómo se definen las categorías de destino (ver Limitaciones, punto 2).
- Las filas de Kaiser, Matthei y "Otros" tienen intervalos muy amplios y dependen en gran medida de las distribuciones previas; sus medias no deben interpretarse como estimaciones precisas.
- Educación superior (Censo 2024), incluida como covariable, no mejora la capacidad predictiva del modelo. Con esta precisión no hay evidencia de un efecto contextual distinguible de cero, lo cual no equivale a demostrar que no exista.

**Advertencia metodológica:** todos los resultados son estimaciones probabilísticas, no conteos observados. El modelo estima tasas agregadas por comuna y no infiere comportamiento individual (falacia ecológica).

## 📊 Resultados (matriz de transferencia)
- Modelo: 7 orígenes × 4 destinos (28 celdas, 21 parámetros libres porque cada fila suma 100%).
- Unidades: 341 comunas (se excluyeron Antártica, Colchane y General Lagos por dinámicas poblacionales extremas).
- Estimación: 2 cadenas, 2000 iteraciones de ajuste y 2000 de muestreo.
- Las celdas muestran la **media posterior** e intervalo de credibilidad del 94%.
- Estado de las cifras: Las medias reportadas son **antes del raking**. El raking se aplicó únicamente para la visualización final del mapa de calor; los intervalos corresponden al modelo crudo.

| Origen (primera vuelta) | Jara | Kast | Nulos/Blancos | Abstención |
| :--- | :--- | :--- | :--- | :--- |
| **Jara** | 87.9% [74.5, 96.0] | 3.9% [0.6, 14.4] | 4.4% [0.8, 13.1] | 2.3% [0.5, 7.1] |
| **Kast** | 3.2% [0.5, 12.2] | 90.6% [79.3, 96.9] | 2.8% [0.5, 8.7] | 2.1% [0.4, 6.4] |
| **Parisi** | 26.1% [10.8, 40.3] | 49.2% [33.8, 65.8] | 20.1% [10.9, 28.5] | 4.1% [1.0, 10.3] |
| **Kaiser** | 11.5% [1.6, 40.4] | 72.3% [40.0, 91.7] | 8.5% [1.3, 24.5] | 4.7% [0.9, 15.1] |
| **Matthei** | 20.2% [2.9, 53.8] | 58.1% [23.9, 84.9] | 11.5% [2.1, 29.2] | 6.8% [1.3, 20.7] |
| **Otros** | 29.5% [3.2, 78.0] | 20.1% [2.7, 68.1] | 21.2% [2.7, 72.8] | 12.0% [1.8, 45.6] |
| **No votó** | 9.0% [2.1, 20.0] | 10.5% [2.5, 21.7] | 5.3% [1.5, 11.2] | 74.1% [63.8, 83.1] |

### Calidad de identificación por fila
Se clasifica con dos criterios: ancho del intervalo y estabilidad ante cambios de prior.

| Fila | Ancho del intervalo | Rango entre priors (3 configuraciones) | Clasificación |
| :--- | :--- | :--- | :--- |
| Jara | ~21.5% (74.5 - 96.0) | ~13.4% (Jara->Jara) | Mejor identificada |
| Kast | ~17.6% (79.3 - 96.9) | ~7.3% (Kast->Kast) | Mejor identificada |
| Parisi | ~29.5% (10.8 - 40.3) | ~4.0% (Parisi->Kast) | Intermedia, sensible a la especificación |
| Kaiser | ~38.8% (1.6 - 40.4) | ~11.6% (Kaiser->Kast) | Poco identificada |
| Matthei | ~50.9% (2.9 - 53.8) | ~9.3% (Matthei->Kast) | Poco identificada |
| Otros | ~74.8% (3.2 - 78.0) | ~12.7% (Otros->Kast) | Poco identificada |

## 🔍 Diagnósticos de convergencia
- R-hat máximo: 1.0039
- Tamaño efectivo de muestra mínimo: 1931
- Divergencias: 0
- Limitación: se usaron solo 2 cadenas por restricciones de cómputo local. Sin embargo, el ESS (>400) y el R-hat sugieren convergencia adecuada.
- Chequeo predictivo posterior: Se generó un gráfico comparativo (ver `resultados/verificacion_predictiva.png`) que muestra que las predicciones del modelo siguen la distribución de los datos reales.

## 🔬 Pruebas de validación y limitaciones
1. **Calibración (raking), no validación.** El modelo crudo tuvo un error de consistencia agregada de 4.9% (Jara) y 2.1% (Kast). Un error de esa magnitud sugiere especificación imperfecta. El raking iterativo fuerza que la matriz final reproduzca los totales nacionales; es una técnica de calibración, no una validación empírica. Los intervalos corresponden al modelo antes del raking. Los totales nacionales usados para calibrar incluyen las 3 comunas excluidas.

2. **Sensibilidad a la especificación y a los priors.**
   - Con tres configuraciones de prior, las celdas de voto duro variaron hasta 14 puntos porcentuales, magnitud comparable al ancho de los intervalos. Por eso los intervalos de una sola configuración probablemente subestiman la incertidumbre.
   - Al separar nulos/blancos de abstención, la transferencia Parisi → Kast cambió de 59.7% a 49.2%. En la versión de tres destinos, el destino "no vota" de Parisi era 12.9%; en la de cuatro, nulos/blancos más abstención suman 24.2%. Separar una categoría no debería cambiar los totales agregados si los priors son coherentes; esto indica que esa fila depende de la especificación.

3. **Cotas de Duncan y Davis.** Se verificó que las estimaciones no violen las cotas lógicas (0 violaciones). Esta prueba tiene poco poder por sí sola, porque el modelo ya opera dentro de lo factible. El ancho de las cotas indica que en comunas pequeñas la identificación proviene más del prior que de los datos.

4. **Simulación sintética (caso ideal).** Se generaron datos con una matriz verdadera fija y ruido multinomial; el modelo recuperó el 100% de las celdas dentro de sus intervalos. Como todas las comunas comparten la misma transferencia (sin heterogeneidad ni sesgo de agregación) y los intervalos son amplios, esta prueba demuestra que el código funciona, no que el método resista datos reales. Queda como trabajo futuro simular escenarios con transferencias dependientes de la composición.

5. **Efecto de la educación.** Entra como efecto lineal en el logit. La comparación por validación cruzada dejando uno fuera (LOO) no favorece al modelo con educación (diferencia menor a 1), y el coeficiente de Parisi → Kast tiene un intervalo que incluye el cero ([-0.206, 0.140]). Limitaciones: se reporta un solo coeficiente (el de mayor interés sustantivo), y la validación cruzada dejando una comuna fuera es optimista con dependencia espacial entre comunas vecinas.

6. **Unidad de análisis.** El modelo opera a nivel comunal (341 unidades), lo que limita la identificación. La mejora prevista es migrar a mesa electoral (cerca de 40,000 unidades), lo que reduciría la dependencia de los priors. Se requiere verificar que las mesas coincidan entre ambas vueltas.

## 📚 Fuentes de datos
- Resultados electorales (SERVEL): Primera y Segunda vuelta presidencial 2025, descargados desde el portal de datos abiertos de SERVEL en Octubre de 2026.
- Censo 2024 (INE): Microdatos procesados con DuckDB. Variable usada: `cine11` (nivel educacional) para calcular la proporción de población con educación superior por comuna. Descargado en Octubre de 2026.
- Los datos crudos no se versionan en el repositorio por su tamaño (Censo supera los 2 GB) y para respetar las políticas de distribución de SERVEL e INE.

## 🛠️ Stack y reproducibilidad
- Python 3.11 o superior (PyMC 5, ArviZ, DuckDB, pandas, seaborn).
- Semilla aleatoria: `np.random.seed(42)`.

**Pipeline de ejecución**
1. `python scripts/01_preparar_datos.py` (Censo y SERVEL)
2. `python scripts/04_modelo_real.py` (modelo bayesiano 7×4)
3. `python scripts/06_prueba_consistencia.py` (calibración)
4. `python scripts/07_prueba_cotas.py` (cotas)
5. `python scripts/08_simulacion_final.py` (simulación sintética)
6. `python scripts/10_efecto_covariable.py` (validación cruzada y comparación con y sin educación)

*(Nota: Los scripts 02, 03, 05 y 09 fueron exploratorios y sus funciones se consolidaron en los scripts principales).*

## ⚠️ Ética
El modelo estima tasas agregadas. No debe usarse para perfilar individuos ni para microtargeting. Los resultados son estimaciones con incertidumbre y no deben citarse como conteos exactos.