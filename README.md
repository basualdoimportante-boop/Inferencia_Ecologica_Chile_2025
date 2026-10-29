# Inferencia Ecológica y Transferencia Electoral (Chile 2025)

## 📌 Resumen Ejecutivo
Este proyecto aplica **inferencia ecológica bayesiana jerárquica** (modelo multinomial-Dirichlet) para estimar la matriz de transferencia de votos entre la primera y la segunda vuelta presidencial de Chile en 2025, a nivel comunal.

**Hallazgos principales**
- Las filas de Jara y Kast son las mejor identificadas, pero no están libres de incertidumbre: sus celdas de voto duro variaron hasta 14 puntos porcentuales al cambiar las distribuciones previas (*priors*).
- La fila de Parisi tiene identificación intermedia. Al separar nulos/blancos de abstención (de 3 a 4 destinos), la transferencia de Parisi a Kast cayó de 59.7% a 49.3%. Este cambio de 10 puntos indica que la estimación es sensible a la especificación del modelo, un hallazgo clave del proyecto.
- Las filas de Kaiser, Matthei y "Otros" tienen intervalos muy amplios. Sus medias no deben interpretarse como estimaciones precisas.
- La educación superior (Censo 2024) no mejora la capacidad predictiva del modelo. Con esta precisión no hay evidencia de un efecto distinguible de cero.

**Advertencia sobre la Falacia Ecológica:** El modelo estima **probabilidades de transición individual** (ej. "el 49.3% de quienes votaron por Parisi pasaron a Kast"). Estas estimaciones solo son válidas bajo el supuesto de homogeneidad (que las comunas se comportan igual dada su demografía), el cual no se puede verificar con datos agregados.

## 📊 Resultados (matriz de transferencia)
- Modelo: 7 orígenes × 4 destinos (21 parámetros libres porque cada fila suma 100%).
- Unidades: 341 comunas (se excluyeron Antártica, Colchane y General Lagos).
- Estimación: 2 cadenas, 2000 iteraciones de ajuste y 2000 de muestreo. `random_seed=42`.
- Las celdas muestran la **media posterior** e intervalo de credibilidad del 94%. (Las filas suman 100% al usar medias).

| Origen (1V) | Jara (2V) | Kast (2V) | Nulos/Blancos | Abstención |
| :--- | :--- | :--- | :--- | :--- |
| **Jara** | 87.4% [74.5, 96.0] | 4.8% [0.6, 14.4] | 5.1% [0.8, 13.1] | 2.8% [0.5, 7.1] |
| **Kast** | 4.2% [0.5, 12.2] | 90.0% [79.3, 96.9] | 3.3% [0.5, 8.7] | 2.5% [0.4, 6.4] |
| **Parisi** | 25.9% [10.8, 40.3] | 49.3% [33.8, 65.8] | 20.2% [10.9, 28.5] | 4.6% [1.0, 10.3] |
| **Kaiser** | 13.8% [1.6, 40.4] | 70.9% [40.0, 91.7] | 9.7% [1.3, 24.5] | 5.6% [0.9, 15.1] |
| **Matthei** | 23.0% [2.9, 53.8] | 56.3% [23.9, 84.9] | 12.8% [2.1, 29.2] | 7.9% [1.3, 20.7] |
| **Otros** | 32.9% [3.2, 78.0] | 24.9% [2.7, 68.1] | 26.9% [2.7, 72.8] | 15.3% [1.8, 45.6] |
| **No votó 1V** | 9.6% [2.1, 20.0] | 11.0% [2.5, 21.7] | 5.5% [1.5, 11.2] | 73.9% [63.8, 83.1] |

## 🔬 Pruebas de Validación y Limitaciones
1. **Calibración (raking), no validación:** El modelo crudo tuvo un error de 4.9% (Jara) y 2.1% (Kast). El raking se usó para cuadrar los totales nacionales. Los intervalos corresponden al modelo crudo. Los totales usados incluyen las 3 comunas excluidas.
2. **Sensibilidad a la especificación:** Como se mencionó, separar Nulos/Blancos de la Abstención movió la estimación de Parisi->Kast en 10 puntos. Esto demuestra que el modelo es sensible a cómo se agrupan las categorías de destino.
3. **Sesgo de agregación no testeado:** La simulación sintética recuperó la verdad usando una matriz idéntica para todas las comunas. No se probó un escenario donde la transferencia dependa de la composición comunal, lo cual queda como trabajo futuro.
4. **Efecto de la Educación:** La comparación LOO no favorece al modelo con educación. El coeficiente Parisi->Kast cruza el cero. No hay evidencia de efecto contextual. Se usó el Censo 2024 (residentes) como proxy del electorado.
5. **Dependencia espacial:** No se modeló la autocorrelación espacial (comunas vecinas). La validación cruzada dejando una comuna fuera es optimista.

## 📚 Fuentes y Reproducibilidad
- Resultados SERVEL (1V y 2V 2025) y Censo 2024 (INE). Descargados en Octubre 2026. Datos crudos excluidos por tamaño.
- `requirements.txt` incluye versiones fijadas. Pipeline: `scripts/01` a `scripts/10`.
- `random_seed=42` fijado tanto en Numpy como en PyMC.

## ⚠️ Ética
El modelo estima tasas agregadas bajo supuestos. No debe usarse para perfilar individuos ni para microtargeting. Los resultados son estimaciones con incertidumbre.