# Viabilidad de SARSA con la base actual

## Veredicto

Sí se puede aplicar SARSA, pero como política de parada aprendida sobre reproducciones de las capturas IQ. No sería un agente que controla el enlace ni una política óptima: las adquisiciones son archivos finitos y no se pueden solicitar nuevas mediciones al equipo durante cada episodio.

La decisión por episodio es: observar el siguiente bloque, declarar traslape o declarar ausencia de traslape. El estado debe contener la creencia posterior sobre las condiciones físicas y los bloques restantes. La probabilidad agregada de traslape por sí sola no siempre es suficiente: dos creencias con el mismo valor agregado pueden tener distinta incertidumbre sobre OSNR y separación, y por tanto distinta evidencia esperada del próximo bloque.

Una formulación viable es discretizar las probabilidades posteriores de condición (o un vector reducido de características de la creencia) y el presupuesto de bloques. SARSA actualiza `Q(s,a)` a partir de episodios reproducidos. El reward es `-c_bloque` al observar y, al parar, `-C_FP` por falsa alarma o `-C_FN` por omitir traslape. La etiqueta real solo calcula el reward terminal y nunca forma parte del estado o de la selección de acción.

## Cómo evaluar sin fuga de datos

- Cada episodio usa una adquisición completa y bloques IQ consecutivos.
- Separar por `song`: `song1` y `song2` para entrenar KDE y SARSA, `song3` para ajustar hiperparámetros y `song4` para prueba final.
- Congelar la política antes de evaluar `song4`; reportar semillas, coste total, error por clase, bloques usados y comparación con ventana fija y umbral de confianza.
- Repetir con varios costes de observación y costes de error. Las capturas están disponibles offline, por lo que la interpretación es una política de parada simulada.

## Alcance y limitaciones

SARSA tiene sentido si la pregunta central es aprender cuándo parar bajo un coste de observación. Si solo se quiere comparar una regla de parada conocida, la actualización bayesiana más una política por umbral es más sencilla, interpretable y probablemente más estable con cuatro adquisiciones (`song1`–`song4`) por condición. La discretización de la creencia puede perder información; reportar sensibilidad a sus bins y comparar con el umbral es obligatorio.

La configuración activa apunta al escenario `32G_B2B`, con `Rs=32 GBd`, `roll-off=0.1` y frontera geométrica `35.2 GHz`. La base también contiene 16 GBd y enlaces de 270 km / 9 dBm; no se deben mezclar con el escenario B2B principal. La etiqueta es geométrica y no demuestra por sí sola degradación de BER.

## Diferencias frente a la propuesta PDF

La propuesta anterior describe 32 GBd y 35.2 GHz, pero sus conteos de adquisiciones y lista de OSNR no corresponden al árbol nuevo. La base nueva tiene archivos Parquet con `rx_i`, `rx_q` y `tx_sym`, separados por escenario, separación y archivos `songN`. El split por canción aprovecha esas réplicas sin repartir símbolos de una misma adquisición entre entrenamiento y prueba.
