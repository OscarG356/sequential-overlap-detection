# Integración de SARSA en el Proyecto Final: Control Secuencial de Traslape Óptico
**Curso:** Sistemas de Decisión Autónoma bajo Incertidumbre (2026-2)  
**Profesor:** Hernán Felipe García Arias, PhD  
**Marco de referencia:** `W04_SARSA_Q_Learning_UdeA.pdf` y `Propuesta__Proyecto_final.pdf`

---

## 1. Justificación Teórica y Estratégica

En la propuesta oficial del proyecto final se plantearon 3 políticas de decisión para determinar cuándo detener la observación de símbolos IQ:
1. **Política A:** Ventana fija ($N$ predeterminado).
2. **Política B:** Umbrales de confianza fijos ($\eta_{\text{low}}, \eta_{\text{high}}$).
3. **Política C:** **Ecuación de Bellman / Rollout Monte Carlo** (parada óptima).

### El problema de la Política C analítica
Resolver la ecuación de Bellman exacta sobre el espacio de creencias continuas:
$$V_k(\pi) = \min \left\{ g_0(\pi), g_1(\pi), c_m + \mathbb{E}_{B \sim \pi}[V_{k+1}(\mathcal{T}(\pi, B))] \right\}$$
requiere calcular la esperanza matemática sobre bloques complejos de $m$ símbolos $B \in \mathbb{C}^m$, lo cual es analíticamente intratable y numéricamente costoso.

### La solución vía Clase W04
Tal como se formuló en la diapositiva 9 de la clase W04 (*"De Bellman al aprendizaje por muestras"*):
> En lugar de calcular toda la esperanza matemática del entorno, el agente actualiza su estimación $Q(s, a)$ mediante la interacción por muestras (Diferencia Temporal, TD).

**SARSA resuelve la Política C como un proceso de aprendizaje por refuerzo libre de modelo (model-free RL)**, cumpliendo con la invitación explícita de la diapositiva 39 de la clase: *"Un baseline heurístico y un agente RL aplicados al proyecto de curso"*.

---

## 2. Formulación como MDP en el Espacio de Creencias (Belief-MDP)

El pipeline de inferencia (KDE o BNN) reduce la alta dimensionalidad de las muestras $I/Q$ a un escalar de creencia $b_k \in [0, 1]$. El agente de control opera sobre este estado:

```text
[Bloque IQ B_k] ---> [Módulo Inferencia] ---> Creencia b_k ---> [Agente SARSA] ---> Acción a_k
                                                                        |
                                                    Recompensa r <------+
```

### A. Espacio de Estados ($\mathcal{S}$)
* **Estado continuo:** $b_k = P(\text{traslape} \mid B_{1:k}) \in [0, 1]$, que representa la probabilidad a posteriori de que la adquisición tenga separación $\text{ChSp} < B_{\text{nom}}$.
* **Discretización tabular:** Se discretiza $b_k$ en $N_b = 20$ a $50$ intervalos uniformes $\mathcal{S} = \{s^{(1)}, s^{(2)}, \dots, s^{(N_b)}\}$, opcionalmente complementado con el número de bloques observados $k \le K_{\max}$.

### B. Espacio de Acciones ($\mathcal{A}$)
El agente dispone de 3 acciones mutuamente excluyentes en cada paso:
* $a_0$: **Parar y declarar "Sin traslape"** (Acción terminal $\to$ finaliza el episodio).
* $a_1$: **Parar y declarar "Con traslape"** (Acción terminal $\to$ finaliza el episodio).
* $a_{\text{obs}}$: **Observar otro bloque** de $m = 100$ símbolos IQ (Acción de control $\to$ avanza al paso $k+1$).

### C. Función de Recompensa / Costos ($r_t$)
Basada en las penalizaciones operativas reales de la red óptica ($C_{\text{FN}} > C_{\text{FP}} \gg c_m$):
* Si ejecuta $a_0$ (declara sin traslape):
  $$r = \begin{cases} 0 & \text{si } z = 0 \text{ (acierto)} \\ -C_{\text{FN}} & \text{si } z = 1 \text{ (falso negativo, colisión de canales no detectada)} \end{cases}$$
* Si ejecuta $a_1$ (declara traslape):
  $$r = \begin{cases} 0 & \text{si } z = 1 \text{ (acierto)} \\ -C_{\text{FP}} & \text{si } z = 0 \text{ (falsa alarma, reconfiguración innecesaria)} \end{cases}$$
* Si ejecuta $a_{\text{obs}}$ (observa otro bloque):
  $$r = -c_m \quad (\text{costo de procesamiento / latencia por bloque})$$

---

## 3. Ventaja Crucial: ¿Por qué SARSA y no Q-Learning?

La presentación W04 demuestra que la diferencia fundamental radica en el objetivo de actualización TD:

$$\text{SARSA (On-Policy):} \quad y_t = r_{t+1} + \gamma Q(s_{t+1}, a_{t+1})$$
$$\text{Q-Learning (Off-Policy):} \quad y_t = r_{t+1} + \gamma \max_b Q(s_{t+1}, b)$$

### Analogía con el Acantilado (CliffWalking) en Redes Ópticas
| Entorno Didáctico (CliffWalking) | Entorno Real (Monitoreo de Traslape Óptico) |
| :--- | :--- |
| **Caída al acantilado** ($r = -100$) | **Falso Negativo** ($r = -C_{\text{FN}}$, ej. $-10$ a $-20$) |
| **Paso ordinario** ($r = -1$) | **Costo de bloque observado** ($r = -c_m = -0.01$) |
| **Comportamiento Q-learning** | Agresivo: asume decisiones perfectas y tiende a detenerse con muy pocas muestras cerca del umbral, provocando errores catastróficos. |
| **Comportamiento SARSA** | **Prudente (Risk-Averse):** incorpora la incertidumbre de la política. Al saber que equivocarse en una parada es fatal, aprende a **observar más bloques ($a_{\text{obs}}$) hasta que $b_k$ esté sólidamente alejado de la zona ambigua**. |

---

## 4. Algoritmo SARSA Tabular para Parada Óptima

```python
import numpy as np

def train_sarsa_stopping(episodes_data, n_bins=50, alpha=0.1, gamma=1.0, 
                         epsilon=0.1, n_episodes=5000, C_FN=10.0, C_FP=5.0, c_m=0.01):
    # Q-table: filas = bins de b_k [0..n_bins-1], columnas = [a0, a1, a_obs]
    Q = np.zeros((n_bins, 3))
    
    def get_state(b):
        return int(np.clip(b * n_bins, 0, n_bins - 1))
    
    def epsilon_greedy(s, eps):
        if np.random.rand() < eps:
            return np.random.choice(3)
        return int(np.argmax(Q[s]))
    
    for ep in range(n_episodes):
        # Muestra una adquisición de entrenamiento y su etiqueta verdadera z in {0, 1}
        trace_b, z = episodes_data.sample_trajectory()
        k = 0
        s = get_state(trace_b[k])
        a = epsilon_greedy(s, epsilon)
        
        while True:
            if a in (0, 1): # Acciones terminales
                r = (-C_FN if (a == 0 and z == 1) else 
                     (-C_FP if (a == 1 and z == 0) else 0.0))
                Q[s, a] += alpha * (r - Q[s, a])
                break
            else: # a == 2 (a_obs): pedir otro bloque
                r = -c_m
                k += 1
                if k >= len(trace_b) - 1: # Límite de captura
                    forced_a = 1 if trace_b[k] >= 0.5 else 0
                    r_term = -C_FN if (forced_a == 0 and z == 1) else (-C_FP if (forced_a == 1 and z == 0) else 0.0)
                    Q[s, a] += alpha * (r + r_term - Q[s, a])
                    break
                
                s_next = get_state(trace_b[k])
                a_next = epsilon_greedy(s_next, epsilon)
                
                # Actualización SARSA on-policy
                td_target = r + gamma * Q[s_next, a_next]
                Q[s, a] += alpha * (td_target - Q[s, a])
                
                s, a = s_next, a_next
                
    return Q
```

---

## 5. Matriz Comparativa de Políticas para la Sustentación

| Política | Criterio de Parada | ¿Requiere RL? | Qué evalúa en el proyecto |
| :--- | :--- | :---: | :--- |
| **A. Ventana fija** | Parar siempre al llegar a $N$ muestras fijas ($N \in \{100, 500, 2000\}$). | No | **Línea base tradicional:** demuestra la ineficiencia de procesar siempre el mismo número de muestras sin importar el ruido. |
| **B. Umbral de confianza** | Parar si $b_k \ge \eta_{\text{high}}$ o $b_k \le \eta_{\text{low}}$. | No | **Línea base bayesiana:** aprovecha la evidencia acumulada pero requiere ajuste manual de umbrales. |
| **C. Agente SARSA (RL)** | $a^* = \arg\max_a Q(s, a)$. | **Sí (W04)** | **Control óptimo por diferencia temporal:** balancea automáticamente el costo de retardo $c_m$ contra el riesgo asimétrico de error ($C_{\text{FN}}$ vs $C_{\text{FP}}$). |

---

## 6. Hoja de Ruta de Implementación en el Repositorio

Para integrar SARSA sin perturbar el código existente:
1. **Conservar intactos** los módulos `src/data/` (inventario y carga) y `src/kde/` o `src/bayes/` (cálculo de $b_k$).
2. **Crear el módulo de políticas:** `src/policy/sarsa.py` implementando la tabla $Q(s, a)$ y el evaluador secuencial.
3. **Generar el script comparativo:** `scripts/eval_policies.py` que aplique las 3 políticas sobre el conjunto de test independiente y reporte:
   * Tasa de Falsos Negativos (FNR) y Falsas Alarmas (FPR).
   * Número promedio de símbolos observados ($\bar{N}$).
   * Costo promedio total de operación $\hat{J} = \text{Costo de error} + c_m \bar{N}$.
