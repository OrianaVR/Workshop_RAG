# Respuestas del workshop RAG
Escribe aquí tus respuestas con tus propias palabras. Pega tablas, resultados y enlaces a las capturas de `evidencias/`.

---

# Bloque 0 · Teoría

## T1 · Cuántos nodos viven en cada capa

_Evidencia:_

<img width="1600" height="1019" alt="WhatsApp Image 2026-10-22 at 9 59 01 AM" src="https://github.com/user-attachments/assets/b3cc5078-8e33-4904-902c-0e8141887a49" />


_Mi respuesta / análisis:_

**1. Cálculo a mano ($M = 16$ y $N = 1,000,000$ vectores)**

Dado que $m_L \approx \frac{1}{\ln(M)}$, la probabilidad de que un vector alcance al menos la capa $k$ es $P(l \ge k) = M^{-k}$. Por lo tanto, el número esperado de nodos en cada capa es $E[N_{\ge k}] = \frac{N}{M^k}$:

- *Capa $\ge 1$:* $\frac{1,000,000}{16^1} = \frac{1,000,000}{16} = \mathbf{62,500\text{ nodos}}$ ($6.25\%$)
- *Capa $\ge 2$:* $\frac{1,000,000}{16^2} = \frac{1,000,000}{256} = 3,906.25 \approx \mathbf{3,906\text{ nodos}}$ ($0.39\%$)
- *Capa $\ge 3$:* $\frac{1,000,000}{16^3} = \frac{1,000,000}{4,096} = 244.14 \approx \mathbf{244\text{ nodos}}$ ($0.024\%$)

**2. Simulación en Python y comparación con el valor teórico**

Código utilizado:
python
import numpy as np

M = 16
mL = 1 / np.log(M)
N = 1_000_000
rng = np.random.default_rng(0)
l = np.floor(-np.log(1 - rng.random(N)) * mL).astype(int)

for k in range(1, 4):
    print(f"capa >= {k}: simulado={(l >= k).sum():>7,}  teórico={N / M**k:>9,.0f}")


Salida obtenida:
text
capa >= 1: simulado= 62,675  teórico=   62,500
capa >= 2: simulado=  4,005  teórico=    3,906
capa >= 3: simulado=    270  teórico=      244


Comparación:Los resultados simulados coinciden muy de cerca con los valores teóricos esperados ($62,675$ vs. $62,500$ en capa $\ge 1$; $4,005$ vs. $3,906$ en capa $\ge 2$; y $270$ vs. $244$ en capa $\ge 3$). Las pequeñas diferencias se deben a la varianza natural del muestreo aleatorio con $N = 1,000,000$, confirmando que la distribución exponencial reduce la cantidad de nodos a razón de un factor de $\frac{1}{16}$ por cada capa que se sube.

**3. ¿Por qué es bueno para la velocidad de búsqueda que haya pocos hubs en las capas altas?**

Tener muy pocos nodos (hubs) en las capas superiores permite que la búsqueda greedy inicial (con ef = 1) avance con enlaces de largo alcance ("autopistas") evaluando distancias contra apenas unas decenas o cientos de vectores en lugar de explorar un millón de nodos. De esta manera, el algoritmo se aproxima rápidamente a la región correcta del espacio vectorial en tiempo logarítmico $O(\log N)$ antes de realizar la búsqueda exhaustiva en la capa base (capa 0).


## T2 · ef, k y el tamaño de la cola

_Evidencia:_

![T2](https://github.com/Azidian/WorkshopRAG/blob/main/evidencias/T2.png?raw=true)

_Mi respuesta / análisis:_

**Respuesta 1:** El problema es que el valor de ef es menor que la cantidad de vecinos solicitados (k). Se recomienda que ef ≥ k, ya que este parámetro controla la amplitud de búsqueda y la cantidad de candidatos considerados. Si se solicitan 10 vecinos y ef = 5, la búsqueda puede tener dificultades para encontrar los 10 vecinos más cercanos con buena calidad. Por eso, se debe aumentar ef al menos hasta 10 y, si es necesario, utilizar un valor mayor para mejorar el recall, teniendo en cuenta que esto puede aumentar el tiempo de búsqueda.

**Respuesta 2:** Conviene comenzar con un valor de ef igual o superior a k. Si el recall es bajo y se necesita encontrar vecinos más cercanos, se debe aumentar ef. Esto permite explorar más candidatos y mejorar la calidad de los resultados, aunque puede incrementar el tiempo de búsqueda. Por tanto, el valor debe ajustarse según el equilibrio deseado entre velocidad y calidad.

**Respuesta 3:** Sí, se puede modificar ef sin reconstruir el índice, porque es un parámetro que controla la búsqueda durante las consultas. En cambio, ef_construction determina la amplitud de búsqueda durante la construcción del grafo y afecta su calidad. Por eso, ef permite ajustar el equilibrio entre velocidad y recall sin necesidad de volver a construir el índice, mientras que modificar los parámetros de construcción puede requerir reconstruirlo.
.


## T3 · Inserción frente a búsqueda

_Evidencia:_

|                                | Inserción | Búsqueda |
|--------------------------------|-----------|----------|
| ¿Se sortea la capa?            | Sí, a cada vector nuevo se le saca al azar una capa máxima `l` con la fórmula `l = ⌊−ln(U)·mL⌋`. | No, la consulta no se guarda en el grafo, entonces siempre arranca desde la capa más alta. |
| Valor de ef en las capas altas | `ef = 1`, bajando desde arriba hasta la capa `l+1`. | `ef = 1`, bajando desde arriba hasta la capa 1. |
| Parámetro de la fase principal | `ef_construction`, en todas las capas desde `l` hasta la 0. | `ef`, pero solo en la capa 0. |
| ¿Modifica el índice?           | Sí, conecta el vector con sus `M` vecinos más cercanos y si algún nodo queda con demasiadas conexiones se le quita la más débil. | No, solo recorre el grafo y devuelve los `k` más cercanos. |

Lo saqué de las presentaciones de [Inserción](https://ia-eafit-rag.netlify.app/teoria/hnsw_insercion) y [Búsqueda](https://ia-eafit-rag.netlify.app/teoria/hnsw_busqueda).

_Mi respuesta / análisis:_

**Por qué de cada fila**

**1. Sorteo de capa**
En la inserción tiene sentido porque hay que decidir en qué capas va a quedar el vector, y hacerlo al azar evita que el grafo dependa del orden en que llegan los datos. En la búsqueda no se agrega nada, entonces no hay nada que sortear.

**2. ef en capas altas**
Las dos usan `ef = 1` porque ahí lo único que importa es llegar rápido a la zona correcta. Todavía no se necesita precisión, con quedarse con el mejor candidato es suficiente.

**3. Fase principal**
En la inserción hay que buscar buenos vecinos en cada capa donde va a vivir el vector, porque de eso depende qué tan bueno queda el grafo. En la búsqueda los resultados salen de la capa 0, que es la única que tiene todos los nodos, entonces ahí es donde vale la pena buscar con más cuidado.

**4. ¿Modifica el índice?**
Insertar es escribir en el grafo y buscar es solo leerlo. Por eso `ef` se puede cambiar en cada consulta sin reconstruir nada, pero `ef_construction` queda fijo cuando se construye el índice.

**¿Qué dispara cada llamada en el notebook 03?**

- `vectorstore.add_documents(...)` → **Inserción.** Cada chunk se pasa a embedding y Qdrant lo mete en su índice HNSW.
- `retriever.invoke(...)` → **Búsqueda.** La pregunta se convierte en embedding y Qdrant recorre el grafo para devolver los `k` chunks más parecidos que pasen el `score_threshold`.



## T4 · El invariante de FAISSVectorStore

### 1. Tabla posición → documento

El índice FAISS no cambia: sigue guardando 4 vectores en las posiciones 0–3.
Solo cambia la lista `self.documents`, de la que se borró B.

| Posición | Lo que FAISS cree que hay | `documents` antes | `documents` después de borrar B |
|---|---|---|---|
| 0 | vector de A | A | A |
| 1 | vector de B | B | C |
| 2 | vector de C | C | D |
| 3 | vector de D | D | (no existe) |

La búsqueda sigue devolviendo las posiciones `[1, 2, 3]`, pero ahora se traducen así:

| Posición devuelta | Antes | Ahora |
|---|---|---|
| 1 | B (correcto) | C (incorrecto) |
| 2 | C (correcto) | D (incorrecto) |
| 3 | D (correcto) | `IndexError` |

Resultado: `[C, D, IndexError]` en lugar de `[B, C, D]`.

### 2. Error silencioso vs. excepción

- **Errores silenciosos:** las posiciones 1 y 2. Devuelven un documento real, pero equivocado, y no se lanza ninguna excepción. El LLM recibiría
  contexto incorrecto y respondería con seguridad algo falso.
- **Excepción:** la posición 3. La lista ahora tiene 3 elementos (posiciones 0–2),
  así que `documents[3]` lanza `IndexError`.

### 3. Relación con Qdrant

FAISS solo guarda vectores y los une con el texto y la metadata por posición en
una lista aparte, así que el programador debe mantener ese invariante a mano.
Cualquier borrado o reordenamiento lo rompe sin avisar. Qdrant guarda el vector,
un ID y el *payload* (texto y metadata) juntos en un mismo registro. Al borrar un
punto se elimina todo a la vez y no hay una lista paralela que desincronizar.
Además ofrece persistencia, filtros por metadata y acceso concurrente.


## T5 · El costo de la fuerza bruta

_Evidencia:_ …

_Mi respuesta / análisis:_

### 1. Cuenta

Cada consulta compara contra 1 millón de vectores de 768 dimensiones:

    10^6 vectores × 768 multiplicaciones = 768 × 10^6 mult/consulta

Con 100 consultas por segundo:

    768 × 10^6 mult/consulta × 100 consultas/s = 76 800 × 10^6 mult/s

### 2. Notación científica y por qué la exactitud deja de ser viable

*≈ 7,68 × 10^10 multiplicaciones por segundo* (casi 77 mil millones).

Ser exacto obliga a comparar cada consulta contra todos los vectores, así que
el costo crece linealmente: con 10 millones de
vectores serían 7,68 × 10^11 mult/s, y con mil millones, 7,68 × 10^13 mult/s.
Además, cada consulta tendría que leer unos 3 GB de vectores de memoria
(10^6 × 768 × 4 bytes), lo que a 100 consultas por segundo exigiría cerca de
300 GB/s de ancho de banda, más de lo que ofrece un servidor típico. El
hardware necesario crece al mismo ritmo que los datos, y eso no es sostenible.

### 3. Qué se sacrifica y qué se gana en ANN

Se sacrifica la garantía de devolver siempre el vecino más cercano real
(el recall puede ser menor a 100 %), y se gana una búsqueda órdenes de magnitud
más rápida, con un costo que crece de forma aproximadamente logarítmica en
lugar de lineal.


---

# Bloque A · App RAG empresarial

# Reporte de Entregables RAG — SI3003

## Bloque A · App RAG Empresarial (Altozano Café)

### A1 · Ejecución RAG Demo (4 preguntas + 1 fuera de contexto)

| # | Pregunta | Respuesta RAG | Fuentes Recuperadas |
|---|---|---|---|
| 1 | ¿Cuál es el precio del plan Enterprise de CloudSync Pro? | El plan Enterprise cuesta $49/mes por usuario con compromiso anual. | CloudSync Pro — Precios |
| 2 | ¿Cuántos días pueden trabajar remotamente los empleados? | Los empleados pueden trabajar de forma remota hasta 3 días por semana con aprobación del jefe directo. | Política de trabajo remoto |
| 3 | ¿Qué pasa con los reembolsos mayores a $100? | Los reembolsos mayores a $100 requieren aprobación del supervisor y se procesan en 3 a 5 días hábiles. | Política de reembolsos |
| 4 | ¿Qué ocurre durante la primera semana de onboarding? | Durante la primera semana se completan módulos obligatorios de seguridad, cumplimiento y cultura de la empresa. | Onboarding |
| 5 | ¿Cuál es el horario de atención al cliente? | No encuentro esa información en los documentos proporcionados. | (ninguna) |

---

### A2 · Corpus Propio (`data/company_documents.json` — Altozano Café)

> **Justificación de `chunk_size`:** Se configuró un `CHUNK_SIZE = 400` y `CHUNK_OVERLAP = 40` debido a que los documentos de Altozano Café son reglas de negocio cortas (entre 150 y 300 caracteres cada una). Un chunk de 400 caracteres evita mezcla de políticas en un solo vector y mantiene la precisión temática.

| # | Pregunta | Esperado | Respuesta RAG | Fuentes Recuperadas | ¿Correcta? |
|---|---|---|---|---|:---:|
| 1 | ¿Cuánto cuesta la suscripción mensual de café? | $68.000 COP; 2 bolsas de 250 g | Cuesta $68.000 COP e incluye 2 bolsas de 250 g de café de origen con envío gratis. | Suscripciones de café | Sí |
| 2 | ¿En cuántos días se procesa un reembolso? | 5 a 7 días hábiles | Los reembolsos se procesan en 5 a 7 días hábiles al método de pago original. | Política de reembolsos | Sí |
| 3 | ¿Cuántos puntos necesito para una bebida gratis? | 100 puntos | Con 100 puntos del programa Altozano Puntos se canjea una bebida gratis. | Programa de fidelización | Sí |
| 4 | ¿Cada cuánto se descalcifican las máquinas de espresso? | Cada 3 meses | La descalcificación de las máquinas se realiza cada 3 meses. | Mantenimiento de máquinas | Sí |
| 5 | ¿Qué días se tuesta el café? | Lunes y jueves | El café se tuesta los días lunes y jueves en la planta de Envigado. | Origen y tueste del café | Sí |
| 6 | ¿Cuál es el salario base de un barista? | NO está en los documentos | No encuentro esa información en los documentos. | (ninguna) | Sí |
| 7 | ¿Aceptan pagos con criptomonedas? | NO está en los documentos | No encuentro esa información en los documentos. | (ninguna) | Sí |

---

### A3 · Ingeniería del Prompt (Antes vs. Después)

#### Diff del Prompt

```diff
- SYSTEM_PROMPT_ANTES = ("Eres un asistente empresarial. Responde únicamente con información "
-     "del contexto proporcionado. Si el contexto no contiene la respuesta, "
-     "indica que no tienes esa información. Responde en español y de forma concisa.")

+ SYSTEM_PROMPT_DESPUES = ("Eres un asistente empresarial. Responde únicamente con información "
+     "del contexto proporcionado. Cita siempre el título del documento entre "
+     "corchetes, por ejemplo [Política de reembolsos]. Si el contexto no alcanza, "
+     "responde exactamente: «No encuentro esa información en los documentos.» "
+     "Responde en español, en máximo 3 oraciones.")
# Bloque B · RAG sobre papers

## B1 · Ejecuta el RAG de papers y evalúa con el ground truth

_Evidencia:_ …

_Mi respuesta / análisis:_


## B2 · Experimentos con chunk_size, k y score_threshold

_Evidencia:_ …

_Mi respuesta / análisis:_


## B3 · Preguntas trampa: ¿el RAG admite que no sabe?

_Evidencia:_ …

_Mi respuesta / análisis:_


## B4 · Preguntas en español sobre papers en inglés

_Evidencia:_ …

_Mi respuesta / análisis:_


## B5 · Escribe tu propio quiz con ground truth

_Evidencia:_ …

_Mi respuesta / análisis:_


---

# Reflexión

## R1 · Reflexión final con tus datos

_Evidencia:_ …

_Mi respuesta / análisis:_


---

# Retos

## C1 · HNSW contra fuerza bruta con FAISS

_Evidencia:_ …

_Mi respuesta / análisis:_


## C2 · Un tercer paper: el RAG que explica HNSW

_Evidencia:_ …

_Mi respuesta / análisis:_


## C3 · App Gradio para los papers con filtro por paper

_Evidencia:_ …

_Mi respuesta / análisis:_


## C4 · Evaluación automática de la recuperación

_Evidencia:_ …

_Mi respuesta / análisis:_


## C5 · MMR frente a similitud pura

_Evidencia:_ …

_Mi respuesta / análisis:_


## C6 · Persistencia del índice en disco

_Evidencia:_ …

_Mi respuesta / análisis:_

