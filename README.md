# LLM Benchmark — Robustez Lexical

Herramienta para evaluar qué tan robustos son los modelos de lenguaje frente a perturbaciones lingüísticas en preguntas de razonamiento verbal en español. El pipeline toma un dataset original, le aplica un ataque (transformación del texto) y guarda el dataset perturbado listo para ser evaluado.

## Objetivo

Medir la robustez de un LLM ante cambios en la superficie textual de las preguntas, sin alterar la respuesta correcta. Si el modelo responde correctamente el dataset original pero falla en el perturbado, el modelo está usando atajos lingüísticos en lugar de razonamiento real.

---

## Estructura del proyecto

```
llm-benchmark/
├── attacks/                  # Implementaciones de los ataques
│   ├── base.py               # Clase abstracta Attack
│   ├── synonym.py            # Reemplazo por sinonimos
│   ├── paraphrase.py         # Parafrasis preservando significado
│   ├── minimal_pair.py       # Cambio de un elemento critico
│   └── shortcut_removal.py   # Eliminacion de conectores logicos
├── evaluation/
│   └── metrics.py            # accuracy, delta_accuracy, flip_rate
├── validation/
│   └── semantic.py           # Validacion de similitud semantica (pendiente)
├── utils/
│   ├── llm_client.py         # Cliente OpenAI-compatible para Ollama
│   └── text_utils.py
├── data/
│   ├── raw/                  # Datasets originales
│   └── perturbed/            # Datasets perturbados (output del pipeline)
├── config.py                 # Parametros globales (modelo, paths, intensidad)
└── pipeline.py               # Script principal
```

---

## Requisitos

- Python 3.10+
- [Ollama](https://ollama.com) instalado y con el modelo descargado

### Dependencias Python
Las dependencias principales son:

| Paquete | Uso |
|---|---|
| `openai` | Cliente HTTP hacia la API compatible de Ollama |
| `httpx` | Verificar si Ollama esta corriendo |
| `tqdm` | Barra de progreso en el pipeline |

### Modelo LLM local

El pipeline usa Ollama como backend de inferencia. El modelo por defecto es `qwen2.5`:

```bash
ollama pull qwen2.5
```

Para cambiar el modelo, edita `config.py`:

```python
OLLAMA_MODEL = "qwen2.5"  # o cualquier modelo disponible en Ollama
```

---

## Uso

### Correr el pipeline

```bash
python pipeline.py --attack <tipo> [--intensity <valor>] [--dataset <ruta>] [--output <ruta>]
```

**Argumentos:**

| Argumento | Descripcion | Default |
|---|---|---|
| `--attack` | Tipo de ataque a aplicar | requerido |
| `--intensity` | Intensidad del ataque, valor entre 0 y 1 | `0.3` |
| `--dataset` | Ruta al dataset JSON de entrada | `data/raw/sample_dataset.json` |
| `--output` | Ruta donde guardar el resultado | auto-generada |

**Tipos de ataque disponibles:**

| Ataque | Descripcion |
|---|---|
| `synonym` | Reemplaza ~N% de las palabras por sinonimos equivalentes |
| `paraphrase` | Reescribe las oraciones preservando el significado |
| `minimal_pair` | Cambia un solo elemento critico (negacion, cuantificador o conector temporal) |
| `shortcut_removal` | Elimina conectores logicos y causales del texto |

### Ejemplos

```bash
# Ataque de sinonimos con intensidad media (default)
python pipeline.py --attack synonym

# Parafrasis con intensidad alta
python pipeline.py --attack paraphrase --intensity 0.5

# Minimal pair sobre un dataset especifico
python pipeline.py --attack minimal_pair --dataset data/raw/dataset.json

# Shortcut removal guardando en ruta personalizada
python pipeline.py --attack shortcut_removal --output data/perturbed/mi_salida.json
```

El archivo de salida se guarda por defecto en:
```
data/perturbed/<nombre_dataset>_<ataque>_<intensidad>.json
```

---

## Formato del dataset

Cada elemento del JSON tiene la siguiente estructura:

```json
{
  "id": 0,
  "task": "sentence_ordering",
  "question": "Texto de la pregunta...",
  "options": ["opcion A", "opcion B", "opcion C", "opcion D", "opcion E"],
  "answer": 3,
  "rationale": "Explicacion de la respuesta correcta."
}
```

Los ataques modifican los campos `question` y/o `options`. El campo `answer` nunca se modifica.

---

## Estado del proyecto

| Componente | Estado |
|---|---|
| Ataques (synonym, paraphrase, minimal_pair, shortcut_removal) | Implementado |
| Pipeline con paralelismo | Implementado |
| Metricas (accuracy, delta_accuracy, flip_rate) | Implementado |
| Validacion semantica | Pendiente (placeholder) |
| Evaluacion del modelo sobre datasets perturbados | Pendiente |

La siguiente fase del proyecto consiste en conectar un modelo evaluador que responda las preguntas del dataset original y del perturbado, y calcular las metricas de robustez (delta de accuracy y flip rate).
