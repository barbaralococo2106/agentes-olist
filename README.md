# Agente analista sobre Olist

Proyecto de aprendizaje: construir un agente con LLMs, de un loop escrito a mano a un sistema en producción, sobre el dataset público de e-commerce de Olist.

## Etapas

| Etapa | Qué se construye | Estado |
|---|---|---|
| 1. Loop a mano | Agente con tool calling y SQL, sin frameworks | ✅ |
| 2. Más herramientas y robustez | Esquema, gráficos y recuperación de errores | ⏳ |
| 3. RAG sobre reseñas | Routing entre SQL y búsqueda semántica | ⏳ |
| 4. Evals | Set de preguntas, execution accuracy y LLM como juez | ⏳ |
| 5. Frameworks y MCP | LangGraph y servidor MCP | ⏳ |
| 6. Producción | Trazas, costos, guardrails y deploy | ⏳ |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # completar GROQ_API_KEY
```

Datos: descargar "Brazilian E-Commerce Public Dataset by Olist" de Kaggle y descomprimir los CSV en `data/olist/`.

## Uso

```bash
python src/agente_v1.py
```

Stack: Python, DuckDB, SDK de OpenAI apuntando a Groq (plan gratuito).
