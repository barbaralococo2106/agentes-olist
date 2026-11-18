"""
Etapa 1 — Agente analista sobre Olist, con el loop escrito a mano (sin frameworks).

Idea central: un agente es un loop.
  1. Le mandás al modelo la conversación + la lista de herramientas disponibles.
  2. El modelo responde con texto (terminó) o con pedidos de herramientas (tool_calls).
  3. Vos ejecutás esas herramientas y le devolvés el resultado como mensajes "tool".
  4. Volvés al paso 1 hasta que responda sin pedir herramientas (o se agoten los pasos).

Uso:
    python agente_v1.py
"""

import json
import os
from pathlib import Path

import duckdb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DATA_DIR = Path(os.getenv("OLIST_DIR", "data/olist"))
MODEL = os.getenv("MODEL", "openai/gpt-oss-120b")  # revisá los modelos disponibles en console.groq.com
MAX_PASOS = 10          # freno de seguridad: evita loops infinitos (y gastar cuota)
MAX_FILAS = 50          # no le mandamos 100k filas al modelo: el contexto es finito
MAX_CHARS = 4000

# Groq expone una API compatible con OpenAI: mismo SDK, otra base_url.
client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)


# ---------------------------------------------------------------------------
# Datos: cada CSV de Olist se vuelve una vista de DuckDB (SQL sobre CSV, sin cargar nada)
# ---------------------------------------------------------------------------
def cargar_base(data_dir: Path) -> duckdb.DuckDBPyConnection:
    csvs = sorted(data_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No encontré CSVs en {data_dir.resolve()}")
    con = duckdb.connect()
    for csv in csvs:
        # olist_orders_dataset.csv -> orders
        nombre = csv.stem.replace("olist_", "").replace("_dataset", "")
        con.execute(
            f"CREATE VIEW {nombre} AS SELECT * FROM read_csv_auto('{csv.as_posix()}')"
        )
    return con


con = cargar_base(DATA_DIR)


# ---------------------------------------------------------------------------
# Herramientas: funciones Python comunes. El modelo nunca las ejecuta; las pide.
# ---------------------------------------------------------------------------
def describir_esquema() -> str:
    """Devuelve tablas y columnas con su tipo."""
    filas = con.execute(
        """
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        ORDER BY table_name, ordinal_position
        """
    ).fetchall()
    tablas: dict[str, list[str]] = {}
    for tabla, columna, tipo in filas:
        tablas.setdefault(tabla, []).append(f"{columna} ({tipo})")
    return "\n".join(f"{t}: {', '.join(cols)}" for t, cols in tablas.items())


def ejecutar_sql(query: str) -> str:
    """Ejecuta una consulta de solo lectura y devuelve hasta MAX_FILAS filas."""
    q = query.strip().lower()
    if not (q.startswith("select") or q.startswith("with")):
        return "ERROR: solo se permiten consultas SELECT/WITH."
    try:
        df = con.execute(query).fetchdf()
    except Exception as e:
        # Clave: devolvemos el error al modelo en vez de cortar.
        # Así puede leerlo y corregir su propio SQL en el próximo paso.
        return f"ERROR al ejecutar SQL: {e}"
    texto = df.head(MAX_FILAS).to_string(index=False)
    if len(df) > MAX_FILAS:
        texto += f"\n... ({len(df)} filas en total, se muestran {MAX_FILAS})"
    return texto[:MAX_CHARS]


FUNCIONES = {
    "describir_esquema": describir_esquema,
    "ejecutar_sql": ejecutar_sql,
}

# La descripción de cada herramienta ES prompt: es lo único que el modelo sabe de ella.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "describir_esquema",
            "description": "Lista las tablas disponibles con sus columnas y tipos. "
                           "Usala antes de escribir SQL si no conocés el esquema.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ejecutar_sql",
            "description": "Ejecuta una consulta SQL (dialecto DuckDB, solo SELECT/WITH) "
                           f"y devuelve hasta {MAX_FILAS} filas.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "La consulta SQL a ejecutar."}
                },
                "required": ["query"],
            },
        },
    },
]

SYSTEM = """Sos un analista de datos experto. Respondés preguntas de negocio sobre
Olist, un e-commerce brasileño, consultando su base con SQL.
- Si no conocés el esquema, consultalo primero.
- Si una consulta falla, leé el error y corregila.
- Basá tu respuesta solo en resultados reales de las consultas; no inventes números.
- Respondé en español, breve, mostrando las cifras clave."""


# ---------------------------------------------------------------------------
# El loop del agente
# ---------------------------------------------------------------------------
def agente(pregunta: str) -> str:
    mensajes = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": pregunta},
    ]

    for paso in range(1, MAX_PASOS + 1):
        resp = client.chat.completions.create(model=MODEL, messages=mensajes, tools=TOOLS)
        msg = resp.choices[0].message

        # Sin pedidos de herramientas = respuesta final.
        if not msg.tool_calls:
            return msg.content or ""

        # Guardamos el pedido del modelo en el historial (si no, se "olvida" de lo que pidió).
        mensajes.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ],
        })

        # Ejecutamos cada herramienta pedida y devolvemos su resultado.
        for tc in msg.tool_calls:
            nombre = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
                resultado = FUNCIONES[nombre](**args)
            except Exception as e:
                resultado = f"ERROR llamando a {nombre}: {e}"

            print(f"\n[paso {paso}] {nombre}({tc.function.arguments})")
            print("  -> " + resultado[:300].replace("\n", "\n     "))

            mensajes.append({"role": "tool", "tool_call_id": tc.id, "content": resultado})

    return f"Me quedé sin pasos ({MAX_PASOS}) antes de llegar a una respuesta."


if __name__ == "__main__":
    print(f"Agente analista Olist — modelo: {MODEL}. Escribí 'salir' para terminar.")
    while True:
        pregunta = input("\nPregunta> ").strip()
        if pregunta.lower() in {"salir", "exit", "q"}:
            break
        if pregunta:
            print("\n" + agente(pregunta))
