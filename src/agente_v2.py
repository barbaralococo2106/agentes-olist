"""
Etapa 2 — Agente analista sobre Olist: contexto, observabilidad y validación.

Cambios respecto de agente_v1.py (buscá los comentarios [CAMBIO N]):
  [CAMBIO 1] Context engineering: el esquema y reglas del dominio van en el prompt de sistema.
  [CAMBIO 2] Observabilidad: cada paso se guarda completo en logs/trazas.jsonl.
  [CAMBIO 3] Validación de la respuesta final + reintento si sale rota.
  [CAMBIO 4] temperature=0 para que las corridas sean comparables.
  [CAMBIO 5] Modo --demo: corre siempre las mismas preguntas (germen de las evals).

Uso:
    python src/agente_v2.py           # modo interactivo
    python src/agente_v2.py --demo    # corre las 3 preguntas de prueba
"""

import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import duckdb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DATA_DIR = Path(os.getenv("OLIST_DIR", "data/olist"))
LOG_FILE = Path("logs/trazas.jsonl")
MODEL = os.getenv("MODEL", "openai/gpt-oss-120b")
MAX_PASOS = 10
MAX_REINTENTOS = 1      # cuántas veces le pedimos que rehaga una respuesta final rota
MAX_FILAS = 50
MAX_CHARS = 4000

client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)


# ---------------------------------------------------------------------------
# Datos (igual que v1)
# ---------------------------------------------------------------------------
def cargar_base(data_dir: Path) -> duckdb.DuckDBPyConnection:
    csvs = sorted(data_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No encontré CSVs en {data_dir.resolve()}")
    con = duckdb.connect()
    for csv in csvs:
        nombre = csv.stem.replace("olist_", "").replace("_dataset", "")
        con.execute(
            f"CREATE VIEW {nombre} AS SELECT * FROM read_csv_auto('{csv.as_posix()}')"
        )
    return con


con = cargar_base(DATA_DIR)


def esquema_como_texto() -> str:
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
    return "\n".join(f"- {t}: {', '.join(cols)}" for t, cols in tablas.items())


# ---------------------------------------------------------------------------
# Herramientas
# [CAMBIO 1] Ya no hay herramienta describir_esquema: el esquema viaja en el prompt.
# ---------------------------------------------------------------------------
def ejecutar_sql(query: str) -> str:
    q = query.strip().lower()
    if not (q.startswith("select") or q.startswith("with")):
        return "ERROR: solo se permiten consultas SELECT/WITH."
    try:
        df = con.execute(query).fetchdf()
    except Exception as e:
        return f"ERROR al ejecutar SQL: {e}"
    texto = df.head(MAX_FILAS).to_string(index=False)
    if len(df) > MAX_FILAS:
        texto += f"\n... ({len(df)} filas en total, se muestran {MAX_FILAS})"
    return texto[:MAX_CHARS]


FUNCIONES = {"ejecutar_sql": ejecutar_sql}

TOOLS = [
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


# ---------------------------------------------------------------------------
# [CAMBIO 1] Prompt de sistema = rol + esquema + reglas del dominio + formato.
# Las reglas salen directo de los errores que vimos en la v1.
# ---------------------------------------------------------------------------
SYSTEM = f"""Sos un analista de datos senior. Respondés preguntas de negocio sobre
Olist, un e-commerce brasileño, consultando su base con SQL (dialecto DuckDB).

## Esquema disponible
{esquema_como_texto()}

## Reglas del dominio (importantes)
- Granularidad: orders tiene una fila por orden; order_items, una fila por producto dentro
  de la orden; order_payments puede tener varias filas por orden (cuotas o medios de pago).
- Reviews: order_reviews es a nivel ORDEN. Si la cruzás con order_items, una orden con
  varios productos repite su review. Para métricas de review por producto o categoría,
  deduplicá: usá SELECT DISTINCT review_id, review_score, product_category_name
  antes de promediar.
- Clientes: customer_id cambia en cada orden; la persona es customer_unique_id.
- Facturación: usá SUM(order_payments.payment_value) salvo que pidan otra definición.
- Promedios y rankings: informá siempre la cantidad de observaciones (n) y descartá grupos
  con menos de 30 observaciones, aclarando que lo hiciste.
- Categorías: traducilas al inglés uniendo con product_category_name_translation.

## Proceso
- Si una consulta falla, leé el error y corregila.
- Basá tu respuesta solo en resultados reales; no inventes números.

## Formato de la respuesta final
- Se lee en una terminal: texto plano en español, con viñetas (-). NO uses tablas markdown.
- Máximo 10 líneas. Incluí las cifras clave y una línea de advertencia si hay algún caveat."""


# ---------------------------------------------------------------------------
# [CAMBIO 2] Observabilidad: una línea JSON por evento, con TODO el contenido.
# ---------------------------------------------------------------------------
def registrar(evento: dict) -> None:
    LOG_FILE.parent.mkdir(exist_ok=True)
    evento["timestamp"] = datetime.now().isoformat(timespec="seconds")
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(evento, ensure_ascii=False, default=str) + "\n")


# ---------------------------------------------------------------------------
# [CAMBIO 3] Validación de la respuesta final (heurísticas simples).
# Devuelve la lista de problemas encontrados; vacía = respuesta OK.
# ---------------------------------------------------------------------------
def validar_respuesta(texto: str) -> list[str]:
    problemas = []
    t = texto.strip()
    if len(t) < 20:
        problemas.append("la respuesta está vacía o es demasiado corta")
    if "continúa" in t.lower():
        problemas.append("la respuesta quedó incompleta ('continúa')")
    if any(linea.strip().startswith("|") for linea in t.splitlines()):
        problemas.append("usaste una tabla markdown, que no está permitida")
    palabras = t.lower().split()
    if any(a == b and len(a) > 3 for a, b in zip(palabras, palabras[1:])):
        problemas.append("hay palabras repetidas seguidas (texto degenerado)")
    if "\n\n\n\n" in t:
        problemas.append("hay bloques de líneas vacías")
    return problemas


# ---------------------------------------------------------------------------
# El loop del agente
# ---------------------------------------------------------------------------
def agente(pregunta: str) -> str:
    run_id = uuid.uuid4().hex[:8]          # identifica todos los eventos de esta pregunta
    inicio = time.time()
    tokens_totales = 0
    reintentos = 0

    mensajes = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": pregunta},
    ]
    registrar({"run_id": run_id, "evento": "pregunta", "contenido": pregunta, "modelo": MODEL})

    for paso in range(1, MAX_PASOS + 1):
        t0 = time.time()
        resp = client.chat.completions.create(
            model=MODEL,
            messages=mensajes,
            tools=TOOLS,
            temperature=0,                   # [CAMBIO 4]
        )
        latencia_ms = int((time.time() - t0) * 1000)
        msg = resp.choices[0].message
        tokens = resp.usage.total_tokens if resp.usage else 0
        tokens_totales += tokens

        # --- Respuesta final (no pidió herramientas) ---
        if not msg.tool_calls:
            texto = msg.content or ""
            problemas = validar_respuesta(texto)
            registrar({
                "run_id": run_id, "evento": "respuesta", "paso": paso,
                "contenido": texto, "problemas": problemas,
                "tokens": tokens, "latencia_ms": latencia_ms,
            })

            if problemas and reintentos < MAX_REINTENTOS:
                # [CAMBIO 3] Le devolvemos su respuesta y le pedimos que la rehaga.
                reintentos += 1
                print(f"\n[paso {paso}] respuesta rechazada: {'; '.join(problemas)}. Reintentando...")
                mensajes.append({"role": "assistant", "content": texto})
                mensajes.append({
                    "role": "user",
                    "content": "Tu respuesta anterior tiene problemas: " + "; ".join(problemas)
                               + ". Reescribila completa respetando el formato pedido, "
                                 "usando los resultados que ya obtuviste.",
                })
                continue

            if problemas:
                texto += "\n\n[advertencia: la respuesta no pasó la validación de formato]"
            print(f"\n({paso} pasos, {tokens_totales} tokens, {time.time() - inicio:.1f} s)")
            return texto

        # --- Pedidos de herramientas (igual que v1) ---
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

        for tc in msg.tool_calls:
            nombre = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
                resultado = FUNCIONES[nombre](**args)
            except Exception as e:
                args = tc.function.arguments
                resultado = f"ERROR llamando a {nombre}: {e}"

            # [CAMBIO 2] En consola mostramos un resumen; en el log, todo.
            print(f"\n[paso {paso}] {nombre}")
            if isinstance(args, dict) and "query" in args:
                print("  SQL: " + args["query"].replace("\n", "\n       "))
            print("  -> " + resultado[:500].replace("\n", "\n     "))
            registrar({
                "run_id": run_id, "evento": "tool_call", "paso": paso,
                "herramienta": nombre, "argumentos": args, "resultado": resultado,
                "tokens": tokens, "latencia_ms": latencia_ms,
            })

            mensajes.append({"role": "tool", "tool_call_id": tc.id, "content": resultado})

    return f"Me quedé sin pasos ({MAX_PASOS}) antes de llegar a una respuesta."


# [CAMBIO 5] Las mismas preguntas de siempre, para comparar versiones.
PREGUNTAS_DEMO = [
    "¿Cuántas órdenes hay por estado de la orden?",
    "¿Cuáles son los 5 estados de Brasil con más facturación?",
    "¿Qué categorías de producto tienen peor review promedio?",
]

if __name__ == "__main__":
    print(f"Agente analista Olist v2 — modelo: {MODEL}. Trazas en {LOG_FILE}")
    if "--demo" in sys.argv:
        for p in PREGUNTAS_DEMO:
            print(f"\n{'=' * 70}\nPregunta> {p}")
            print("\n" + agente(p))
    else:
        print("Escribí 'salir' para terminar.")
        while True:
            pregunta = input("\nPregunta> ").strip()
            if pregunta.lower() in {"salir", "exit", "q"}:
                break
            if pregunta:
                print("\n" + agente(pregunta))
