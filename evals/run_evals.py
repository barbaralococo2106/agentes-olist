"""
Etapa 4 — Evals del agente analista.

Corre cada caso de evals/casos.yaml contra una versión del agente y lo mide de 3 formas:

  1. Ejecución (determinística): ¿alguna consulta SQL del agente devolvió el mismo
     resultado que la consulta de referencia? Es la métrica estándar de text-to-SQL
     ("execution accuracy"). No mira la respuesta en texto, solo los datos.
  2. Valores (determinística): ¿las cifras y nombres clave de la referencia aparecen
     en la respuesta final que lee el usuario?
  3. Juez (LLM-as-judge): otro modelo compara la respuesta con la referencia y decide
     correcta / parcial / incorrecta. Capta lo que las reglas fijas no ven
     (p. ej. el caso sin datos), pero es menos confiable: por eso no es la única métrica.

Uso (desde la raíz del repo, con el .venv activo):
    python evals/run_evals.py --agente v2
    python evals/run_evals.py --agente v1
    python evals/run_evals.py --agente v2 --casos c02,c10
    python evals/run_evals.py --agente v2 --sin-juez
"""

import argparse
import contextlib
import importlib
import io
import json
import math
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import openai
import pandas as pd
import yaml

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

CASOS_FILE = RAIZ / "evals" / "casos.yaml"
RESULTADOS_DIR = RAIZ / "evals" / "resultados"
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-oss-20b")


# ---------------------------------------------------------------------------
# Llamadas al modelo: reintentos ante rate limit + conteo de llamadas y tokens
# ---------------------------------------------------------------------------
def llamar_con_reintentos(fn, **kwargs):
    for intento in range(6):
        try:
            return fn(**kwargs)
        except openai.RateLimitError as e:
            espera = 5 * 2 ** intento
            try:
                espera = float(e.response.headers.get("retry-after", espera)) + 1
            except Exception:
                pass
            print(f"    (límite de uso de la API: espero {espera:.0f} s y reintento)", file=sys.__stdout__)
            time.sleep(espera)
    raise RuntimeError("Se superó el límite de uso de la API varias veces seguidas.")


class Medidor:
    """Reemplaza client.chat.completions.create del agente para contar llamadas y tokens."""

    def __init__(self, crear_original):
        self.crear_original = crear_original
        self.reset()

    def reset(self):
        self.llamadas = 0
        self.tokens = 0

    def __call__(self, **kwargs):
        resp = llamar_con_reintentos(self.crear_original, **kwargs)
        self.llamadas += 1
        if getattr(resp, "usage", None):
            self.tokens += resp.usage.total_tokens or 0
        return resp


# ---------------------------------------------------------------------------
# Métrica 1: comparar resultados de SQL (execution accuracy)
# ---------------------------------------------------------------------------
def cerca(valor, referencia):
    return abs(valor - referencia) <= max(0.01 * abs(referencia), 0.011)


def _normalizar(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    try:
        return float(str(v))
    except ValueError:
        return str(v).strip().lower()


def _misma_columna(ref_vals, ag_vals, ordenado):
    if len(ref_vals) != len(ag_vals):
        return False
    if not ordenado:
        clave = lambda x: (x is None, type(x).__name__, x if x is not None else 0)
        ref_vals, ag_vals = sorted(ref_vals, key=clave), sorted(ag_vals, key=clave)
    for r, a in zip(ref_vals, ag_vals):
        if isinstance(r, float) and isinstance(a, float):
            if not cerca(a, r):
                return False
        elif r != a:
            return False
    return True


def coincide(ref: pd.DataFrame, ag: pd.DataFrame, ordenado: bool) -> bool:
    """Cada columna de la referencia tiene que estar en alguna columna del agente
    (los nombres de columna no importan; columnas extra del agente se toleran)."""
    if ordenado:
        ag = ag.head(len(ref))
    if len(ag) != len(ref):
        return False
    cols_agente = [[_normalizar(v) for v in ag[c].tolist()] for c in ag.columns]
    for c in ref.columns:
        ref_vals = [_normalizar(v) for v in ref[c].tolist()]
        if not any(_misma_columna(ref_vals, av, ordenado) for av in cols_agente):
            return False
    return True


# ---------------------------------------------------------------------------
# Métrica 2: buscar los valores clave en el texto de la respuesta
# ---------------------------------------------------------------------------
SEP_MILES = re.compile(r"(?<=\d)[    ](?=\d{3}(?!\d))")


def _interpretar(tok):
    """'5.998.226,96' / '5,998,226.96' / '96 478' / '3,62' -> posibles floats."""
    if "," in tok and "." in tok:
        dec = "," if tok.rfind(",") > tok.rfind(".") else "."
        mil = "." if dec == "," else ","
        return {float(tok.replace(mil, "").replace(dec, "."))}
    for sep in (",", "."):
        if sep in tok:
            otro = "." if sep == "," else ","
            if re.fullmatch(rf"\d{{1,3}}(\{sep}\d{{3}})+", tok):
                opciones = {float(tok.replace(sep, ""))}       # separador de miles
                if tok.count(sep) == 1:
                    opciones.add(float(tok.replace(sep, ".")))  # o decimal (ambiguo)
                return opciones
            if tok.count(sep) == 1:
                return {float(tok.replace(sep, "."))}
            return set()
    return {float(tok)}


def numeros_en_texto(texto):
    t = SEP_MILES.sub("", texto)
    numeros = set()
    for tok in re.findall(r"\d[\d.,]*\d|\d", t):
        numeros |= _interpretar(tok)
    return numeros


def valores_en_respuesta(ref: pd.DataFrame, columnas, respuesta: str):
    """Devuelve (encontrados, total) de los valores clave de la referencia."""
    numeros = numeros_en_texto(respuesta)
    texto = respuesta.lower()
    encontrados = total = 0
    for c in columnas:
        for v in ref[c].tolist():
            total += 1
            v = _normalizar(v)
            if isinstance(v, float):
                encontrados += any(cerca(n, v) for n in numeros)
            elif v is not None:
                encontrados += v in texto
    return encontrados, total


# ---------------------------------------------------------------------------
# Métrica 3: LLM como juez
# ---------------------------------------------------------------------------
PROMPT_JUEZ = """Sos un evaluador estricto de un agente analista de datos.

Pregunta del usuario:
{pregunta}

Respuesta correcta de referencia:
{referencia}

Respuesta del agente:
{respuesta}

Criterios:
- "correcta": responde la pregunta y las cifras o nombres clave coinciden con la referencia
  (se toleran redondeos y formatos distintos).
- "parcial": va en la dirección correcta pero falta información o alguna cifra difiere.
- "incorrecta": cifras equivocadas, no responde lo que se preguntó o inventa datos.

Respondé SOLO con un JSON: {{"veredicto": "correcta" | "parcial" | "incorrecta", "motivo": "una oración"}}"""


def juzgar(client, caso, referencia, respuesta):
    prompt = PROMPT_JUEZ.format(pregunta=caso["pregunta"], referencia=referencia, respuesta=respuesta)
    try:
        resp = llamar_con_reintentos(
            client.chat.completions.create,
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        return data.get("veredicto", "error_juez"), data.get("motivo", "")
    except Exception as e:
        return "error_juez", f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Correr un caso
# ---------------------------------------------------------------------------
def correr_caso(modulo, medidor, consultas, caso, usar_juez):
    consultas.clear()
    medidor.reset()

    t0 = time.time()
    salida_agente = io.StringIO()
    error = ""
    try:
        with contextlib.redirect_stdout(salida_agente):   # silenciamos los prints del agente
            respuesta = modulo.agente(caso["pregunta"])
    except Exception as e:
        respuesta, error = "", f"{type(e).__name__}: {e}"
    segundos = time.time() - t0

    fila = {
        "id": caso["id"], "dificultad": caso["dificultad"], "pregunta": caso["pregunta"],
        "ejecucion": None, "valores": None, "juez": None, "motivo_juez": "",
        "llamadas": medidor.llamadas, "tokens": medidor.tokens, "segundos": round(segundos, 1),
        "respuesta": respuesta, "consultas": json.dumps(consultas, ensure_ascii=False), "error": error,
    }

    if caso.get("sql_referencia"):
        ref = modulo.con.execute(caso["sql_referencia"]).fetchdf()
        referencia_txt = ref.to_string(index=False)

        # Métrica 1: ¿alguna consulta del agente coincide con la referencia?
        fila["ejecucion"] = False
        for q in consultas:
            try:
                if coincide(ref, modulo.con.execute(q).fetchdf(), caso.get("ordenado", False)):
                    fila["ejecucion"] = True
                    break
            except Exception:
                continue

        # Métrica 2: ¿los valores clave aparecen en la respuesta?
        columnas = caso.get("columnas_clave", list(ref.columns))
        encontrados, total = valores_en_respuesta(ref, columnas, respuesta)
        fila["valores"] = encontrados == total
        fila["valores_detalle"] = f"{encontrados}/{total}"
    else:
        referencia_txt = caso["respuesta_esperada"]

    # Métrica 3: juez
    if usar_juez and not error:
        fila["juez"], fila["motivo_juez"] = juzgar(modulo.client, caso, referencia_txt, respuesta)

    return fila


# ---------------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------------
def simbolo(v):
    return {True: "✓", False: "✗", None: "–"}.get(v, str(v))


def imprimir_resumen(df, agente, modelo):
    print(f"\n{'=' * 78}\nRESUMEN — agente {agente} ({modelo}), {len(df)} casos\n{'=' * 78}")

    con_ref = df[df["ejecucion"].notna()]
    if len(con_ref):
        print(f"Ejecución (SQL correcto):        {con_ref['ejecucion'].astype(bool).sum()}/{len(con_ref)}"
              f"  ({con_ref['ejecucion'].astype(bool).mean():.0%})")
        print(f"Valores en la respuesta:         {con_ref['valores'].astype(bool).sum()}/{len(con_ref)}"
              f"  ({con_ref['valores'].astype(bool).mean():.0%})")
    juzgados = df[df["juez"].notna()]
    if len(juzgados):
        puntaje = juzgados["juez"].map({"correcta": 1, "parcial": 0.5}).fillna(0)
        print(f"Juez (correcta=1, parcial=0.5):  {puntaje.sum():g}/{len(juzgados)}  ({puntaje.mean():.0%})")

    print(f"Costo: {df['llamadas'].sum()} llamadas, {df['tokens'].sum():,} tokens, "
          f"{df['segundos'].mean():.1f} s promedio por pregunta")

    if len(juzgados):
        print("\nPor dificultad (juez = correcta):")
        for dif, g in juzgados.groupby("dificultad", sort=False):
            print(f"  {dif:<10} {(g['juez'] == 'correcta').sum()}/{len(g)}")


def main():
    parser = argparse.ArgumentParser(description="Evals del agente analista")
    parser.add_argument("--agente", default="v2", help="versión del agente: v1, v2, ...")
    parser.add_argument("--casos", help="ids separados por coma, p. ej. c01,c10")
    parser.add_argument("--sin-juez", action="store_true", help="no usar el LLM como juez")
    args = parser.parse_args()

    casos = yaml.safe_load(CASOS_FILE.read_text(encoding="utf-8"))
    if args.casos:
        elegidos = set(args.casos.split(","))
        casos = [c for c in casos if c["id"] in elegidos]

    # Importamos el agente y "espiamos" sus herramientas y llamadas al modelo
    # sin modificar su código.
    modulo = importlib.import_module(f"agente_{args.agente}")
    medidor = Medidor(modulo.client.chat.completions.create)
    modulo.client.chat.completions.create = medidor

    consultas = []
    ejecutar_original = modulo.FUNCIONES["ejecutar_sql"]

    def ejecutar_espiado(query):
        consultas.append(query)
        return ejecutar_original(query)

    modulo.FUNCIONES["ejecutar_sql"] = ejecutar_espiado

    print(f"Evaluando agente {args.agente} ({modulo.MODEL}) en {len(casos)} casos. "
          f"Juez: {'no' if args.sin_juez else JUDGE_MODEL}\n")
    print(f"{'caso':<5} {'dificultad':<10} {'ejec':^5} {'valores':^8} {'juez':<11} {'llam':>4} {'tokens':>7} {'seg':>5}")

    filas = []
    for caso in casos:
        fila = correr_caso(modulo, medidor, consultas, caso, not args.sin_juez)
        filas.append(fila)
        print(f"{fila['id']:<5} {fila['dificultad']:<10} {simbolo(fila['ejecucion']):^5} "
              f"{fila.get('valores_detalle', '–'):^8} {str(fila['juez'] or '–'):<11} "
              f"{fila['llamadas']:>4} {fila['tokens']:>7} {fila['segundos']:>5}"
              + (f"  ERROR: {fila['error'][:60]}" if fila["error"] else ""))

    df = pd.DataFrame(filas)
    imprimir_resumen(df, args.agente, modulo.MODEL)

    RESULTADOS_DIR.mkdir(exist_ok=True)
    archivo = RESULTADOS_DIR / f"{args.agente}_{datetime.now():%Y%m%d_%H%M}.csv"
    df.to_csv(archivo, index=False)
    print(f"\nDetalle completo (respuestas, SQL, motivos del juez) en {archivo.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
