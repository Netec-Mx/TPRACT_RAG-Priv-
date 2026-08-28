# Construcción de un auditor RAG para detectar inconsistencias entre documentos

## Metadatos

| Campo | Valor |
|---|---|
| Duración | 105 minutos |
| Complejidad | Alta |
| Nivel de Bloom | Aplicar |

## Descripción general

En este laboratorio construirás desde cero un auditor documental basado en RAG híbrido. El sistema recuperará evidencia desde documentos versionados usando búsqueda vectorial en Qdrant y búsqueda léxica BM25, aplicará reranking con diversidad y autoridad documental, y pedirá a un LLM que detecte contradicciones con evidencia trazable.

El resultado será un proyecto reproducible llamado `rag-auditor`, con corpus de prueba, colección Qdrant, índice BM25 serializado, casos de evaluación etiquetados, informes JSON de auditoría y métricas de precisión, recall y F1.

## Objetivos de aprendizaje

Al finalizar esta práctica podrás:

- [ ] Preparar y normalizar un corpus documental con versiones, estados y conflictos deliberados.
- [ ] Crear fragmentos con procedencia completa: documento, versión, sección, vigencia, fuente y autoridad.
- [ ] Indexar fragmentos mediante embeddings `text-embedding-3-small` en Qdrant y BM25 local.
- [ ] Implementar recuperación híbrida, reranking por relevancia y diversidad de fuentes.
- [ ] Generar, validar y evaluar informes de auditoría JSON con detección de inconsistencias.

## Requisitos previos

### Conocimientos

Debes comprender los siguientes conceptos:

- Recuperación semántica, búsqueda léxica BM25 y similitud coseno.
- Embeddings y bases de datos vectoriales.
- Fragmentación documental y preservación de metadatos.
- Diseño de prompts para respuestas fundamentadas en evidencia.
- Precisión, recall, F1 y matriz de confusión.
- Python intermedio: entornos virtuales, archivos JSON, funciones y variables de entorno.

### Acceso y herramientas

Debes disponer de:

- Docker Engine 26.0.0 y Docker Compose 2.26.0 operativos.
- Python 3.12.1.
- Una clave de OpenAI configurada en `OPENAI_API_KEY`.
- Permisos para usar:
  - `text-embedding-3-small`
  - `gpt-4o-mini-2024-07-18`
- Conexión a Internet.
- Al menos 16 GB de RAM y 20 GB de espacio libre.

## Entorno del laboratorio

### Recursos recomendados

| Recurso | Mínimo | Recomendado |
|---|---:|---:|
| CPU | 4 núcleos físicos | 8 núcleos físicos |
| RAM | 16 GB | 32 GB |
| Disco libre | 20 GB | 30 GB |
| Red | Conexión estable | Conexión de baja latencia |

### Versiones utilizadas

| Componente | Versión |
|---|---|
| Python | 3.12.1 |
| Qdrant | 1.8.4 |
| Imagen Docker | `qdrant/qdrant:v1.8.4` |
| qdrant-client | 1.8.2 |
| OpenAI SDK | 1.13.3 |
| rank-bm25 | 0.2.2 |
| Pydantic | 2.6.3 |
| pandas | 2.2.1 |
| scikit-learn | 1.4.1.post1 |

### Preparación inicial

> En Linux y macOS debes trabajar en `~/rag-auditor`. En Windows, ejecuta estos comandos dentro de WSL 2 y utiliza `/home/<usuario>/rag-auditor`.

```bash
cd ~
mkdir -p rag-auditor/{data/raw,data/processed,data/qdrant_storage,notebooks,src,tests,outputs,config}
cd ~/rag-auditor
```

Crea y activa un entorno virtual:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

En PowerShell con WSL, sigue utilizando el comando anterior dentro de la terminal Linux. Instala las dependencias:

```bash
python -m pip install --upgrade pip==23.3.2

pip install \
  qdrant-client==1.8.2 \
  openai==1.13.3 \
  rank-bm25==0.2.2 \
  pydantic==2.6.3 \
  pandas==2.2.1 \
  scikit-learn==1.4.1.post1 \
  python-dotenv==1.0.1
```

Crea el archivo `.env`:

```bash
cat > .env <<'EOF'
OPENAI_API_KEY=REEMPLAZAR_CON_TU_CLAVE
EOF
```

Protege el archivo para evitar incluirlo en control de versiones:

```bash
cat > .gitignore <<'EOF'
.venv/
.env
__pycache__/
*.pyc
data/qdrant_storage/
outputs/
EOF
```

---

## Procedimiento paso a paso

## Paso 1. Levantar Qdrant con Docker Compose

**Objetivo:** Crear el servicio persistente de Qdrant 1.8.4 con el nombre de proyecto, contenedor, puertos e imagen requeridos.

### Instrucciones

1. Crea el archivo `docker-compose.yml`:

```bash
cat > docker-compose.yml <<'EOF'
name: rag_auditor

services:
  qdrant:
    image: qdrant/qdrant:v1.8.4
    container_name: rag-auditor-qdrant
    restart: unless-stopped
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - ./data/qdrant_storage:/qdrant/storage
EOF
```

2. Inicia Qdrant en segundo plano:

```bash
docker compose up -d
```

3. Comprueba el estado del contenedor:

```bash
docker compose ps
```

4. Verifica la disponibilidad de la API REST:

```bash
curl http://localhost:6333/collections
```

### Salida esperada

Debes observar un contenedor llamado `rag-auditor-qdrant` en estado `running`. La consulta REST debe devolver una respuesta JSON similar a:

```json
{
  "result": {
    "collections": []
  },
  "status": "ok",
  "time": 0.0
}
```

### Verificación

Ejecuta:

```bash
docker inspect --format='{{.Name}} - {{.Config.Image}}' rag-auditor-qdrant
```

La salida debe contener:

```text
/rag-auditor-qdrant - qdrant/qdrant:v1.8.4
```

---

## Paso 2. Crear un corpus documental versionado y contradictorio

**Objetivo:** Construir un corpus controlado de políticas, procedimientos, fichas técnicas y comunicaciones internas que permita evaluar detección de contradicciones.

### Instrucciones

1. Crea el archivo `data/raw/documents.json`:

```bash
cat > data/raw/documents.json <<'EOF'
[
  {
    "document_id": "POL-FIN-001",
    "title": "Política Corporativa de Gastos",
    "document_type": "política",
    "version": "3.0",
    "effective_date": "2024-01-01",
    "status": "vigente",
    "source": "Dirección Financiera",
    "authority_level": 3,
    "sections": [
      {
        "section": "4.2 Límites de aprobación",
        "text": "Toda solicitud de gasto superior a 500 EUR requiere aprobación previa de la Dirección Financiera. Los responsables de área pueden aprobar gastos de hasta 500 EUR."
      },
      {
        "section": "5.1 Retención documental",
        "text": "Los comprobantes y registros financieros deberán conservarse durante siete años desde el cierre del ejercicio fiscal correspondiente."
      }
    ]
  },
  {
    "document_id": "PROC-FIN-014",
    "title": "Procedimiento Operativo de Reembolsos",
    "document_type": "procedimiento",
    "version": "2.1",
    "effective_date": "2023-06-15",
    "status": "vigente",
    "source": "Operaciones Financieras",
    "authority_level": 2,
    "sections": [
      {
        "section": "3.4 Aprobación de reembolsos",
        "text": "Los responsables de área podrán aprobar solicitudes de reembolso de hasta 1000 EUR. Las solicitudes superiores a 1000 EUR deberán enviarse a Finanzas."
      },
      {
        "section": "6.2 Archivo de comprobantes",
        "text": "El equipo de operaciones conservará los comprobantes de reembolso durante cinco años en el repositorio financiero."
      }
    ]
  },
  {
    "document_id": "POL-SEC-010",
    "title": "Política de Seguridad de la Información",
    "document_type": "política",
    "version": "4.2",
    "effective_date": "2024-02-01",
    "status": "vigente",
    "source": "Oficina de Seguridad de la Información",
    "authority_level": 3,
    "sections": [
      {
        "section": "6.3 Baja de usuarios",
        "text": "Los accesos de una persona desvinculada deberán revocarse en un plazo máximo de 24 horas desde la notificación formal de baja."
      },
      {
        "section": "6.4 Acceso remoto",
        "text": "Todo acceso remoto corporativo deberá utilizar autenticación multifactor. No se permiten excepciones permanentes sin aprobación del CISO."
      }
    ]
  },
  {
    "document_id": "PROC-IT-022",
    "title": "Procedimiento de Gestión de Identidades",
    "document_type": "procedimiento",
    "version": "2.1",
    "effective_date": "2023-09-01",
    "status": "vigente",
    "source": "Infraestructura TI",
    "authority_level": 2,
    "sections": [
      {
        "section": "4.5 Revocación de accesos",
        "text": "El administrador de sistemas revocará las cuentas de usuarios desvinculados en un plazo de hasta 48 horas desde la recepción del ticket de recursos humanos."
      },
      {
        "section": "4.8 Acceso VPN",
        "text": "Las conexiones VPN corporativas requieren autenticación multifactor y credenciales personales no compartidas."
      }
    ]
  },
  {
    "document_id": "FIC-OPS-003",
    "title": "Ficha Técnica de Clasificación de Incidentes",
    "document_type": "ficha_técnica",
    "version": "1.4",
    "effective_date": "2024-03-01",
    "status": "vigente",
    "source": "Centro de Operaciones de Seguridad",
    "authority_level": 2,
    "sections": [
      {
        "section": "2.1 Definición de incidente",
        "text": "Un incidente de seguridad es cualquier evento observado que pueda afectar la confidencialidad, integridad o disponibilidad de información corporativa."
      }
    ]
  },
  {
    "document_id": "COM-SEC-018",
    "title": "Comunicación Interna sobre Reporte de Eventos",
    "document_type": "comunicación_interna",
    "version": "1.0",
    "effective_date": "2024-03-10",
    "status": "vigente",
    "source": "Mesa de Ayuda",
    "authority_level": 1,
    "sections": [
      {
        "section": "Mensaje operativo",
        "text": "Solo deben registrarse como incidentes de seguridad los eventos confirmados por un analista de nivel 2. Los eventos no confirmados se tratarán como consultas."
      }
    ]
  },
  {
    "document_id": "POL-FIN-001",
    "title": "Política Corporativa de Gastos",
    "document_type": "política",
    "version": "2.0",
    "effective_date": "2022-01-01",
    "status": "sustituido",
    "source": "Dirección Financiera",
    "authority_level": 3,
    "sections": [
      {
        "section": "4.2 Límites de aprobación",
        "text": "Toda solicitud de gasto superior a 1000 EUR requiere aprobación previa de la Dirección Financiera."
      }
    ]
  }
]
EOF
```

2. Inspecciona el corpus:

```bash
python - <<'PY'
import json
from pathlib import Path

docs = json.loads(Path("data/raw/documents.json").read_text())
print(f"Documentos cargados: {len(docs)}")
for doc in docs:
    print(
        f"{doc['document_id']} v{doc['version']} | "
        f"{doc['status']} | {doc['document_type']}"
    )
PY
```

### Salida esperada

Debes obtener siete documentos, incluyendo una versión sustituida de `POL-FIN-001`.

Los conflictos deliberados son:

| Tema | Fuente 1 | Fuente 2 | Tipo esperado |
|---|---|---|---|
| Límite monetario | Política: 500 EUR | Procedimiento: 1000 EUR | `limite_monetario` |
| Retención | Política: 7 años | Procedimiento: 5 años | `plazo_retencion` |
| Baja de usuarios | Política: 24 horas | Procedimiento: 48 horas | `plazo_operativo` |
| Definición de incidente | Ficha: evento potencial | Comunicación: evento confirmado | `definicion` |
| VPN con MFA | Política y procedimiento exigen MFA | No existe contradicción | `ninguna` |

### Verificación

Confirma que la versión `2.0` de `POL-FIN-001` tiene estado `sustituido`:

```bash
grep -A8 '"version": "2.0"' data/raw/documents.json
```

La recuperación puede incluir documentos sustituidos como evidencia histórica, pero el auditor debe considerar el estado y la autoridad para evitar tratarlos como norma vigente.

---

## Paso 3. Implementar normalización, fragmentación e indexación híbrida

**Objetivo:** Normalizar documentos, generar fragmentos con metadatos completos, crear embeddings en Qdrant y persistir un índice BM25 local.

### Instrucciones

1. Crea el archivo `src/auditor_rag.py`:

```bash
cat > src/auditor_rag.py <<'PY'
import json
import os
import pickle
import re
import sys
import uuid
from pathlib import Path
from typing import Literal

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data/raw/documents.json"
PROCESSED_PATH = ROOT / "data/processed/chunks.json"
BM25_PATH = ROOT / "data/processed/bm25_index.pkl"
EVAL_PATH = ROOT / "data/processed/evaluation_cases.json"
OUTPUTS = ROOT / "outputs"
COLLECTION = "audit_documents"
EMBEDDING_MODEL = "text-embedding-3-small"
AUDIT_MODEL = "gpt-4o-mini-2024-07-18"

load_dotenv(ROOT / ".env")
openai_client = OpenAI()
qdrant = QdrantClient(url="http://localhost:6333")


class EvidenceQuote(BaseModel):
    chunk_id: str
    document_id: str
    version: str
    section: str
    quote: str


class AuditReport(BaseModel):
    claim: str
    inconsistency_detected: bool
    inconsistency_type: Literal[
        "limite_monetario",
        "plazo_retencion",
        "plazo_operativo",
        "rol_aprobador",
        "definicion",
        "ninguna",
        "evidencia_insuficiente"
    ]
    severity: Literal["baja", "media", "alta", "critica", "no_aplica"]
    conflicting_sources: list[str] = Field(default_factory=list)
    evidence_quotes: list[EvidenceQuote] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    recommended_action: str


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower(), flags=re.UNICODE)


def chunk_text(text: str, size: int = 90, overlap: int = 20) -> list[str]:
    words = text.split()
    if len(words) <= size:
        return [text]
    chunks = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
        start += size - overlap
    return chunks


def normalize_and_chunk() -> list[dict]:
    documents = json.loads(RAW_PATH.read_text(encoding="utf-8"))
    chunks = []

    for document in documents:
        for section_data in document["sections"]:
            for index, text in enumerate(chunk_text(section_data["text"])):
                chunk_id = (
                    f"{document['document_id']}_v{document['version']}_"
                    f"{len(chunks):03d}"
                )
                chunks.append({
                    "chunk_id": chunk_id,
                    "document_id": document["document_id"],
                    "title": document["title"],
                    "document_type": document["document_type"],
                    "version": document["version"],
                    "effective_date": document["effective_date"],
                    "status": document["status"],
                    "source": document["source"],
                    "authority_level": document["authority_level"],
                    "section": section_data["section"],
                    "chunk_index": index,
                    "text": text
                })

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_PATH.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    return chunks


def embed(texts: list[str]) -> list[list[float]]:
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    return [item.embedding for item in response.data]


def build_indexes():
    chunks = normalize_and_chunk()
    embeddings = embed([chunk["text"] for chunk in chunks])

    qdrant.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )

    points = []
    for chunk, vector in zip(chunks, embeddings):
        points.append(
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk["chunk_id"])),
                vector=vector,
                payload=chunk
            )
        )

    qdrant.upsert(collection_name=COLLECTION, points=points, wait=True)

    tokenized_corpus = [tokenize(chunk["text"]) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    with BM25_PATH.open("wb") as file:
        pickle.dump({"bm25": bm25, "chunks": chunks}, file)

    print(f"Fragmentos normalizados: {len(chunks)}")
    print(f"Colección creada: {COLLECTION}")
    print(f"Índice BM25 guardado en: {BM25_PATH}")


def minmax(scores: list[float]) -> list[float]:
    if not scores:
        return []
    low, high = min(scores), max(scores)
    if high == low:
        return [1.0 for _ in scores]
    return [(score - low) / (high - low) for score in scores]


def retrieve_hybrid(query: str, limit: int = 6) -> list[dict]:
    with BM25_PATH.open("rb") as file:
        data = pickle.load(file)

    bm25 = data["bm25"]
    chunks = data["chunks"]

    query_vector = embed([query])[0]
    vector_hits = qdrant.search(
        collection_name=COLLECTION,
        query_vector=query_vector,
        limit=12,
        with_payload=True
    )

    vector_scores = minmax([hit.score for hit in vector_hits])
    candidates = {}

    for hit, score in zip(vector_hits, vector_scores):
        payload = dict(hit.payload)
        payload["vector_score"] = score
        candidates[payload["chunk_id"]] = payload

    lexical_raw = bm25.get_scores(tokenize(query))
    lexical_scores = minmax(list(lexical_raw))
    lexical_order = sorted(
        range(len(chunks)),
        key=lambda index: lexical_scores[index],
        reverse=True
    )[:12]

    for index in lexical_order:
        payload = dict(chunks[index])
        payload["bm25_score"] = lexical_scores[index]
        candidates.setdefault(payload["chunk_id"], payload)
        candidates[payload["chunk_id"]]["bm25_score"] = lexical_scores[index]

    ranked = []
    for candidate in candidates.values():
        vector_score = candidate.get("vector_score", 0.0)
        bm25_score = candidate.get("bm25_score", 0.0)
        authority_score = candidate["authority_level"] / 3.0

        candidate["hybrid_score"] = (
            0.65 * vector_score +
            0.35 * bm25_score
        )
        candidate["authority_score"] = authority_score
        ranked.append(candidate)

    ranked.sort(
        key=lambda item: item["hybrid_score"] + 0.10 * item["authority_score"],
        reverse=True
    )

    selected = []
    seen_sources = set()
    seen_documents = set()

    while ranked and len(selected) < limit:
        best = None
        best_score = -1.0

        for candidate in ranked:
            diversity_bonus = 0.0
            if candidate["source"] not in seen_sources:
                diversity_bonus += 0.08
            if candidate["document_id"] not in seen_documents:
                diversity_bonus += 0.07

            score = (
                candidate["hybrid_score"] +
                0.10 * candidate["authority_score"] +
                diversity_bonus
            )

            if score > best_score:
                best = candidate
                best_score = score

        best["rerank_score"] = round(best_score, 4)
        selected.append(best)
        seen_sources.add(best["source"])
        seen_documents.add(best["document_id"])
        ranked.remove(best)

    return selected


def audit_claim(claim: str) -> AuditReport:
    evidence = retrieve_hybrid(claim, limit=6)

    evidence_text = "\n\n".join(
        [
            (
                f"[{item['chunk_id']}]\n"
                f"Documento: {item['document_id']} v{item['version']}\n"
                f"Estado: {item['status']}; Autoridad: {item['authority_level']}/3\n"
                f"Fuente: {item['source']}; Sección: {item['section']}\n"
                f"Texto: {item['text']}"
            )
            for item in evidence
        ]
    )

    prompt = f"""
Eres un auditor documental. Analiza exclusivamente la evidencia suministrada.

Afirmación objetivo:
{claim}

Reglas obligatorias:
1. Declara contradicción solo si dos o más fuentes expresan reglas incompatibles.
2. Distingue una contradicción de información complementaria.
3. Considera estado, versión y nivel de autoridad; una fuente sustituida puede aportar contexto, pero no debe prevalecer sobre una vigente.
4. Si no hay evidencia suficiente, usa inconsistency_type="evidencia_insuficiente".
5. Si no existe contradicción, usa inconsistency_type="ninguna" y severity="no_aplica".
6. Cita literalmente fragmentos breves y usa solo chunk_id presentes en la evidencia.
7. No inventes documentos, citas, fechas ni reglas.
8. Responde exclusivamente con un objeto JSON válido.

Evidencia recuperada:
{evidence_text}
"""

    response = openai_client.chat.completions.create(
        model=AUDIT_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "Tu salida debe ser JSON válido y conforme a las reglas proporcionadas."
            },
            {"role": "user", "content": prompt}
        ]
    )

    report = AuditReport.model_validate_json(response.choices[0].message.content)
    report_path = OUTPUTS / "latest_audit.json"
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        report.model_dump_json(indent=2),
        encoding="utf-8"
    )

    trace_path = OUTPUTS / "latest_retrieval_trace.json"
    trace_path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    return report


def create_evaluation_cases():
    cases = [
        {
            "case_id": "EVAL-01",
            "claim": "Los responsables de área pueden aprobar gastos de hasta 1000 EUR.",
            "expected_inconsistency": True,
            "expected_type": "limite_monetario"
        },
        {
            "case_id": "EVAL-02",
            "claim": "Los comprobantes financieros deben conservarse durante cinco años.",
            "expected_inconsistency": True,
            "expected_type": "plazo_retencion"
        },
        {
            "case_id": "EVAL-03",
            "claim": "La revocación de acceso de usuarios desvinculados debe completarse en 24 horas.",
            "expected_inconsistency": True,
            "expected_type": "plazo_operativo"
        },
        {
            "case_id": "EVAL-04",
            "claim": "Un incidente requiere confirmación previa de un analista de nivel 2.",
            "expected_inconsistency": True,
            "expected_type": "definicion"
        },
        {
            "case_id": "EVAL-05",
            "claim": "El acceso VPN corporativo requiere autenticación multifactor.",
            "expected_inconsistency": False,
            "expected_type": "ninguna"
        }
    ]

    EVAL_PATH.write_text(
        json.dumps(cases, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"Casos de evaluación creados: {len(cases)}")


def evaluate():
    cases = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    rows = []

    for case in cases:
        report = audit_claim(case["claim"])
        rows.append({
            "case_id": case["case_id"],
            "claim": case["claim"],
            "expected_inconsistency": case["expected_inconsistency"],
            "predicted_inconsistency": report.inconsistency_detected,
            "expected_type": case["expected_type"],
            "predicted_type": report.inconsistency_type,
            "confidence": report.confidence,
            "severity": report.severity,
            "rationale": report.rationale
        })

    frame = pd.DataFrame(rows)
    y_true = frame["expected_inconsistency"].astype(int)
    y_pred = frame["predicted_inconsistency"].astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0
    )

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    metrics = {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "confusion_matrix_labels": ["sin_inconsistencia", "inconsistencia"],
        "confusion_matrix": matrix,
        "total_cases": len(rows)
    }

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUTS / "evaluation_results.csv", index=False)
    (OUTPUTS / "evaluation_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    failures = frame[
        frame["expected_inconsistency"] != frame["predicted_inconsistency"]
    ]

    report_lines = [
        "# Reporte de evaluación",
        "",
        f"- Precisión: **{metrics['precision']}**",
        f"- Recall: **{metrics['recall']}**",
        f"- F1: **{metrics['f1']}**",
        f"- Matriz de confusión `[TN, FP], [FN, TP]`: `{matrix}`",
        "",
        "## Casos clasificados incorrectamente"
    ]

    if failures.empty:
        report_lines.append(
            "No hubo errores de clasificación binaria en esta ejecución. "
            "Aun así, deben revisarse los siguientes riesgos de fallo."
        )
    else:
        for _, row in failures.iterrows():
            report_lines.append(
                f"- {row['case_id']}: esperado={row['expected_inconsistency']}, "
                f"predicho={row['predicted_inconsistency']}. "
                f"Razonamiento: {row['rationale']}"
            )

    report_lines.extend([
        "",
        "## Tres casos de fallo y acciones de mejora",
        "",
        "1. **Fallo de recuperación léxica:** una consulta puede recuperar solo "
        "el procedimiento con el número exacto `1000 EUR` y omitir la política "
        "con `500 EUR`. Mejora: ampliar `limit`, ajustar pesos híbridos y añadir "
        "expansión de consulta con términos como `límite`, `aprobación` y `gasto`.",
        "",
        "2. **Fallo de razonamiento sobre precedencia:** el modelo puede tratar "
        "una versión sustituida como si fuera vigente. Mejora: reforzar el prompt, "
        "aplicar filtros por estado cuando la consulta solicite la norma vigente "
        "y aumentar el peso de autoridad para documentos vigentes.",
        "",
        "3. **Falso positivo por complementariedad:** dos documentos pueden usar "
        "redacciones distintas sobre MFA sin ser contradictorios. Mejora: exigir "
        "que el modelo identifique valores incompatibles, sujetos distintos o "
        "condiciones mutuamente excluyentes antes de marcar una contradicción."
    ])

    (OUTPUTS / "evaluation_report.md").write_text(
        "\n".join(report_lines),
        encoding="utf-8"
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def main():
    if len(sys.argv) < 2:
        raise SystemExit(
            "Uso: python src/auditor_rag.py "
            "[build|create-eval|audit|evaluate] [afirmación]"
        )

    command = sys.argv[1]

    if command == "build":
        build_indexes()
    elif command == "create-eval":
        create_evaluation_cases()
    elif command == "audit":
        if len(sys.argv) < 3:
            raise SystemExit("Indica una afirmación para auditar.")
        report = audit_claim(" ".join(sys.argv[2:]))
        print(report.model_dump_json(indent=2))
    elif command == "evaluate":
        evaluate()
    else:
        raise SystemExit(f"Comando no reconocido: {command}")


if __name__ == "__main__":
    main()
PY
```

2. Construye el corpus normalizado, el índice vectorial y el índice BM25:

```bash
python src/auditor_rag.py build
```

3. Inspecciona los fragmentos normalizados:

```bash
python - <<'PY'
import json
from pathlib import Path

chunks = json.loads(Path("data/processed/chunks.json").read_text())
print(f"Total de fragmentos: {len(chunks)}")
print(json.dumps(chunks[0], ensure_ascii=False, indent=2))
PY
```

4. Comprueba la colección Qdrant:

```bash
curl http://localhost:6333/collections/audit_documents
```

### Salida esperada

El comando de construcción debe informar una salida similar a:

```text
Fragmentos normalizados: 11
Colección creada: audit_documents
Índice BM25 guardado en: .../data/processed/bm25_index.pkl
```

La respuesta de Qdrant debe indicar que la colección `audit_documents` contiene puntos y tiene vectores de tamaño `1536`.

### Verificación

Confirma que se han creado los artefactos persistentes:

```bash
find data/processed -maxdepth 1 -type f -printf '%f\n'
```

Debes ver, al menos:

```text
chunks.json
bm25_index.pkl
```

---

## Paso 4. Crear el conjunto de evaluación etiquetado

**Objetivo:** Definir afirmaciones de auditoría con etiquetas esperadas para medir la detección de contradicciones.

### Instrucciones

1. Ejecuta el generador de casos de evaluación:

```bash
python src/auditor_rag.py create-eval
```

2. Revisa los casos creados:

```bash
cat data/processed/evaluation_cases.json
```

3. Identifica los componentes de cada etiqueta:

- `claim`: afirmación o regla a auditar.
- `expected_inconsistency`: etiqueta binaria esperada.
- `expected_type`: clasificación esperada de la inconsistencia.

### Salida esperada

Debes obtener cinco casos de evaluación:

- Cuatro casos con contradicción esperada.
- Un caso sin contradicción: autenticación multifactor para VPN.

### Verificación

Ejecuta:

```bash
python - <<'PY'
import json

with open("data/processed/evaluation_cases.json", encoding="utf-8") as file:
    cases = json.load(file)

positivos = sum(case["expected_inconsistency"] for case in cases)
negativos = len(cases) - positivos

print(f"Casos positivos: {positivos}")
print(f"Casos negativos: {negativos}")
assert positivos == 4
assert negativos == 1
PY
```

---

## Paso 5. Ejecutar una auditoría individual con trazabilidad

**Objetivo:** Validar que el sistema recupere evidencia desde fuentes distintas y genere un objeto JSON estructurado.

### Instrucciones

1. Ejecuta una auditoría sobre el límite de aprobación:

```bash
python src/auditor_rag.py audit \
  "Los responsables de área pueden aprobar gastos de hasta 1000 EUR."
```

2. Revisa el informe generado:

```bash
cat outputs/latest_audit.json
```

3. Revisa la traza de recuperación:

```bash
cat outputs/latest_retrieval_trace.json
```

4. Comprueba que la evidencia contiene metadatos de trazabilidad:

```bash
python - <<'PY'
import json

with open("outputs/latest_retrieval_trace.json", encoding="utf-8") as file:
    evidence = json.load(file)

for item in evidence:
    print(
        f"{item['chunk_id']} | {item['document_id']} v{item['version']} | "
        f"{item['status']} | autoridad={item['authority_level']} | "
        f"score={item['rerank_score']}"
    )
PY
```

### Salida esperada

El informe debe ser JSON válido y contener, como mínimo, los campos:

```json
{
  "claim": "Los responsables de área pueden aprobar gastos de hasta 1000 EUR.",
  "inconsistency_detected": true,
  "inconsistency_type": "limite_monetario",
  "severity": "alta",
  "conflicting_sources": [
    "POL-FIN-001 v3.0",
    "PROC-FIN-014 v2.1"
  ],
  "evidence_quotes": [],
  "confidence": 0.0,
  "rationale": "",
  "recommended_action": ""
}
```

Los valores exactos de `confidence`, severidad y redacción pueden variar, pero el informe debe citar evidencia recuperada y no debe inventar identificadores de documentos.

### Verificación

Valida el JSON con Pydantic reutilizando el esquema del proyecto:

```bash
python - <<'PY'
import json
import sys

sys.path.append("src")
from auditor_rag import AuditReport

with open("outputs/latest_audit.json", encoding="utf-8") as file:
    report = AuditReport.model_validate(json.load(file))

print("Informe JSON válido.")
print("Inconsistencia detectada:", report.inconsistency_detected)
print("Tipo:", report.inconsistency_type)
print("Confianza:", report.confidence)
PY
```

La salida debe indicar:

```text
Informe JSON válido.
```

---

## Paso 6. Ejecutar la evaluación del auditor

**Objetivo:** Calcular métricas de detección de inconsistencias y producir artefactos reutilizables para análisis posterior.

### Instrucciones

1. Ejecuta la evaluación completa:

```bash
python src/auditor_rag.py evaluate
```

2. Consulta las métricas:

```bash
cat outputs/evaluation_metrics.json
```

3. Consulta los resultados por caso:

```bash
column -s, -t < outputs/evaluation_results.csv
```

Si el comando `column` no está disponible, usa:

```bash
cat outputs/evaluation_results.csv
```

4. Lee el reporte de evaluación:

```bash
cat outputs/evaluation_report.md
```

### Salida esperada

Se generarán los siguientes archivos:

```text
outputs/evaluation_metrics.json
outputs/evaluation_results.csv
outputs/evaluation_report.md
```

El archivo de métricas tendrá una estructura similar a:

```json
{
  "precision": 1.0,
  "recall": 0.75,
  "f1": 0.8571,
  "confusion_matrix_labels": [
    "sin_inconsistencia",
    "inconsistencia"
  ],
  "confusion_matrix": [
    [1, 0],
    [1, 3]
  ],
  "total_cases": 5
}
```

Los valores concretos pueden variar por la naturaleza probabilística del LLM y por cambios en el servicio de modelos.

### Verificación

Comprueba que las métricas se encuentran entre `0.0` y `1.0`:

```bash
python - <<'PY'
import json

with open("outputs/evaluation_metrics.json", encoding="utf-8") as file:
    metrics = json.load(file)

for metric in ["precision", "recall", "f1"]:
    value = metrics[metric]
    assert 0.0 <= value <= 1.0, f"Valor inválido para {metric}: {value}"
    print(f"{metric}: {value}")

assert metrics["total_cases"] == 5
print("Métricas verificadas correctamente.")
PY
```

---

## Validación y pruebas

La validación final debe confirmar cuatro propiedades: infraestructura disponible, trazabilidad de evidencia, salida estructurada y métricas calculadas.

### Prueba 1. Disponibilidad de Qdrant

```bash
curl -s http://localhost:6333/collections/audit_documents | python -m json.tool
```

Criterio de aceptación:

- La colección `audit_documents` existe.
- La configuración vectorial tiene tamaño `1536`.
- El número de puntos es mayor que cero.

### Prueba 2. Persistencia de artefactos

```bash
test -f data/processed/chunks.json && echo "Corpus normalizado: OK"
test -f data/processed/bm25_index.pkl && echo "BM25 serializado: OK"
test -f data/processed/evaluation_cases.json && echo "Casos etiquetados: OK"
test -f outputs/evaluation_metrics.json && echo "Métricas: OK"
test -f outputs/evaluation_report.md && echo "Reporte: OK"
```

Criterio de aceptación: todos los mensajes deben terminar con `OK`.

### Prueba 3. Trazabilidad de una decisión

Ejecuta nuevamente una auditoría de plazos:

```bash
python src/auditor_rag.py audit \
  "Los comprobantes y registros financieros deben conservarse durante cinco años."
```

Revisa que:

- El resultado incluya `POL-FIN-001 v3.0`.
- El resultado incluya `PROC-FIN-014 v2.1`.
- Las citas tengan `chunk_id`, `document_id`, `version`, `section` y `quote`.
- El modelo marque una inconsistencia de tipo `plazo_retencion`, si recuperó evidencia suficiente.
- El informe indique una recomendación de revisión humana o actualización documental.

### Prueba 4. Caso negativo de complementariedad

Ejecuta:

```bash
python src/auditor_rag.py audit \
  "El acceso VPN corporativo requiere autenticación multifactor."
```

Criterio de aceptación:

- `inconsistency_detected` debe ser `false`.
- `inconsistency_type` debe ser `ninguna`.
- `severity` debe ser `no_aplica`.
- La justificación debe reconocer que la política y el procedimiento son complementarios.

### Interpretación de resultados

| Situación observada | Interpretación |
|---|---|
| Alta precisión y bajo recall | El sistema evita falsos positivos, pero omite contradicciones. Revisa recuperación y `limit`. |
| Bajo precisión y alto recall | El sistema detecta muchos conflictos, pero marca información complementaria como contradictoria. Mejora el prompt y las reglas de clasificación. |
| Baja precisión y bajo recall | Existen problemas de recuperación, fragmentación, etiquetas o razonamiento. Revisa la traza de evidencia por caso. |
| F1 alto | El conjunto de prueba está razonablemente equilibrado para este corpus controlado; no implica validez general en producción. |

---

## Resolución de problemas

### Problema 1: Qdrant no inicia o el puerto 6333 está ocupado

**Síntoma:**

```text
Bind for 0.0.0.0:6333 failed: port is already allocated
```

o bien:

```text
Connection refused: http://localhost:6333
```

**Causa:** Otro proceso o contenedor utiliza el puerto REST `6333` o el servicio `rag-auditor-qdrant` no se inició correctamente.

**Solución:**

```bash
docker compose down
docker ps --format "table {{.Names}}\t{{.Ports}}"
```

Identifica el contenedor que ocupa los puertos `6333` o `6334`, detenlo si no es necesario y vuelve a iniciar el laboratorio:

```bash
docker compose up -d
docker compose ps
curl http://localhost:6333/collections
```

No cambies los puertos del laboratorio, porque `6333` y `6334` son requisitos de la práctica.

### Problema 2: Error de autenticación o cuota al llamar a OpenAI

**Síntoma:**

```text
openai.AuthenticationError
```

o:

```text
openai.RateLimitError
```

**Causa:** La variable `OPENAI_API_KEY` no está definida, contiene una clave inválida, no tiene permisos para el modelo solicitado o la cuenta alcanzó un límite de cuota o velocidad.

**Solución:**

Comprueba que el archivo `.env` existe y que la clave no contiene el texto de ejemplo:

```bash
grep OPENAI_API_KEY .env
```

Carga la variable manualmente en la sesión actual:

```bash
export $(grep -v '^#' .env | xargs)
```

Comprueba que Python puede verla sin mostrar el secreto:

```bash
python - <<'PY'
import os
key = os.getenv("OPENAI_API_KEY", "")
print("Clave configurada:", len(key) > 20)
PY
```

Si el error es de cuota o límite de velocidad, espera unos minutos, reduce ejecuciones repetidas de `evaluate` o revisa la cuota y permisos del proyecto de OpenAI.

---

## Limpieza

Si deseas detener Qdrant sin borrar los índices persistentes:

```bash
docker compose stop
```

Para reiniciar el entorno conservando los datos:

```bash
docker compose start
```

Para eliminar el contenedor y la red Docker, manteniendo los archivos del proyecto:

```bash
docker compose down
```

Para eliminar completamente los índices, resultados y almacenamiento persistente del laboratorio:

```bash
docker compose down -v
rm -rf data/processed/*
rm -rf data/qdrant_storage/*
rm -rf outputs/*
```

Para salir del entorno virtual:

```bash
deactivate
```

> No elimines `data/raw/documents.json`, `src/auditor_rag.py`, `docker-compose.yml` ni `data/processed/evaluation_cases.json` si deseas reutilizar el proyecto en laboratorios posteriores.

## Resumen

En esta práctica construiste un auditor RAG híbrido orientado a consistencia documental. El flujo implementado incluye:

1. Un corpus versionado con políticas, procedimientos, fichas técnicas y comunicaciones internas.
2. Fragmentación con metadatos de trazabilidad: documento, versión, estado, sección, fuente y autoridad.
3. Recuperación vectorial mediante Qdrant y embeddings `text-embedding-3-small`.
4. Recuperación léxica local con BM25 sobre los mismos fragmentos.
5. Fusión híbrida ponderada y reranking que favorece relevancia, diversidad de fuentes y autoridad.
6. Auditoría con `gpt-4o-mini-2024-07-18` y salida JSON validada mediante Pydantic.
7. Evaluación con precisión, recall, F1, matriz de confusión y análisis de casos de fallo.

Los artefactos persistentes generados son reutilizables:

| Artefacto | Ubicación |
|---|---|
| Corpus original | `data/raw/documents.json` |
| Corpus normalizado | `data/processed/chunks.json` |
| Índice BM25 | `data/processed/bm25_index.pkl` |
| Colección vectorial | `audit_documents` en Qdrant |
| Casos etiquetados | `data/processed/evaluation_cases.json` |
| Informe de auditoría individual | `outputs/latest_audit.json` |
| Traza de recuperación | `outputs/latest_retrieval_trace.json` |
| Resultados de evaluación | `outputs/evaluation_results.csv` |
| Métricas | `outputs/evaluation_metrics.json` |
| Reporte de evaluación | `outputs/evaluation_report.md` |

Como mejora posterior, puedes incorporar filtros Qdrant por estado documental, extracción de entidades normativas, rerankers semánticos dedicados, evaluación de relevancia de recuperación por fragmento y revisión humana obligatoria para hallazgos de severidad alta o crítica.
