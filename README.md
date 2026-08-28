# Taller Práctico Generación Aumentada de Recuperación (RAG)

Los participantes desarrollarán un sistema RAG orientado a la auditoría documental. El sistema analizará versiones sintéticas de políticas, procedimientos y manuales para detectar contradicciones, requisitos desactualizados, información duplicada y diferencias entre documentos.

La práctica no consiste en crear otro asistente de preguntas y respuestas. Su propósito es utilizar recuperación híbrida, generación fundamentada y evaluación para producir un reporte de hallazgos con trazabilidad hacia los fragmentos que sustentan cada observación

## Estructura

- `CapituloXX/README.md`: guía de laboratorio por capítulo.

## Lista de laboratorios

### Capítulo 1

- [Construcción de un auditor RAG para detectar inconsistencias entre documentos](Capitulo01/README.md#construcción-de-un-auditor-rag-para-detectar-inconsistencias-entre-documentos)
  - Descripción: Evaluar de forma guiada un sistema RAG para comparar documentos, detectar posibles inconsistencias y generar hallazgos sustentados en las fuentes recuperadas.

Los participantes trabajarán en Visual Studio Code con un notebook de Jupyter previamente preparado y documentos sintéticos. A partir de diferentes preguntas, ejecutarán un proceso de recuperación que identificará automáticamente los fragmentos más relevantes de los documentos. El notebook generará un bloque de contexto listo para copiar y pegar en Copilot Chat Standard junto con la pregunta y las instrucciones proporcionadas.

Los participantes utilizarán ese contexto para generar respuestas, comparar información entre documentos e identificar posibles inconsistencias. Durante el ejercicio modificarán instrucciones sencillas del prompt para mejorar la respuesta y probarán casos donde la información sea contradictoria, incompleta o no se encuentre disponible, verificando que los hallazgos estén respaldados por las fuentes proporcionadas.
  - Duración estimada: 105 min

## Flujo de colaboración

- Trabajar en `changes_course`.
- Crear Pull Request hacia `main`.
- Merge por `Squash and merge`.
