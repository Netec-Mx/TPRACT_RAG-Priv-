# Taller Práctico Generación Aumentada de Recuperación (RAG)

Este taller práctico aplica los conceptos de **LLM y generación de texto** y **Sistemas RAG en producción** mediante una auditoría de consistencia documental. Los participantes analizarán un expediente de políticas, procedimientos y manuales para recuperar evidencia relevante, comparar requisitos, revisar versiones y elaborar hallazgos con referencias verificables.

En Visual Studio Code trabajarán con un notebook Jupyter que utiliza **BM25 + LSA** para recuperar fragmentos y preparar contextos fundamentados. Durante la práctica evaluarán las citas, la cobertura de evidencia, la vigencia documental, la latencia y aspectos de seguridad. Utilizarán **Microsoft 365 Copilot Chat Básico** para contrastar manualmente respuestas a partir de los fragmentos seleccionados y finalizarán con un **dictamen de auditoría trazable**.

## Estructura

- `Capitulo01/README.md`: guía práctica del capítulo, con los retos y sus tiempos.
- `Capitulo01/materiales/`: notebook, código Python, documentos y archivos de apoyo para la auditoría.
- `preparacion_entorno.md`: requisitos y preparación de Python, Visual Studio Code y Jupyter.

## Lista de laboratorios

### Capítulo 1

- [Auditoría de consistencia y trazabilidad documental mediante RAG](Capitulo01/README.md#auditoría-de-consistencia-y-trazabilidad-documental-mediante-rag)
  - **Descripción:** Desarrollar una auditoría guiada sobre ocho documentos de operaciones digitales y siete casos de revisión. El participante recuperará fragmentos relevantes, comprobará posibles contradicciones, coincidencias y desactualizaciones, preparará prompts aumentados para su contraste en Copilot Chat Básico y evaluará la fundamentación de afirmaciones y citas. También experimentará con reconsultas, métricas de recuperación, observabilidad y seguridad. El notebook registrará las evidencias y generará un dictamen documentado para revisión humana.
  - **Duración estimada:** 105 min.

## Flujo de colaboración

- Trabajar en `changes_course`.
- Crear Pull Request hacia `main`.
- Merge por `Squash and merge`.
