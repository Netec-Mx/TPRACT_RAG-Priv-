# Práctica 1:  Construcción de un auditor RAG para detectar inconsistencias entre documentos



**Capítulo 1 · Práctica guiada: 105 minutos · Módulos 4 y 5 de Coursera**

## Descripción general

La Oficina de Operación Digital necesita revisar reglas de seguridad, acceso privilegiado y atención de incidentes. Cuenta con políticas, estándares, manuales, guías y registros que pueden contener versiones diferentes, instrucciones coincidentes o requisitos contradictorios. El objetivo es **localizar las fuentes, comprobar su vigencia y elaborar un dictamen sustentado en párrafos verificables**.

El participante seguirá un mismo caso de principio a fin: comenzará investigando una discrepancia sobre la duración de sesiones (**A01**), ampliará la auditoría a otros controles y emitirá un dictamen integrado (**A07**).

El laboratorio aplica los contenidos de **LLM y generación de texto (módulo 4)** en la preparación de prompts fundamentados y la revisión de respuestas; y los de **Sistemas RAG en producción (módulo 5)** en la evaluación de evidencia, observabilidad, eficiencia y seguridad.

## Objetivo

Aplicar y evaluar un flujo RAG de auditoría documental que recupere evidencia relevante, compare documentos, identifique contradicciones, coincidencias y versiones aplicables, y produzca hallazgos con referencias rastreables y criterios de revisión humana.

## Antes de comenzar

El laboratorio utiliza **Python 3.11/3.12**, **Visual Studio Code**, **Jupyter** y las bibliotecas incluidas en [requirements.txt](https://netec-mx.github.io/TPRACT_RAG-Priv-/Capitulo01/materiales/vistas/requirements.txt.html). Se utiliza **Microsoft 365 Copilot Chat Básico** para redactar y contrastar manualmente respuestas basadas en los fragmentos recuperados.

1. En VS Code abra **File > Open Folder** y seleccione `Capitulo01/materiales`.
2. Consulte la [vista del notebook](https://netec-mx.github.io/TPRACT_RAG-Priv-/Capitulo01/materiales/vistas/01_auditoria_rag_produccion.ipynb.html), use **Descargar notebook para VS Code** y abra el archivo descargado en VS Code dentro de `Capitulo01/materiales`.
3. Seleccione el kernel Python del curso.
4. Ejecute las celdas en orden con **Shift+Enter**. Modifique los parámetros identificados con **MODIFIQUE AQUÍ** y complete las interpretaciones dentro del notebook.
5. Guarde su trabajo con **Ctrl+S**.

**Entregables:** notebook diligenciado, once archivos de evidencia generados en `materiales/resultados/` y dictamen preparado para revisión.

## Conozca el expediente

El listado de documentos del expediente contiene ocho documentos con vistas de lectura en GitHub Pages. Cada archivo conserva código, versión y apartados citables.

| Documento | ¿Qué información aporta al análisis? |
|---|---|
| [POL-SEG-014 v3.2](https://netec-mx.github.io/TPRACT_RAG-Priv-/Capitulo01/materiales/vistas/POL-SEG-014_v3.2.md.html) | Política de autenticación y aprobación de excepciones |
| [EST-SEG-009 v2.1](https://netec-mx.github.io/TPRACT_RAG-Priv-/Capitulo01/materiales/vistas/EST-SEG-009_v2.1.md.html) | Regla de cierre de sesiones privilegiadas por inactividad |
| [MAN-OPS-011 v2.3](https://netec-mx.github.io/TPRACT_RAG-Priv-/Capitulo01/materiales/vistas/MAN-OPS-011_v2.3.md.html) | Configuración descrita para la consola operativa |
| [PRC-INC-005 v3.1](https://netec-mx.github.io/TPRACT_RAG-Priv-/Capitulo01/materiales/vistas/PRC-INC-005_v3.1.md.html) | Procedimiento y tiempos de escalamiento de incidentes |
| [GUI-INC-004 v1.8](https://netec-mx.github.io/TPRACT_RAG-Priv-/Capitulo01/materiales/vistas/GUI-INC-004_v1.8.md.html) | Guía de escalamiento aplicable antes de septiembre de 2026 |
| [GUI-INC-004 v2.0](https://netec-mx.github.io/TPRACT_RAG-Priv-/Capitulo01/materiales/vistas/GUI-INC-004_v2.0.md.html) | Guía de escalamiento aplicable desde septiembre de 2026 |
| [REG-EXC-006 v1.0](https://netec-mx.github.io/TPRACT_RAG-Priv-/Capitulo01/materiales/vistas/REG-EXC-006_v1.0.md.html) | Registro documental de solicitudes de excepción |
| [NOT-OPS-018 v1.1](https://netec-mx.github.io/TPRACT_RAG-Priv-/Capitulo01/materiales/vistas/NOT-OPS-018_v1.1.md.html) | Nota de transferencia y texto para el ejercicio de seguridad |

Una referencia como **`[EST-SEG-009-v2.1#2.2]`** identifica el documento, su versión y el apartado que respalda una afirmación.

Además, [catalogo_documental.csv](https://netec-mx.github.io/TPRACT_RAG-Priv-/Capitulo01/materiales/vistas/catalogo_documental.csv.html) describe versiones y fechas; [casos_auditoria.csv](https://netec-mx.github.io/TPRACT_RAG-Priv-/Capitulo01/materiales/vistas/casos_auditoria.csv.html) contiene las siete preguntas; [plan_consultas.csv](https://netec-mx.github.io/TPRACT_RAG-Priv-/Capitulo01/materiales/vistas/plan_consultas.csv.html) reúne búsquedas por subtema; [respuestas_de_contraste.jsonl](https://netec-mx.github.io/TPRACT_RAG-Priv-/Capitulo01/materiales/vistas/respuestas_de_contraste.jsonl.html) permite practicar la evaluación de respuestas; y [anexo_parametros_sesion.csv](https://netec-mx.github.io/TPRACT_RAG-Priv-/Capitulo01/materiales/vistas/anexo_parametros_sesion.csv.html) aporta datos tabulares de configuración. El archivo [manifiesto_materiales.json](https://netec-mx.github.io/TPRACT_RAG-Priv-/Capitulo01/materiales/vistas/manifiesto_materiales.json.html) permite verificar la integridad de los documentos.

La lógica reutilizable se encuentra en [motor_rag.py](https://netec-mx.github.io/TPRACT_RAG-Priv-/Capitulo01/materiales/vistas/motor_rag.py.html). Este componente recupera fragmentos mediante **BM25** (coincidencia de términos) y **LSA** (relaciones estadísticas entre textos), prepara evidencia citable y permite producir hallazgos extractivos. El notebook organiza los ejercicios y muestra cómo aplicar esas funciones.

## Casos de auditoría

| Caso | Pregunta de trabajo |
|---|---|
| A01 | ¿Coinciden el estándar y el manual en la duración de sesiones? |
| A02 | ¿Qué controles corresponden al acceso privilegiado de emergencia? |
| A03 | ¿Qué versión de la guía aplicaba en la fecha de un incidente? |
| A04 | ¿Dos documentos establecen el mismo plazo de escalamiento? |
| A05 | ¿Qué puede concluirse sobre una excepción pendiente de verificar? |
| A06 | ¿Cómo se trata una instrucción incrustada en un documento recuperado? |
| A07 | ¿Qué hallazgos integrados pueden sustentarse con el expediente? |

## Recorrido de la práctica

| Etapa del ciclo | Actividad | Duración |
|---|---|---:|
| Preparación | Verificar bibliotecas, documentos y casos | 5 min |
| 1. Recuperar | Reto 1. Buscar los párrafos correctos | 10 min |
| 2. Preparar la generación | Reto 2. Construir y contrastar prompts aumentados | 15 min |
| 3. Evaluar | Reto 3. Comprobar afirmaciones y citas | 15 min |
| 1 y 3. Ampliar y verificar | Reto 4. Reconsultar y comprobar vigencias | 10 min |
| 3. Evaluar | Reto 5. Evaluar los siete casos y aplicar una rúbrica | 20 min |
| 4. Registrar | Reto 6. Medir cobertura, contexto, latencia y memoria | 15 min |
| Control transversal | Reto 7. Revisar seguridad y datos tabulares | 10 min |
| 5. Presentar | Reto 8. Emitir el dictamen y contrastarlo con Copilot | 5 min |
| **Total** | | **105 min** |

El **notebook** incluye los espacios titulados *Interpretación 1* a *Interpretación 8* y las *Conclusiones de la práctica*. Complete allí sus respuestas; la guía explica qué ejecutar y qué observar.

---

## Preparación — 5 minutos

**Qué realizará:** comprobar las dependencias y reconocer el expediente.

1. Ejecute la primera celda del notebook. `%pip install -r requirements.txt --progress-bar on` verifica los paquetes y utiliza las versiones compatibles ya disponibles; instala las dependencias necesarias en el kernel cuando corresponde.
2. Ejecute la celda de carga. Confirmará **8 documentos, 33 fragmentos citables y 7 casos**.
3. Observe las tres tablas: catálogo de versiones; casos de auditoría; y párrafos originales que establecen **15 minutos** y **30 minutos** de inactividad.
4. Abra en `documentos/` los archivos [EST-SEG-009_v2.1.md](https://netec-mx.github.io/TPRACT_RAG-Priv-/Capitulo01/materiales/vistas/EST-SEG-009_v2.1.md.html) y [MAN-OPS-011_v2.3.md](https://netec-mx.github.io/TPRACT_RAG-Priv-/Capitulo01/materiales/vistas/MAN-OPS-011_v2.3.md.html) para comprobar la diferencia directamente.

**Resultado esperado:** expediente íntegro y fuentes reconocibles por sus identificadores.

## Reto 1. Recuperar los documentos correctos — 10 minutos

**Etapa 1: recuperar contexto.** El objetivo es encontrar **ambos** párrafos que permiten investigar A01.

1. Ejecute la celda que compara una consulta general con consultas por subtema mediante `motor.recuperar_caso()`.
2. Observe la **tabla de cobertura**: `1.0` significa que se han recuperado todas las referencias esperadas para ese caso.
3. En la tabla de fragmentos, revise `cita`, `texto`, `bm25`, `lsa` y `estado_corte`. Las puntuaciones BM25 y LSA ayudan a ordenar resultados; representan relevancia calculada, no certeza de una conclusión.
4. Cambie `TOP_K` de **5** a **3** y **7**, y pruebe `PRESUPUESTO = 3200`. Repita la celda y compruebe qué fuentes permanecen en el contexto.
5. Complete **Interpretación 1** en el notebook.

**Comprobación:** localice `[EST-SEG-009-v2.1#2.2]` y `[MAN-OPS-011-v2.3#3.1]` en la recuperación o explique qué configuración requiere ampliar la búsqueda.

## Reto 2. Preparar el contexto para una respuesta fundamentada — 15 minutos

**Etapa 2: preparar la generación.** Un **prompt aumentado** reúne la pregunta, fragmentos recuperados, referencias e instrucciones de respuesta.

1. Ejecute la celda que crea un prompt de resumen y otro fundamentado mediante `prompt_aumentado()`.
2. Observe la salida: tamaño de cada prompt y las instrucciones que acompañarán a las fuentes.
3. Abra `resultados/paquete_contexto_A01.md`: contiene el prompt fundamentado completo y los fragmentos seleccionados.
4. Copie ese contexto en un chat nuevo de **Microsoft 365 Copilot Chat Básico** y revise si la respuesta diferencia los **15 minutos** del estándar y los **30 minutos** del manual, citando sus fuentes.
5. Complete **Interpretación 2** en el notebook.

**Concepto de Coursera:** temperatura y `top_p` describen métodos de selección de tokens durante la generación con un LLM. Aquí el experimento se centra en las instrucciones y evidencias suministradas a Copilot para obtener una respuesta verificable.

## Reto 3. Comprobar afirmaciones y citas — 15 minutos

**Etapa 3: evaluar evidencia y respuesta.** Una cita puede existir, pero no respaldar el significado de la frase a la que acompaña.

1. Ejecute la celda que lee [respuestas_de_contraste.jsonl](https://netec-mx.github.io/TPRACT_RAG-Priv-/Capitulo01/materiales/vistas/respuestas_de_contraste.jsonl.html) y muestra variantes de respuesta para A01.
2. Compare `citas`, `validas` y `en_contexto`: indican cantidad de referencias y proporciones de referencias localizables.
3. Ejecute la celda de los casos **A01** y **A05**. Observe la tabla con `tipo`, `observacion`, `citas` y `estado`.
4. Contraste la afirmación de ejemplo «El estándar permite 45 minutos» con el texto recuperado del estándar. Distinga **validez de una cita** de **fundamentación de la afirmación**.
5. Complete **Interpretación 3** en el notebook.

**Comprobación:** A01 permite reconocer una discrepancia; A05 permite explicar qué información falta para tomar una decisión documentada.

## Reto 4. Reconsultar y comprobar vigencias — 10 minutos

**Regreso a las etapas 1 y 3.** Algunas preguntas requieren ampliar la búsqueda antes de formular un hallazgo.

1. Ejecute el experimento del caso **A03**. La consulta comienza con `top_k = 3` y amplía a **5** o **7** cuando el estado del análisis indica `Abstencion`.
2. Lea la tabla: cada fila es un intento de búsqueda con sus citas y estado.
3. Observe qué guía era aplicable el **15 de agosto de 2026** y cuál el **15 de septiembre de 2026**.
4. Complete **Interpretación 4** en el notebook.

**Concepto de Coursera:** la reconsulta controlada permite ilustrar un ciclo de búsqueda y evaluación. Para actualizar reglas documentales mediante RAG se revisan las fuentes y su vigencia.

## Reto 5. Evaluar los hallazgos del expediente — 20 minutos

**Etapa 3: medir y revisar calidad.** La calidad de un recuperador y la validez semántica de un hallazgo se examinan con criterios complementarios.

1. Ejecute los siete casos con `TOP_K_EVAL = 5` y `PRESUPUESTO_EVAL = 5600`.
2. Observe la tabla: cada caso muestra `cobertura`, `tasa_citas_en_contexto` y `estado_hallazgo`. Localice los casos que requieren una revisión adicional.
3. Pruebe `TOP_K_EVAL = 3` y `7`, manteniendo el presupuesto. Compare la cobertura obtenida.
4. En la siguiente celda, examine la rúbrica de **A01** y **A05**, con cinco criterios puntuados de 0 a 2: **fundamentación, vigencia, trazabilidad, incertidumbre y utilidad**. Ajuste los valores marcados **MODIFIQUE AQUÍ** según su lectura de las fuentes.
5. Complete **Interpretación 5** en el notebook.

**Archivos generados:** `evaluacion_hallazgos.csv`, `fuentes_recuperadas.csv` y `rubrica_revision.csv`.

## Reto 6. Medir y registrar el funcionamiento — 15 minutos

**Etapa 4: observabilidad.** Compare configuraciones para comprender el compromiso entre recuperación de evidencia, tamaño del contexto y tiempo local de búsqueda.

1. Ejecute el análisis con `top_k = 3/5` y presupuestos de **3200/5600 caracteres**.
2. Lea la tabla: `cobertura`, `caracteres` y `latencia_ms` son observaciones de recuperación; `costo_ilustrativo_usd` es una estimación didáctica derivada del contexto.
3. Examine el gráfico `calidad_vs_contexto.png`: eje horizontal = caracteres promedio; eje vertical = cobertura media.
4. Ejecute el registro de trazas. Observe los campos `caso_id`, `n_recuperados` y `latencia_recuperacion_ms`, y localice `trazas_operacion.jsonl`.
5. Observe la tabla de memoria teórica para **16, 8 y 4 bits por peso** de un modelo de referencia. Relaciónela con el concepto de cuantización del módulo 5.
6. Complete **Interpretación 6**.

**Archivos generados:** `benchmark_configuraciones.csv`, `calidad_vs_contexto.png` y `trazas_operacion.jsonl`.

## Reto 7. Revisar seguridad e información tabular — 10 minutos

**Control transversal: seguridad y formatos de información.**

1. Ejecute la revisión de **A06** y lea el fragmento `[NOT-OPS-018-v1.1#3.1]`. Observe cómo se identifica una instrucción incrustada en una fuente.
2. Revise en la tabla `observacion` y `accion`: explican el riesgo y la revisión propuesta.
3. Ejecute la lectura de [anexo_parametros_sesion.csv](https://netec-mx.github.io/TPRACT_RAG-Priv-/Capitulo01/materiales/vistas/anexo_parametros_sesion.csv.html) y observe su tabla y gráfico de barras.
4. Compare los tiempos configurados para los tres componentes y relacione el dato de la consola operativa con el caso A01.
5. Complete **Interpretación 7**.

**Archivo generado:** `anexo_visual_parametros.png`. El ejercicio muestra cómo preparar valores tabulares para incorporarlos con una referencia verificable en un flujo RAG con múltiples formatos.

## Reto 8. Emitir el dictamen — 5 minutos

**Etapa 5: presentar el resultado trazable.**

1. Ejecute la integración del caso **A07** y abra `resultados/dictamen_auditoria.md`.
2. Revise en la tabla `subcaso`, `tipo`, `estado` y `citas`. Cada sección del dictamen conserva también su observación, límite y acción de revisión.
3. Ejecute la celda que crea `paquete_validacion_copilot.md`. Copie ese texto a Copilot Chat Básico para contrastar la observación de A01 con sus referencias.
4. Ejecute la comprobación técnica final y confirme el mensaje de evidencias verificadas.
5. Complete **Interpretación 8** y las **Conclusiones de la práctica**. Guarde el notebook.

**Resultado esperado:** un dictamen estructurado y rastreable, preparado para la revisión de los responsables del proceso.

## Evidencias que genera el notebook

La carpeta `materiales/resultados/` se crea durante la ejecución. Contendrá:

| Archivo | Utilidad |
|---|---|
| `paquete_contexto_A01.md` | Prompt fundamentado con las fuentes recuperadas |
| `fuentes_recuperadas.csv` | Referencias seleccionadas en los casos de evaluación |
| `evaluacion_hallazgos.csv` | Indicadores de cobertura, citas y estado |
| `rubrica_revision.csv` | Evaluaciones humanas de A01 y A05 |
| `benchmark_configuraciones.csv` | Configuraciones y métricas de recuperación |
| `calidad_vs_contexto.png` | Visualización de cobertura y tamaño de contexto |
| `trazas_operacion.jsonl` | Registro operativo por caso |
| `anexo_visual_parametros.png` | Gráfico de los parámetros del anexo |
| `dictamen_auditoria.md` | Observaciones integradas de A07 |
| `paquete_validacion_copilot.md` | Hallazgo A01 y fuentes para la revisión generativa |
| `manifiesto_ejecucion.json` | Identificador de ejecución, bibliotecas y huellas de los documentos |

## Solución de problemas

| Situación | Comprobación recomendada |
|---|---|
| VS Code vuelve a solicitar el kernel | Verifique las extensiones Python/Jupyter y la instalación de `ipykernel` en el intérprete seleccionado. |
| Una biblioteca no se puede importar | Revise [requirements.txt](https://netec-mx.github.io/TPRACT_RAG-Priv-/Capitulo01/materiales/vistas/requirements.txt.html) y el repositorio de paquetes autorizado por TI. |
| No se encuentran los materiales | Abra la carpeta completa `Capitulo01/materiales` antes de ejecutar la primera celda. |
| Se detecta una diferencia de integridad SHA-256 | Restaure el documento original y vuelva a ejecutar la carga. |
| Una consulta recupera poca evidencia | Compare `TOP_K`, el presupuesto de contexto y las consultas por subtema. |
| Copilot genera una conclusión distinta | Identifique la afirmación discutida y compruebe sus referencias en los documentos. |

## Recursos de consulta

- [Curso RAG de Coursera](https://www.coursera.org/learn/retrieval-augmented-generation-rag)
- [Extracción de características de texto — scikit-learn](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
- [Análisis semántico latente — scikit-learn](https://scikit-learn.org/stable/modules/decomposition.html#lsa)
- [Jupyter notebooks en Visual Studio Code](https://code.visualstudio.com/docs/datascience/jupyter-notebooks)
