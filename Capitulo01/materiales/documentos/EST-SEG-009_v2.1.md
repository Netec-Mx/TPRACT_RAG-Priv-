# Estándar técnico de sesiones privilegiadas

| Campo | Información |
|---|---|
| Código | EST-SEG-009 |
| Versión | 2.1 |
| Estado | Vigente |
| Fecha de vigencia | 2026-03-01 |
| Área responsable | Seguridad de plataformas |
| Alcance | Parámetros mínimos de sesiones de administración |
| Clasificación | Uso interno del caso de capacitación |

## 1. Objetivo

Precisar los parámetros técnicos que complementan los requisitos definidos para accesos privilegiados.

## 2. Autenticación y duración

[EST-SEG-009-v2.1#2.1] Las sesiones iniciadas con cuentas privilegiadas requieren segundo factor; no se contempla una omisión automática del control por tratarse de operaciones urgentes.

[EST-SEG-009-v2.1#2.2] Toda consola privilegiada debe finalizar la sesión tras 15 minutos consecutivos de inactividad. El parámetro se configura sobre el mecanismo de sesión y se verifica durante la revisión técnica del servicio.

[EST-SEG-009-v2.1#2.3] El vencimiento por inactividad es independiente de la duración total del ticket de atención. Una tarea de varias horas puede requerir múltiples sesiones autenticadas.

## 3. Evidencia técnica

[EST-SEG-009-v2.1#3.1] La verificación técnica registra identificador de plataforma, valor configurado de expiración, fecha de prueba y resultado observado.

## 4. Control de cambios

| Versión | Vigencia | Cambio principal |
|---|---|---|
| 2.0 | 2025-09-01 | Parámetros generales de sesión |
| 2.1 | 2026-03-01 | Tiempo de inactividad especificado en 15 minutos |
