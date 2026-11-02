"""Genera el PDF breve del Ejercicio 06 (VAE, evaluación y Model Card)."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
REPO_URL = os.getenv("REPO_URL", "https://github.com/edvvdev/unidad04-06")

metricas = json.loads((REPORTS / "ejercicio06_metricas.json").read_text(encoding="utf-8"))
cierre = (ROOT / "docs/CIERRE_INTERPRETATIVO_LAB10.md").read_text(encoding="utf-8")


def negritas(texto: str) -> str:
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", texto)


styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="Title2", parent=styles["Title"], fontSize=17, spaceAfter=6))
styles.add(ParagraphStyle(name="H1s", parent=styles["Heading1"], fontSize=12.5, spaceBefore=10, spaceAfter=4))
styles.add(ParagraphStyle(name="BodyJ", parent=styles["BodyText"], alignment=TA_JUSTIFY, leading=13.5, fontSize=9.6))
styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8.2, leading=10.5))

story = []
story.append(Paragraph("Ejercicio 06 · VAE, evaluación y Model Card", styles["Title2"]))
story.append(Paragraph(
    "INF-8239 Ciencia de Datos II (UASD) · Unidad 04 · Consolidación de LAB10", styles["Small"]))
story.append(Spacer(1, 6))

identificacion = [
        ["Estudiante", "Edwing Perez · matrícula: PENDIENTE"],
        ["Profesor", "Edwin Ramón José Nolasco"],
    ["Repositorio", REPO_URL],
    ["Dataset", "Fashion-MNIST (Zalando) · licencia MIT · 60.000 entrenamiento + 10.000 prueba, 28×28 grises"],
    ["Entorno", "Python 3.12 · TensorFlow 2.21 · CPU (sin GPU detectada) · semilla 42"],
]
tabla_id = Table(identificacion, colWidths=[4.4 * cm, 11.4 * cm])
tabla_id.setStyle(TableStyle([
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#94a3b8")),
    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f1f5f9")),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 8.4),
    ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ("TOPPADDING", (0, 0), (-1, -1), 3),
]))
story.append(tabla_id)
story.append(Spacer(1, 10))

story.append(Paragraph("1. Tabla principal de resultados", styles["H1s"]))
filas = [
    ["Métrica", "Autoencoder (AE)", "VAE", "Lectura"],
    ["MSE de reconstrucción (1.000 de prueba)",
     f"{metricas['mse_reconstruccion_ae']:.5f}", f"{metricas['mse_reconstruccion_vae']:.5f}",
     "El AE reconstruye mejor: la penalización KL compra espacio muestreable a costa de nitidez"],
    ["Divergencia KL media", "—", f"{metricas['kl_media_vae']:.4f}",
     "El latente se aleja de la referencia N(0,1)"],
    ["Tiempo de entrenamiento (s)", f"{metricas['segundos_entrenamiento_ae']:.2f}",
     f"{metricas['segundos_entrenamiento_vae']:.2f}", "Coste en CPU; contextual, no energía"],
    ["Parámetros", f"{metricas['parametros_ae']:,}", f"{metricas['parametros_decoder']:,} (decoder)",
     "El decoder del VAE cuadruplica los parámetros del AE"],
    ["Tamaño del modelo (KB)", f"{metricas['kb_modelo_ae']:.0f}", f"{metricas['kb_decoder']:.0f}",
     "Artefactos en models/, no versionados"],
    ["Diversidad media entre muestras", f"{metricas['diversidad_media_entre_muestras']:.4f}",
     f"{metricas['diversidad_media_entre_muestras']:.4f}", "Distancia media entre 64 muestras del decoder"],
    ["Distancia media al vecino más cercano", "—", f"{metricas['distancia_media_al_vecino_mas_cercano']:.4f}",
     "Indicador de posible memorización (no concluyente)"],
]
tabla_res = Table(filas, colWidths=[4.6 * cm, 2.6 * cm, 2.6 * cm, 6.0 * cm], repeatRows=1)
tabla_res.setStyle(TableStyle([
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#94a3b8")),
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f7c7b")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 7.8),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
    ("TOPPADDING", (0, 0), (-1, -1), 3),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
]))
story.append(tabla_res)
story.append(Spacer(1, 8))

story.append(Paragraph("2. Figuras", styles["H1s"]))
story.append(Paragraph(
    "Figura 1. Panel de 16 ejemplos de prueba elegidos con semilla 42 (no seleccionados a mano): "
    "originales, reconstrucción AE, reconstrucción VAE y muestras nuevas del decoder.", styles["Small"]))
story.append(Image(str(REPORTS / "ejercicio06_panel.png"), width=16 * cm, height=4.6 * cm))
story.append(Spacer(1, 4))
story.append(Paragraph("Figura 2. Interpolación en el espacio latente de 2 dimensiones del VAE.", styles["Small"]))
story.append(Image(str(REPORTS / "ejercicio06_interpolacion.png"), width=16 * cm, height=1.6 * cm))
story.append(Spacer(1, 4))
story.append(Paragraph(
    "Figura 3. Curvas: pérdida del autoencoder (entrenamiento y validación) y del VAE "
    "(total, reconstrucción y KL).", styles["Small"]))
story.append(Image(str(REPORTS / "ejercicio06_curvas.png"), width=14 * cm, height=5.2 * cm))
story.append(Spacer(1, 4))

story.append(PageBreak())
story.append(Paragraph("3. Conclusión y decisión", styles["H1s"]))
cuerpo = cierre.split("## Resultado de reconstrucción", 1)[-1]
for parrafo in cuerpo.strip().split("\n\n"):
    texto = negritas(parrafo.strip().replace("\n", " "))
    if texto:
        story.append(Paragraph(texto, styles["BodyJ"]))
        story.append(Spacer(1, 4))

story.append(Spacer(1, 6))
story.append(Paragraph(
    "<b>Uso responsable de IA.</b> El asistente de IA apoyó la redacción de la Model Card, la "
    "extensión del script base para guardar curvas y métricas de reconstrucción y la construcción de "
    "este documento; los cambios están enumerados en <font face='Courier'>CAMBIOS_AL_BASE.md</font> y la "
    "declaración completa en <font face='Courier'>AI_USE_DECLARATION.md</font>. El estudiante conserva "
    "la responsabilidad sobre los datos, el código y estas conclusiones. Ninguna muestra generada se "
    "presenta como imagen real, original o libre de sesgo: sin análisis de vecinos cercanos sobre el "
    "conjunto de entrenamiento completo no puede afirmarse privacidad.", styles["Small"]))

target = ROOT / "docs/EJERCICIO_06_PDF_BRIEF.pdf"
document = SimpleDocTemplate(str(target), pagesize=LETTER, title="Ejercicio 06 · INF-8239",
                              leftMargin=1.5 * cm, rightMargin=1.5 * cm, topMargin=1.5 * cm, bottomMargin=1.5 * cm)
document.build(story)
print("PDF generado:", target)
