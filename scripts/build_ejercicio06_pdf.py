"""Genera el PDF breve del Ejercicio 06 (VAE, evaluación y Model Card)."""

from __future__ import annotations

import html
import json
import os
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
REPO_URL = os.getenv("REPO_URL", "https://github.com/edvvdev/unidad04-06")

metricas = json.loads((REPORTS / "ejercicio06_metricas.json").read_text(encoding="utf-8"))
cierre = (ROOT / "docs/CIERRE_INTERPRETATIVO_LAB10.md").read_text(encoding="utf-8")
model_card = (ROOT / "MODEL_CARD.md").read_text(encoding="utf-8")

ENCABEZADO_CANONICO = "### Párrafo canónico · diversidad y vecinos cercanos"
SECCION_DIVERSIDAD = "Evidencia de diversidad"


def negritas(texto: str) -> str:
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", texto)


def numero(valor, decimales: int = 4) -> str:
    """Formato español: coma decimal y punto de millar."""
    return f"{valor:,.{decimales}f}".replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def fecha_corta(iso: str) -> str:
    return iso.split("T")[0]


def parrafo_canonico(texto: str) -> str:
    """Extrae de la Model Card el párrafo canónico de diversidad y vecinos cercanos."""
    partes = texto.split(ENCABEZADO_CANONICO, 1)
    if len(partes) != 2:
        raise ValueError("La Model Card no contiene el encabezado del párrafo canónico")
    return partes[1].strip().split("\n\n", 1)[0].strip()


def bloques_markdown(texto: str) -> list[tuple[str, object]]:
    """Convierte markdown sencillo en bloques ('h2', 'p' o 'tabla')."""
    bloques: list[tuple[str, object]] = []
    parrafo: list[str] = []
    filas: list[str] = []
    for linea in texto.split("\n"):
        limpia = linea.strip()
        if limpia.startswith("|"):
            if parrafo:
                bloques.append(("p", " ".join(parrafo)))
                parrafo = []
            filas.append(limpia)
            continue
        if filas:
            bloques.append(("tabla", filas))
            filas = []
        if limpia.startswith("## "):
            if parrafo:
                bloques.append(("p", " ".join(parrafo)))
                parrafo = []
            bloques.append(("h2", limpia[3:].strip()))
        elif not limpia or set(limpia) == {"-"}:
            if parrafo:
                bloques.append(("p", " ".join(parrafo)))
                parrafo = []
        elif limpia[0].isdigit() and limpia[1:3] == ". ":
            if parrafo:
                bloques.append(("p", " ".join(parrafo)))
                parrafo = []
            bloques.append(("p", limpia))
        else:
            parrafo.append(limpia)
    if parrafo:
        bloques.append(("p", " ".join(parrafo)))
    if filas:
        bloques.append(("tabla", filas))
    return bloques


def celda(texto: str, estilo) -> Paragraph:
    """Convierte un texto de celda en párrafo para que ajuste el ancho de la columna."""
    return Paragraph(html.escape(str(texto)), estilo)


def filas_de_tabla(lineas: list[str]) -> list[list[str]]:
    """Convierte las líneas de una tabla markdown en celdas."""
    celdas = []
    for linea in lineas:
        valores = [valor.strip() for valor in linea.strip("|").split("|")]
        if all(set(valor) <= set("-: ") for valor in valores):
            continue
        celdas.append(valores)
    return celdas


canonico = parrafo_canonico(model_card)
if canonico not in cierre:
    raise ValueError("El cierre interpretativo debe citar literalmente el párrafo canónico de la Model Card")
for clave in (
    "diversidad_media_entre_muestras",
    "distancia_media_al_vecino_mas_cercano",
    "distancia_media_entre_pares_de_prueba",
):
    valor = numero(metricas[clave])
    if valor not in canonico:
        raise ValueError(f"El párrafo canónico no está sincronizado con {clave} ({valor})")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="Title2", parent=styles["Title"], fontSize=17, spaceAfter=6))
styles.add(ParagraphStyle(name="H1s", parent=styles["Heading1"], fontSize=12.5, spaceBefore=10, spaceAfter=4))
styles.add(ParagraphStyle(name="BodyJ", parent=styles["BodyText"], alignment=TA_JUSTIFY, leading=13.5, fontSize=9.6))
styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8.2, leading=10.5))
styles.add(ParagraphStyle(name="Cita", parent=styles["BodyText"], alignment=TA_JUSTIFY, leading=13, fontSize=9.2,
                          leftIndent=0.6 * cm, borderColor=colors.HexColor("#0f7c7b"), borderWidth=0,
                          borderPadding=4, spaceBefore=4, spaceAfter=4))
styles.add(ParagraphStyle(name="Celda", parent=styles["BodyText"], fontSize=7.8, leading=9.4, spaceAfter=0))
styles.add(ParagraphStyle(name="CeldaId", parent=styles["BodyText"], fontSize=8.4, leading=10, spaceAfter=0))

story = []
story.append(Paragraph("Ejercicio 06 · VAE, evaluación y Model Card", styles["Title2"]))
story.append(Paragraph(
    "INF-8239 Ciencia de Datos II (UASD) · Unidad 03 · Consolidación de LAB10", styles["Small"]))
story.append(Spacer(1, 6))

identificacion = [
        ["Estudiante", "Edwing Perez"],
        ["Profesor", "Edwin Ramón José Nolasco"],
    ["Repositorio", REPO_URL],
    ["Dataset", "Fashion-MNIST (Zalando) · licencia MIT · 60.000 entrenamiento + 10.000 prueba, 28×28 grises"],
    ["Entorno", "Python 3.12 · TensorFlow 2.21 · CPU (sin GPU detectada) · semilla 42"],
    ["Corrida de la evaluación",
     (f"{fecha_corta(metricas['fecha_ejecucion'])} "
      f"(fecha_ejecucion: {metricas['fecha_ejecucion']})")],
]
identificacion = [[celda(valor, styles["CeldaId"]) for valor in fila] for fila in identificacion]
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
     numero(metricas["mse_reconstruccion_ae"], 5), numero(metricas["mse_reconstruccion_vae"], 5),
     "El AE reconstruye mejor: la penalización KL compra espacio muestreable a costa de nitidez"],
    ["Divergencia KL media", "—", numero(metricas["kl_media_vae"]),
     "El latente se aleja de la referencia N(0,1)"],
    ["Tiempo de entrenamiento (s)", numero(metricas["segundos_entrenamiento_ae"], 2),
     numero(metricas["segundos_entrenamiento_vae"], 2),
     f"CPU de la corrida del {fecha_corta(metricas['fecha_ejecucion'])}; contextual, no energía"],
    ["Parámetros", numero(metricas["parametros_ae"], 0), numero(metricas["parametros_decoder"], 0) + " (decoder)",
     "El decoder del VAE cuadruplica los parámetros del AE"],
    ["Tamaño del modelo (KB)", numero(metricas["kb_modelo_ae"], 0), numero(metricas["kb_decoder"], 0),
     "Artefactos en models/, no versionados en Git"],
    ["Diversidad media entre muestras (VAE)", "—", numero(metricas["diversidad_media_entre_muestras"]),
     "Distancia media entre 64 muestras del decoder, en el espacio de píxeles aplanados"],
    ["Distancia media al vecino más cercano (VAE)", "—", numero(metricas["distancia_media_al_vecino_mas_cercano"]),
     "Contra 1.000 imágenes del conjunto de prueba: no permite descartar la memorización"],
    ["Línea base: distancia media entre pares de prueba", "—", numero(metricas["distancia_media_entre_pares_de_prueba"]),
     "Referencia del conjunto de prueba: es lo que hace interpretable la distancia al vecino más cercano"],
]
tabla_res = Table([[celda(valor, styles["Celda"]) for valor in fila] for fila in filas],
                  colWidths=[4.6 * cm, 2.6 * cm, 2.6 * cm, 6.0 * cm], repeatRows=1)
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
story.append(Spacer(1, 4))
story.append(Paragraph(
    "Nota 1. Las tres métricas de muestreo (diversidad media, distancia al vecino más cercano y "
    "línea base de pares) aplican <b>solo al VAE</b>: el AE no tiene muestreador, solo reconstruye, "
    "por lo que su columna queda en «—».", styles["Small"]))
story.append(Paragraph(
    f"Nota 2. Todas las cifras proceden de una única corrida (notebook ejecutado el "
    f"{fecha_corta(metricas['fecha_ejecucion'])}) y están en `reports/ejercicio06_metricas.json`, "
    f"que es la única fuente de verdad de la entrega; por ejemplo, "
    f"`distancia_media_entre_pares_de_prueba` = {metricas['distancia_media_entre_pares_de_prueba']:.4f} "
    f"(en el texto se usa la coma decimal).", styles["Small"]))
story.append(Spacer(1, 6))
story.append(Paragraph("Lectura canónica de diversidad y vecinos cercanos", styles["H1s"]))
story.append(Paragraph(
    "Párrafo canónico de MODEL_CARD.md, citado literalmente en el cierre interpretativo y en este PDF.",
    styles["Small"]))
story.append(Paragraph(negritas(canonico), styles["Cita"]))
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
story.append(Paragraph(
    "La evidencia de diversidad y de vecinos cercanos se presenta en la Tabla 1 y en el párrafo "
    "canónico citado literalmente en la Model Card y en el cierre interpretativo "
    "(docs/CIERRE_INTERPRETATIVO_LAB10.md).", styles["Small"]))
story.append(Spacer(1, 4))
for indice, seccion in enumerate(re.split(r"\n## ", cierre)):
    seccion = seccion.strip()
    if not seccion or indice == 0:
        continue
    titulo, _, resto = seccion.partition("\n")
    titulo = titulo.removeprefix("## ").strip()
    if titulo == SECCION_DIVERSIDAD:
        continue
    story.append(Paragraph(negritas(titulo), styles["H1s"]))
    for tipo, contenido in bloques_markdown(resto):
        if tipo == "h2":
            continue
        if tipo == "p":
            texto = negritas(str(contenido).replace("\n", " "))
            if texto:
                story.append(Paragraph(texto, styles["BodyJ"]))
                story.append(Spacer(1, 4))
        else:
            celdas = filas_de_tabla(contenido)
            if celdas:
                tabla = Table([[celda(valor, styles["Celda"]) for valor in fila] for fila in celdas],
                              colWidths=[7.4 * cm, 8.4 * cm])
                tabla.setStyle(TableStyle([
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#94a3b8")),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8.2),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 2),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ]))
                story.append(tabla)
                story.append(Spacer(1, 6))

story.append(Spacer(1, 6))
story.append(Paragraph(
    "<b>Uso responsable de IA.</b> El asistente de IA apoyó la redacción de la Model Card, la "
    "extensión del script base para guardar curvas y métricas de reconstrucción y la construcción de "
    "este documento; los cambios están enumerados en <font face='Courier'>CAMBIOS_AL_BASE.md</font> y la "
    "declaración completa en <font face='Courier'>AI_USE_DECLARATION.md</font>. El estudiante conserva "
    "la responsabilidad sobre los datos, el código y estas conclusiones. Ninguna muestra generada se "
    "presenta como imagen real, original o libre de sesgo: el análisis de vecinos cercanos de esta "
    "entrega se calculó contra el conjunto de prueba y no contra el conjunto de entrenamiento, de modo "
    "que no permite descartar la memorización.", styles["Small"]))

target = ROOT / "docs/EJERCICIO_06_PDF_BRIEF.pdf"
document = SimpleDocTemplate(str(target), pagesize=LETTER, title="Ejercicio 06 · INF-8239",
                              leftMargin=1.5 * cm, rightMargin=1.5 * cm, topMargin=1.5 * cm, bottomMargin=1.5 * cm)
document.build(story)
print("PDF generado:", target)
