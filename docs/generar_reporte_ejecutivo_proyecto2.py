from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


OUTPUT = Path(__file__).with_name("Reporte_Ejecutivo_Proyecto2.pdf")

BODY = [
    (
        "Resumen ejecutivo",
        [
            "Este reporte evalúa un prototipo de priorización de riesgo de lavado de dinero construido sobre el dataset IBM AML. El objetivo no es automatizar una decisión sancionatoria: es ordenar remitentes para que un equipo de cumplimiento investigue primero los casos con mayor señal estadística. El conjunto original contiene 5,078,345 transacciones y 496,995 remitentes. Para el experimento se formó un subconjunto final de 60,000 remitentes, con 1,463 positivos, equivalente a una prevalencia de 2.4383 por ciento. Esta baja prevalencia hace que la precisión operativa y la recuperación de casos en el tramo superior sean más útiles que una exactitud global aislada.",
            "La evidencia disponible respalda una conclusión matizada. El ajuste fino del codificador recupera casi todo el desempeño del modelo base; al añadir reconstrucción obtiene la mejor media de AUPRC y ROC-AUC. Sin embargo, el modelo base conserva mejor Precision@1% y Recall@1%, las métricas que describen una capacidad de investigación restringida. Por ello, no corresponde presentar el prototipo como listo para producción ni afirmar un cumplimiento de tiempo en GPU T4. Su valor actual es demostrar una arquitectura secuencial viable, documentar los riesgos de evaluación y proponer los controles que separarían un piloto analítico de una operación de cumplimiento responsable.",
        ],
    ),
    (
        "Problema de negocio y contexto regulatorio",
        [
            "Las instituciones financieras deben detectar patrones que, vistos transacción por transacción, pueden parecer ordinarios, pero que como secuencia sugieren estructuración, dispersión de fondos o actividad incongruente con el perfil conocido. El costo operativo es asimétrico: investigar manualmente a todos los remitentes es inviable, mientras que ignorar señales relevantes expone a pérdidas, sanciones y deterioro reputacional. Un ranking de riesgo permite asignar la capacidad humana limitada hacia expedientes con mayor prioridad, sin sustituir el juicio del analista ni la investigación documentada.",
            "En Guatemala, el marco de prevención debe leerse junto con el Decreto 67-2001 y con las recomendaciones del Grupo de Acción Financiera Internacional. La evaluación mutua de GAFILAT para Guatemala de 2016 aporta contexto institucional sobre la eficacia del sistema. Estas fuentes orientan hacia un enfoque basado en riesgo, debida diligencia, trazabilidad y reporte de operaciones sospechosas. Este reporte no atribuye artículos específicos al Decreto 67-2001 porque no se verificaron para este documento. Tampoco interpreta el modelo como una prueba legal: una puntuación es una señal de priorización que requiere evidencia adicional, revisión humana y reglas de escalamiento consistentes con la política de la institución.",
            "Desde la perspectiva de negocio, el umbral no debe elegirse por una métrica académica aislada. Debe ajustarse a la cantidad real de expedientes que el equipo puede revisar, al costo de los falsos positivos, a la severidad del riesgo no detectado y a la calidad de la retroalimentación investigativa. Precision@1% y Recall@1% son especialmente útiles porque describen el desempeño al concentrar el esfuerzo en el uno por ciento superior del ranking. Una política responsable calibraría ese corte por capacidad y mediría la utilidad de las alertas concluidas, no solo las etiquetas históricas.",
        ],
    ),
    (
        "Datos, diseño y protocolo de evaluación",
        [
            "Cada remitente se representa mediante una secuencia de hasta 29 transacciones y 14 variables de entrada. El máximo observado es de 29 transacciones. El diseño emplea un codificador secuencial con atención para resumir la historia transaccional en un vector contextual y producir una puntuación de riesgo. La atención ayuda a redistribuir el peso de las posiciones de la secuencia para construir esa representación; no demuestra causalidad, intención ni responsabilidad de una transacción particular. Por tanto, cualquier explicación presentada al analista debe describirse como contribución al vector contextual y debe complementarse con hechos verificables del expediente.",
            "La partición es de 41,999 remitentes de entrenamiento con 1,024 positivos, 8,999 de validación con 219 positivos y 9,002 de prueba con 220 positivos. El protocolo compara cinco configuraciones en tres semillas y reporta media con desviación estándar: una base supervisada, la etapa A, el codificador congelado, ajuste fino y ajuste fino con reconstrucción. AUPRC resume la discriminación en un escenario desbalanceado; ROC-AUC aporta una vista global de ordenamiento; Precision@1% y Recall@1% muestran el resultado en la cola de revisión prioritaria.",
            "La etapa A aprende una representación inicial, mientras que las variantes posteriores examinan si transferir esa representación mejora la clasificación. El codificador congelado conserva sus pesos y entrena solamente la parte de decisión; el ajuste fino permite adaptar el codificador al objetivo supervisado. La variante con reconstrucción mantiene una tarea auxiliar para preservar información de la secuencia. Este diseño permite distinguir una representación útil de un transfer learning útil para la tarea concreta, en vez de concluir por una sola corrida que una técnica funciona o falla.",
        ],
    ),
    (
        "Resultados de ablación y lectura operativa",
        [
            "La tabla siguiente concentra los resultados. La etapa A por sí sola es claramente insuficiente como clasificador operativo. Congelar el codificador mejora frente a esa etapa, pero permanece muy por debajo de la base. El ajuste fino recupera el desempeño general y la variante con reconstrucción alcanza la AUPRC más alta, 0.371 más o menos 0.013, y el ROC-AUC más alto, 0.901 más o menos 0.009. La desviación estándar entre semillas recuerda que las diferencias pequeñas deben leerse con prudencia y contrastarse antes de tomar una decisión de implementación.",
            "La decisión práctica no es automática. La base alcanza Precision@1% de 0.663 más o menos 0.017 y Recall@1% de 0.274 más o menos 0.007; ajuste fino con reconstrucción obtiene 0.623 más o menos 0.017 y 0.258 más o menos 0.007. Así, la variante con reconstrucción mejora la calidad global de ranking, pero no supera a la base en el segmento que recibiría la primera oleada de investigación. Si la capacidad de revisión se limita al uno por ciento superior, la base es el referente operativo actual. Si se busca mejorar el ordenamiento en un rango más amplio, la variante con reconstrucción merece validación adicional, calibración y análisis de costo.",
            "El resultado enseña una disciplina importante: no conviene seleccionar por la métrica más favorable. El tablero de decisión debe mostrar simultáneamente prevalencia, AUPRC, rendimiento en la capacidad efectiva, estabilidad entre semillas y consecuencias de error. La evidencia favorece conservar ambas variantes como candidatas para una validación temporal futura, no reemplazar una por la otra con base exclusiva en ROC-AUC. En particular, un descenso en Precision@1% puede traducirse directamente en más expedientes improductivos cuando la unidad de trabajo es el analista.",
        ],
    ),
    (
        "Casos ilustrativos y uso prudente de atención",
        [
            "Los casos de prueba muestran cómo debe comunicarse una alerta. Los verdaderos positivos 808839F70, 8007EF950 y 80011FF60 recibieron probabilidades de 97.8 por ciento, 96.1 por ciento y 95.4 por ciento, con errores de la etapa A de 0.73, 1.11 y 1.13, respectivamente. Son ejemplos de señales que el ranking eleva para inspección temprana, no veredictos. El analista debe contrastar la secuencia con datos KYC, beneficiario final, canal, geografía, contrapartes y evidencia de investigación antes de concluir que existe una operación sospechosa.",
            "El falso positivo 801B16420 recibió 69.4 por ciento y tuvo error de etapa A de 1.95. Es una advertencia concreta de que un puntaje moderadamente alto puede movilizar trabajo sin corresponder a una etiqueta positiva. El falso negativo 800296920 recibió solo 5.0 por ciento, aunque su error de etapa A fue 0.58. Este último caso es más delicado: ilustra que un proceso basado únicamente en el ranking puede dejar un riesgo sin priorizar. Las reglas vigentes, los reportes internos y la posibilidad de que el analista eleve un caso por evidencia externa deben seguir funcionando como capas de defensa.",
            "Cuando se visualicen pesos de atención, la afirmación correcta es limitada: describen qué posiciones contribuyeron más al vector contextual del modelo en esa predicción. No identifican la causa real del comportamiento, no prueban que una transacción haya sido ilícita y no sustituyen una narrativa investigativa. Un diseño de interfaz responsable debería mostrar la secuencia, los atributos disponibles, la incertidumbre y un enlace a la evidencia, evitando etiquetas que sugieran causalidad. Los cinco casos sirven para entrenar esta lectura disciplinada del sistema.",
        ],
    ),
    (
        "Limitaciones, evidencia pendiente y riesgos",
        [
            "Las limitaciones condicionan cualquier interpretación. Los datos y las etiquetas son sintéticos, por lo que las relaciones aprendidas pueden no reflejar el comportamiento, el sesgo de registro o las tácticas adaptativas de una cartera real. Además, no existe una validación temporal estricta demostrada; una partición que mezcla periodos puede ofrecer una estimación optimista cuando los patrones cambian con el tiempo. Antes de comparar modelos para adopción, debe probarse una ventana histórica de entrenamiento seguida por periodos futuros, con una definición clara de disponibilidad de cada variable.",
            "Hay dos detalles técnicos que requieren corrección antes de usar el resultado como evidencia definitiva. MAX_LEN se calculó antes del filtro, lo cual puede hacer que el límite de secuencia no represente exactamente la población modelada. También existe una heurística de fraccionamiento defectuosa. Ambos puntos afectan la reproducibilidad y la interpretación del preprocesamiento, por lo que deben resolverse y volver a ejecutarse bajo un protocolo congelado. Ninguna tabla del presente reporte debe ocultar estos defectos ni usarla para reclamar preparación productiva.",
            "El MVP disponible está limitado a 2,000 de los 9,002 remitentes de prueba. Es útil para demostrar navegación y explicación, pero no representa una cobertura completa de evaluación ni un servicio de inferencia. La evidencia de ejecución disponible en CPU registra 32.8 minutos y terminó en AssertionError. El notebook actual usa 10 epochs en la etapa B, pero no cuenta con una medición T4 ni con resultados actualizados verificables. En consecuencia, no se afirma que haya terminado en una T4 en menos de 30 minutos; esa condición permanece pendiente de una ejecución limpia, medida y archivada.",
        ],
    ),
    (
        "Ruta a producción y gobierno",
        [
            "La transición responsable empieza con datos históricos en los que las investigaciones estén concluidas y las etiquetas tengan linaje. El conjunto debe incorporar, bajo controles de privacidad y acceso, señales de KYC, beneficiario final, canal, geografía y relaciones con contrapartes. La validación debe ser temporal, con periodos de entrenamiento, validación y prueba que respeten cuándo se conocía cada dato. Las reglas de calidad deben impedir fuga de información, registrar transformaciones y conservar versiones inmutables del conjunto, del código, de los parámetros y de los umbrales.",
            "La operación debe mantener revisión humana como autoridad de decisión. El modelo entrega una cola priorizada, explicaciones prudentes y evidencia navegable; el analista documenta el resultado, solicita información adicional o descarta la alerta. Esa resolución alimenta un ciclo de aprendizaje gobernado, no un reentrenamiento automático sin controles. Deben definirse responsabilidades, criterios de escalamiento, tiempos de respuesta y trazas de auditoría que permitan reconstruir qué versión generó una alerta, qué información se mostró y quién tomó la decisión.",
            "El monitoreo debe cubrir drift de datos, drift de desempeño, tasas de alertas, distribución de puntajes, precisión de casos investigados y diferencias por segmentos relevantes. Una alerta de drift exige análisis y posible recalibración, no necesariamente un cambio inmediato de modelo. El umbral debe calibrarse por capacidad de revisión y por el costo de los errores, con simulaciones de carga antes de modificar producción. Finalmente, la auditoría periódica debe revisar seguridad, acceso, equidad operacional, retención de datos y cumplimiento de la política AML. Este camino convierte el prototipo en un candidato a piloto controlado, no en un sustituto de los controles existentes.",
        ],
    ),
    (
        "Conclusión",
        [
            "El Proyecto 2 entrega evidencia útil para priorización analítica: un modelo secuencial evaluado con ablación, métricas pertinentes al desbalance y casos que revelan tanto aciertos como fallos. El ajuste fino con reconstrucción lidera AUPRC y ROC-AUC, mientras la base conserva ventaja en el primer uno por ciento de revisión. La conclusión ejecutiva es conservar el prototipo como instrumento de aprendizaje y evaluación controlada, con la base como referencia para capacidad restringida y la variante con reconstrucción como candidata para análisis adicional.",
            "La decisión responsable depende menos de una cifra sobresaliente que de la calidad del proceso: datos reales con investigación concluida, validación temporal, corrección de defectos de preprocesamiento, revisión humana, gobierno de versiones, monitoreo y calibración por capacidad. Hasta contar con una ejecución T4 medida y resultados actualizados, el requisito de 30 minutos no está demostrado. Esta transparencia protege la utilidad del proyecto y evita convertir una promesa técnica en una afirmación que la evidencia actual no sostiene.",
            "Como siguiente hito, la organización debería fijar un protocolo de aceptación antes de volver a entrenar: datos y fecha de corte definidos, controles de fuga revisados, semillas y dependencias registradas, métricas de ranking vinculadas a capacidad, y criterios explícitos para aprobar, rechazar o investigar diferencias entre variantes. El resultado debe incluir duración medida, estado final de ejecución y artefactos reproducibles. Este registro permitirá a responsables técnicos, de cumplimiento y de auditoría discutir la evidencia sobre una base común, sin confundir una demostración visual con una validación operativa completa.",
        ],
    ),
]

REFERENCES = [
    "Jullum, M., Løland, A., Huseby, R. B., Ånonsen, G., & Lorentzen, J. (2020). Detecting money laundering transactions with machine learning. Journal of Money Laundering Control. https://doi.org/10.1108/JMLC-07-2019-0055",
    "Kute, D. V., Pradhan, B., Shukla, N., & Alamri, A. (2021). Deep learning and explainable artificial intelligence techniques applied for detecting money laundering - A critical review. IEEE Access, 9, 82300-82317. https://doi.org/10.1109/ACCESS.2021.3086230",
    "Cheng, D., Xiang, S., Shang, C., Zhang, Y., Yang, F., & Zhang, L. (2023). Spatio-temporal attention-based neural network for credit card fraud detection. IEEE Transactions on Knowledge and Data Engineering. https://doi.org/10.1109/TKDE.2023.3272396",
    "Financial Action Task Force. (2012-2023). International Standards on Combating Money Laundering and the Financing of Terrorism & Proliferation: The FATF Recommendations. https://www.fatf-gafi.org/en/publications/Fatfrecommendations/Fatf-recommendations.html",
    "GAFILAT. (2016). Informe de evaluación mutua de la República de Guatemala. https://www.gafilat.org/index.php/es/biblioteca-virtual/miembros/guatemala/evaluaciones-mutuas-11/3374-informe-de-evaluacion-mutua-de-guatemala/file",
    "Congreso de la República de Guatemala. (2001). Decreto Número 67-2001, Ley contra el lavado de dinero u otros activos. https://www.oj.gob.gt/es/QueEsOJ/EstructuraOJ/UnidadesAdministrativas/CentroAnalisisDocumentacionJudicial/cds/CDs%20leyes/2001/pdfs/decretos/D067-2001.pdf",
]

AI_DECLARATION = (
    "Declaración de uso de IA: Se utilizó asistencia de IA generativa para apoyar "
    "la redacción, organización y revisión de claridad de este reporte. El contenido "
    "factual se limitó al dossier proporcionado para el Proyecto 2; las métricas, "
    "limitaciones y casos no fueron generados como resultados experimentales. La "
    "responsabilidad de verificar las fuentes, revisar el PDF y aprobar el contenido "
    "final corresponde al autor del trabajo."
)


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+(?:[.-][A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+)*", text))


def page_number(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#1E3A5F"))
    canvas.line(doc.leftMargin, 1.45 * cm, A4[0] - doc.rightMargin, 1.45 * cm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#425466"))
    canvas.drawString(doc.leftMargin, 0.95 * cm, "Proyecto 2 | Reporte ejecutivo")
    canvas.drawRightString(A4[0] - doc.rightMargin, 0.95 * cm, f"Página {doc.page}")
    canvas.restoreState()


def build_pdf():
    styles = getSampleStyleSheet()
    title = ParagraphStyle("ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=23, leading=28, textColor=colors.HexColor("#17365D"), alignment=TA_CENTER, spaceAfter=9)
    subtitle = ParagraphStyle("Subtitle", parent=styles["Normal"], fontName="Helvetica", fontSize=11, leading=15, textColor=colors.HexColor("#425466"), alignment=TA_CENTER, spaceAfter=22)
    heading = ParagraphStyle("Heading", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=14, leading=18, textColor=colors.HexColor("#17365D"), spaceBefore=13, spaceAfter=7)
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontName="Helvetica", fontSize=10, leading=14.5, alignment=TA_JUSTIFY, spaceAfter=8)
    small = ParagraphStyle("Small", parent=body, fontSize=8.5, leading=11, alignment=TA_LEFT, spaceAfter=5)
    callout = ParagraphStyle("Callout", parent=body, fontName="Helvetica-Bold", textColor=colors.HexColor("#17365D"), backColor=colors.HexColor("#EAF2F8"), borderColor=colors.HexColor("#B8CCE4"), borderWidth=0.5, borderPadding=8, spaceBefore=4, spaceAfter=12)

    document = SimpleDocTemplate(str(OUTPUT), pagesize=A4, rightMargin=2.0 * cm, leftMargin=2.0 * cm, topMargin=1.8 * cm, bottomMargin=2.0 * cm, title="Reporte Ejecutivo - Proyecto 2", author="Proyecto 2")
    story = [
        Spacer(1, 1.2 * cm),
        Paragraph("Reporte Ejecutivo", title),
        Paragraph("Proyecto 2 | Priorización de riesgo AML con modelado secuencial", subtitle),
        Paragraph("Decisión clave: el prototipo demuestra valor para priorizar investigación, pero la evidencia actual no justifica una operación productiva ni un reclamo de tiempo T4.", callout),
    ]

    for section, paragraphs in BODY:
        story.append(Paragraph(section, heading))
        for text in paragraphs:
            story.append(Paragraph(text, body))
        if section == "Resultados de ablación y lectura operativa":
            headers = ["Configuración", "AUPRC", "ROC-AUC", "P@1%", "R@1%"]
            rows = [
                ["Base", "0.351 ± 0.004", "0.867 ± 0.002", "0.663 ± 0.017", "0.274 ± 0.007"],
                ["Etapa A", "0.060 ± 0.001", "0.734 ± 0.003", "0.073 ± 0.006", "0.030 ± 0.003"],
                ["Codificador congelado", "0.091 ± 0.014", "0.717 ± 0.013", "0.190 ± 0.050", "0.079 ± 0.020"],
                ["Ajuste fino", "0.341 ± 0.012", "0.870 ± 0.008", "0.634 ± 0.017", "0.262 ± 0.007"],
                ["Ajuste fino + reconstrucción", "0.371 ± 0.013", "0.901 ± 0.009", "0.623 ± 0.017", "0.258 ± 0.007"],
            ]
            table = Table([headers] + rows, colWidths=[4.4 * cm, 2.35 * cm, 2.35 * cm, 2.25 * cm, 2.25 * cm], repeatRows=1)
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("LEADING", (0, 0), (-1, -1), 10),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B8CCE4")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F8FB")]),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            story.append(Spacer(1, 4))
            story.append(KeepTogether([table, Spacer(1, 10)]))

    story.extend([PageBreak(), Paragraph("Referencias", heading)])
    for reference in REFERENCES:
        story.append(Paragraph(reference, small))
    story.extend([Spacer(1, 8), Paragraph("Declaración de uso de IA", heading), Paragraph(AI_DECLARATION, small)])
    document.build(story, onFirstPage=page_number, onLaterPages=page_number)


if __name__ == "__main__":
    build_pdf()
    body_text = " ".join(text for _, paragraphs in BODY for text in paragraphs)
    print(f"PDF created: {OUTPUT}")
    print(f"Body words: {word_count(body_text)}")
    print(f"AI declaration words: {word_count(AI_DECLARATION)}")
