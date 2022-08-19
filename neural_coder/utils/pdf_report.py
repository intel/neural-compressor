# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# from reportlab.pdfgen import canvas
# from reportlab.platypus import (
#     SimpleDocTemplate, Paragraph, PageBreak, Image, Spacer, Table, TableStyle)
# from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_CENTER, TA_JUSTIFY
# from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
# from reportlab.lib.pagesizes import LETTER, inch
# from reportlab.graphics.shapes import Line, LineShape, Drawing, String
# from reportlab.lib.colors import Color


# class FooterCanvas(canvas.Canvas):

#     def __init__(self, *args, **kwargs):
#         canvas.Canvas.__init__(self, *args, **kwargs)
#         self.pages = []
#         self.width, self.height = LETTER

#     def showPage(self):
#         self.pages.append(dict(self.__dict__))
#         self._startPage()

#     def save(self):
#         page_count = len(self.pages)
#         for page in self.pages:
#             self.__dict__.update(page)
#             if (self._pageNumber > 1):
#                 self.draw_canvas(page_count)
#             canvas.Canvas.showPage(self)
#         canvas.Canvas.save(self)

#     def draw_canvas(self, page_count):
#         page = "Page %s of %s" % (self._pageNumber, page_count)
#         x = 128
#         self.saveState()
#         self.setStrokeColorRGB(0, 0, 0)
#         self.setLineWidth(0.5)
#         self.line(20, 740, LETTER[0] - 50, 740)
#         self.line(66, 78, LETTER[0] - 66, 78)
#         self.setFont('Times-Roman', 10)
#         self.drawString(LETTER[0] - x, 65, page)
#         self.restoreState()


# class PDFReport:

#     def __init__(self,
#                  path,
#                  list_optimization_set_top3,
#                  list_performance_top3,
#                  original_model_ranking,
#                  original_model_performance,
#                  list_config_best_ncpi,
#                  list_config_best_nins,
#                  list_config_best_bs,
#                  list_config_best_performance,
#                  TCO_unit_pricing,  # 2.448
#                  cloud_vendor,  # "AWS"
#                  cloud_instance_type,  # "c6i"
#                  ):
#         self.path = path
#         self.styleSheet = getSampleStyleSheet()
#         self.elements = []

#         # from superbench results
#         self.list_optimization_set_top3 = list_optimization_set_top3
#         self.list_performance_top3 = list_performance_top3
#         self.original_model_ranking = original_model_ranking
#         self.original_model_performance = original_model_performance

#         self.list_config_best_ncpi = list_config_best_ncpi
#         self.list_config_best_nins = list_config_best_nins
#         self.list_config_best_bs = list_config_best_bs
#         self.list_config_best_performance = list_config_best_performance

#         # from Cloud
#         self.TCO_unit_pricing = TCO_unit_pricing
#         self.cloud_vendor = cloud_vendor
#         self.cloud_instance_type = cloud_instance_type

#         # colors
#         self.IntelBlueDark = Color((0 / 255), (74 / 255), (134 / 255), 1)
#         self.IntelBlueLight = Color((0 / 255), (178 / 255), (227 / 255), 1)
#         self.IntelBlueReg = Color((0 / 255), (104 / 255), (181 / 255), 1)

#         self.nextPagesHeader(True)
#         self.MainpageMaker()

#         self.doc = SimpleDocTemplate(path, pagesize=LETTER)
#         self.doc.multiBuild(self.elements, canvasmaker=FooterCanvas)

#     def nextPagesHeader(self, isSecondPage):
#         if isSecondPage:
#             psHeaderText = ParagraphStyle(
#                 'Hed0', fontSize=20, alignment=TA_LEFT, borderWidth=3, textColor=self.IntelBlueDark)
#             text = '<b>AI Deployment Report</b>'
#             paragraphReportHeader = Paragraph(text, psHeaderText)
#             self.elements.append(paragraphReportHeader)

#             spacer = Spacer(10, 15)
#             self.elements.append(spacer)

#             d = Drawing(500, 1)
#             line = Line(0, 0, 460, 0)
#             line.strokeColor = self.IntelBlueReg
#             line.strokeWidth = 2.5
#             d.add(line)
#             self.elements.append(d)

#             spacer = Spacer(10, 2)
#             self.elements.append(spacer)

#             d = Drawing(500, 1)
#             line = Line(0, 0, 460, 0)
#             line.strokeColor = self.IntelBlueReg
#             line.strokeWidth = 1
#             d.add(line)
#             self.elements.append(d)

#             spacer = Spacer(10, 15)
#             self.elements.append(spacer)

#     def MainpageMaker(self):
#         # Optimization Table

#         psHeaderText = ParagraphStyle(
#             'Hed0', fontSize=16, alignment=TA_LEFT, borderWidth=3, textColor=self.IntelBlueDark)
#         text = '<b>Best Optimization (Default Config)</b>'
#         paragraphReportHeader = Paragraph(text, psHeaderText)
#         self.elements.append(paragraphReportHeader)

#         spacer = Spacer(5, 11)
#         self.elements.append(spacer)

#         psHeaderText = ParagraphStyle(
#             'Hed0', fontSize=12, alignment=TA_JUSTIFY)
#         text = "Congratulations! You are able to boost deployment performance up to " \
#             + "<u><b>" + str(round(self.list_performance_top3[0] / self.original_model_performance, 1)) + \
#             "</b></u>" + "<u><b>X</b></u> on your model with the most performant "
#         "Deep Learning acceleration optimization set."
#         paragraphReportHeader = Paragraph(text, psHeaderText)
#         self.elements.append(paragraphReportHeader)

#         spacer = Spacer(10, 15)
#         self.elements.append(spacer)

#         d = []
#         textData = ["Ranking", "Optimization Set", "Performance (sample/sec)"]

#         fontSize = 8
#         centered = ParagraphStyle(name="centered", alignment=TA_LEFT)
#         for text in textData:
#             ptext = "<font size='%s'><b>%s</b></font>" % (fontSize, text)
#             titlesTable = Paragraph(ptext, centered)
#             d.append(titlesTable)

#         data = [d]
#         formattedLineData = []

#         alignStyle = [ParagraphStyle(name="01", alignment=TA_LEFT),
#                       ParagraphStyle(name="02", alignment=TA_LEFT),
#                       ParagraphStyle(name="03", alignment=TA_LEFT), ]

#         list_ranking = [1, 2, 3]
#         list_ranking.append(self.original_model_ranking)
#         list_optimization_set = self.list_optimization_set_top3
#         list_optimization_set.append(['Default'])
#         list_performance = self.list_performance_top3
#         list_performance.append(self.original_model_performance)

#         list_optimization_set_display = []
#         for i in list_optimization_set:
#             i = str(i).replace("[", "").replace("]", "").replace(
#                 "'", "").replace('"', "").replace(",", " +")
#             i = i.replace("pytorch_", "")
#             i = i.replace(
#                 "inc_static_quant_fx", "Intel Neural Compressor Post-Training Static Quantization (FX)")
#             i = i.replace(
#                 "inc_static_quant_ipex", "Intel Neural Compressor Post-Training Static Quantization (IPEX)")
#             i = i.replace(
#                 "inc_dynamic_quant", "Intel Neural Compressor Post-Training Dynamic Quantization")
#             i = i.replace("channels_last", "Channels Last")
#             i = i.replace("ipex_fp32", "Intel Extension for PyTorch FP32")
#             i = i.replace("ipex_bf16", "Intel Extension for PyTorch BF16")
#             i = i.replace("ipex_bf16", "Intel Extension for PyTorch BF16")
#             i = i.replace("ipex_int8_static_quant",
#                           "Intel Extension for PyTorch INT8 Static Quantization")
#             i = i.replace("ipex_int8_dynamic_quant",
#                           "Intel Extension for PyTorch INT8 Dynamic Quantization")
#             i = i.replace("torchdynamo_", "TorchDynamo ")
#             i = i.replace("jit_script", "JIT Script")
#             i = i.replace("jit_trace", "JIT Trace")
#             i = i.replace("_ofi", " with JIT Optimize-for-Inference")
#             i = i.replace("mixed_precision_cpu",
#                           "Automatic Mixed Precision (CPU)")
#             i = i.replace("mixed_precision_cuda",
#                           "Automatic Mixed Precision (CUDA)")

#             list_optimization_set_display.append(i)
#         for i in range(len(list_ranking)):
#             lineData = [list_ranking[i], list_optimization_set_display[i], int(
#                 list_performance[i])]
#             columnNumber = 0
#             for item in lineData:
#                 ptext = "<font size='%s'>%s</font>" % (fontSize - 1, item)
#                 p = Paragraph(ptext, alignStyle[columnNumber])
#                 formattedLineData.append(p)
#                 columnNumber = columnNumber + 1
#             data.append(formattedLineData)
#             formattedLineData = []

#         # totalRow = ["The most performant optimization set is: " + list_optimization_set_display[0]]
#         # for item in totalRow:
#         #     ptext = "<font size='%s'>%s</font>" % (fontSize-1, item)
#         #     p = Paragraph(ptext, alignStyle[1])
#         #     formattedLineData.append(p)
#         # data.append(formattedLineData)

#         table = Table(data, colWidths=[50, 290, 120])
#         tStyle = TableStyle([
#             ('ALIGN', (0, 0), (0, -1), 'LEFT'),
#             ('VALIGN', (0, 0), (-1, -1), 'TOP'),
#             ("ALIGN", (1, 0), (1, -1), 'RIGHT'),
#             ('LINEABOVE', (0, 0), (-1, -1), 1, self.IntelBlueLight),
#             ('BACKGROUND', (0, 0), (-1, 0), self.IntelBlueLight),
#             ('BACKGROUND', (0, 5), (0, 0), Color(
#                 (225 / 255), (225 / 255), (225 / 255), 1)),
#             ('BACKGROUND', (3, 5), (0, 3), Color(
#                 (240 / 255), (240 / 255), (240 / 255), 1)),
#             ('LINEBELOW', (0, 0), (-1, -1), 1, self.IntelBlueLight),
#         ])
#         table.setStyle(tStyle)
#         self.elements.append(table)

#         spacer = Spacer(10, 2)
#         self.elements.append(spacer)
#         psHeaderText = ParagraphStyle('Hed0', fontSize=7, alignment=TA_JUSTIFY)
#         from neural_coder.utils.cpu_info import get_num_cpu_cores
#         text = "*(1) All optimization sets are measured with default configuration " \
#                "(single instance on single socket), among which the top 3 performant ones are displayed. " \
#                "(2) This report evaluates performance only (accuracy under development)."
#         paragraphReportHeader = Paragraph(text, psHeaderText)
#         self.elements.append(paragraphReportHeader)

#         spacer = Spacer(10, 10)
#         self.elements.append(spacer)

#         # Configuration Table

#         psHeaderText = ParagraphStyle(
#             'Hed0', fontSize=16, alignment=TA_LEFT, borderWidth=3, textColor=self.IntelBlueDark)
#         text = '<b>Best Optimization (Sweeping Configs)</b>'
#         paragraphReportHeader = Paragraph(text, psHeaderText)
#         self.elements.append(paragraphReportHeader)

#         spacer = Spacer(5, 11)
#         self.elements.append(spacer)

#         psHeaderText = ParagraphStyle(
#             'Hed0', fontSize=12, alignment=TA_JUSTIFY)
#         text = "For the most performant optimization set, " \
#             "you can further boost your deployment performance to up to " \
#             + "<u><b>" + \
#             str(round(self.list_config_best_performance[0] / self.original_model_performance, 1)) + "</b></u>" + \
#             "<u><b>X</b></u> if using the most performant deployment configuration according to our sweeping result."
#         paragraphReportHeader = Paragraph(text, psHeaderText)
#         self.elements.append(paragraphReportHeader)

#         spacer = Spacer(10, 15)
#         self.elements.append(spacer)

#         d = []
#         textData = ["Category", "Num Intances",
#                     "Num Cores Per Instance", "BS", "Performance (sample/sec)"]

#         fontSize = 8
#         centered = ParagraphStyle(name="centered", alignment=TA_LEFT)
#         for text in textData:
#             ptext = "<font size='%s'><b>%s</b></font>" % (fontSize, text)
#             titlesTable = Paragraph(ptext, centered)
#             d.append(titlesTable)

#         data = [d]
#         formattedLineData = []

#         alignStyle = [ParagraphStyle(name="01", alignment=TA_LEFT),
#                       ParagraphStyle(name="02", alignment=TA_LEFT),
#                       ParagraphStyle(name="03", alignment=TA_LEFT),
#                       ParagraphStyle(name="04", alignment=TA_LEFT),
#                       ParagraphStyle(name="05", alignment=TA_LEFT)]

#         list_category = ["Throughput",
#                          "Throughput based on P50-Latency",
#                          "Throughput based on P90-Latency",
#                          "Throughput based on P99-Latency"]

#         for i in range(len(list_category)):
#             lineData = [list_category[i],
#                         self.list_config_best_ncpi[i],
#                         self.list_config_best_nins[i],
#                         self.list_config_best_bs[i],
#                         int(self.list_config_best_performance[i])]
#             columnNumber = 0
#             for item in lineData:
#                 ptext = "<font size='%s'>%s</font>" % (fontSize - 1, item)
#                 p = Paragraph(ptext, alignStyle[columnNumber])
#                 formattedLineData.append(p)
#                 columnNumber = columnNumber + 1
#             data.append(formattedLineData)
#             formattedLineData = []

#         table = Table(data, colWidths=[140, 65, 105, 30, 120])
#         tStyle = TableStyle([
#             ('ALIGN', (0, 0), (0, -1), 'LEFT'),
#             ('VALIGN', (0, 0), (-1, -1), 'TOP'),
#             ("ALIGN", (1, 0), (1, -1), 'RIGHT'),
#             ('LINEABOVE', (0, 0), (-1, -1), 1, self.IntelBlueLight),
#             ('BACKGROUND', (0, 0), (-1, 0), self.IntelBlueLight),
#             ('BACKGROUND', (0, 5), (0, 0), Color(
#                 (235 / 255), (235 / 255), (235 / 255), 1)),
#             ('LINEBELOW', (0, 0), (-1, -1), 1, self.IntelBlueLight),
#         ])
#         table.setStyle(tStyle)
#         self.elements.append(table)

#         spacer = Spacer(10, 2)
#         self.elements.append(spacer)
#         psHeaderText = ParagraphStyle('Hed0', fontSize=7, alignment=TA_JUSTIFY)
#         text = "*Measured on the most performant optimization set (Ranking 1 in above table) " \
#                "by sweeping configurations among batch size, number of instances, and number of cores per instance."
#         paragraphReportHeader = Paragraph(text, psHeaderText)
#         self.elements.append(paragraphReportHeader)

#         spacer = Spacer(10, 10)
#         self.elements.append(spacer)

#         # TCO

#         psHeaderText = ParagraphStyle(
#             'Hed0', fontSize=16, alignment=TA_LEFT, borderWidth=3, textColor=self.IntelBlueDark)
#         text = '<b>Cost Saving</b>'
#         paragraphReportHeader = Paragraph(text, psHeaderText)
#         self.elements.append(paragraphReportHeader)

#         # spacer = Spacer(5, 11)
#         # self.elements.append(spacer)

#         # psHeaderText = ParagraphStyle('Hed0', fontSize=12, alignment=TA_LEFT)
#         # text = "Samples per Dollar:"
#         # paragraphReportHeader = Paragraph(text, psHeaderText)
#         # self.elements.append(paragraphReportHeader)

#         spacer = Spacer(5, 11)
#         self.elements.append(spacer)

#         from reportlab.graphics.shapes import Drawing
#         from reportlab.graphics.charts.barcharts import HorizontalBarChart
#         from reportlab.graphics.charts.textlabels import Label
#         drawing = Drawing(0, 80)
#         TCO_raw = int(self.original_model_performance /
#                       (self.TCO_unit_pricing / 3600))
#         TCO_accelerated = int(
#             self.list_config_best_performance[0] / (self.TCO_unit_pricing / 3600))
#         data = [
#             (TCO_raw, TCO_accelerated)
#         ]
#         bc = HorizontalBarChart()
#         bc.valueAxis.labels.fontName = 'Helvetica'
#         bc.barLabels.fontName = 'Helvetica'
#         bc.categoryAxis.labels.fontName = 'Helvetica'
#         bc.x = 180
#         bc.y = 10
#         bc.height = 65
#         bc.width = 250
#         bc.data = data
#         bc.strokeColor = Color((235 / 255), (235 / 255), (235 / 255), 1)
#         bc.bars[(0, 0)].fillColor = self.IntelBlueDark
#         bc.bars[(0, 1)].fillColor = self.IntelBlueLight
#         bc.valueAxis.valueMin = 0
#         import math
#         bc.valueAxis.valueMax = max(TCO_accelerated, TCO_raw)
#         # bc.valueAxis.valueStep = math.ceil(TCO_raw)/10
#         bc.categoryAxis.labels.boxAnchor = 'ne'
#         bc.categoryAxis.labels.dx = -10
#         bc.categoryAxis.labels.dy = 4
#         bc.categoryAxis.labels.angle = 0
#         bc.categoryAxis.categoryNames = ["Default: " + str(format(TCO_raw, ',')) + " (sample/$)",
#                                          "Optimized: " + str(format(TCO_accelerated, ',')) + " (sample/$)"]
#         drawing.add(bc)
        
#         # add label
#         lab = Label()
#         lab.setOrigin(0, 0)
#         lab.boxAnchor = 'ne'
#         lab.angle = 0
#         lab.dx = 420
#         lab.dy = 35
#         lab.fontName = 'Helvetica-Bold'
#         lab.fontSize = 13
#         lab.setText(str(round(self.list_config_best_performance[0] / self.original_model_performance, 1)) + 'X')
#         drawing.add(lab)

#         self.elements.append(drawing)

#         spacer = Spacer(10, 9)
#         self.elements.append(spacer)
#         psHeaderText = ParagraphStyle('Hed0', fontSize=8, alignment=TA_JUSTIFY)
#         text = "*Sample/$ is calculated based on " + self.cloud_vendor + \
#             " " + self.cloud_instance_type + " instance and on-demand price."
#         paragraphReportHeader = Paragraph(text, psHeaderText)
#         self.elements.append(paragraphReportHeader)
