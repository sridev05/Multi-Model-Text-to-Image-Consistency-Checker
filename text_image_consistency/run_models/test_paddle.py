from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='en')

result = ocr.ocr(r"C:\Users\vishn\Documents\Multi-Model-Text-to-Image-Consistency-Checker\text_image_consistency\data\images\image.png", cls=True)

print(result)