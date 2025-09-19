

将这个OCR识别的文件改成一个简单的API-server，同一时刻仅允许单个OCR处理请求。阅读dots_ocr/utils/image_utils.py文件，获取image要用dots_ocr/utils/image_utils.py。
用flask，并且用json格式的请求和响应。注意要支持steam方式的响应。当请求的参数里stream=True的时候，模仿ollama的返回格式，每一行返回一个json那种。
