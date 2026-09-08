# Lycian OCR

A Lycian script OCR application with:
- Camera and gallery input
- Dynamic bounding boxes
- Confidence/IoU controls
- Lycian Unicode post-processing
- Final rendering as actual Lycian characters
- Soft pink/purple feminine UI

## Model
The supplied modified model is included as `best.tflite`.

## Font
Add the project Lycian font as:
`fonts/NotoSansLycian-Regular.ttf`

## Run
```bash
python main.py
```

## Android
Build with:
```bash
buildozer android debug
```
