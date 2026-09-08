# -*- coding: utf-8 -*-
import re
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFont

try:
    import tensorflow as tf
except Exception:
    tf = None

try:
    from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter
except Exception:
    TFLiteInterpreter = None

from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.metrics import dp
from kivy.uix.image import Image
from kivy.uix.slider import Slider
from kivy.core.clipboard import Clipboard

from kivymd.app import MDApp
from kivymd.uix.screen import MDScreen
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDFillRoundFlatIconButton, MDRoundFlatIconButton, MDFlatButton
from kivymd.uix.label import MDLabel
from kivymd.uix.dialog import MDDialog
from kivymd.uix.scrollview import MDScrollView
from kivymd.uix.card import MDCard

try:
    from plyer import filechooser
except Exception:
    filechooser = None


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best.tflite"
FONT_PATH = BASE_DIR / "fonts" / "NotoSansLycian-Regular.ttf"

LYCIAN_CHARS = ['𐊀', '𐊁', '𐊂', '𐊃', '𐊄', '𐊅', '𐊆', '𐊇', '𐊈', '𐊉', '𐊊', '𐊋', '𐊌', '𐊍', '𐊎', '𐊏', '𐊐', '𐊑', '𐊒', '𐊓', '𐊔', '𐊕', '𐊖', '𐊗', '𐊘', '𐊙', '𐊚', '𐊛']
LYCIAN_CODEPOINTS = {i: ord(ch) for i, ch in enumerate(LYCIAN_CHARS)}
UNICODE_TO_LYCIAN = {ord(ch): ch for ch in LYCIAN_CHARS}
WORD_DIVIDER_CLASS = 30

DEFAULT_CONF = 0.30
DEFAULT_IOU = 0.50
DEFAULT_MAX_DET = 40
DEFAULT_FPS = 6
DEFAULT_BOX_THICKNESS = 3
DEFAULT_LABEL_SIZE = 18

_FONT_CACHE = {}


def get_font(size):
    key = (str(FONT_PATH), int(size))
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    try:
        _FONT_CACHE[key] = ImageFont.truetype(str(FONT_PATH), int(size))
    except Exception:
        _FONT_CACHE[key] = ImageFont.load_default()
    return _FONT_CACHE[key]


def read_image_unicode(path):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def letterbox(image, size=256):
    h, w = image.shape[:2]
    ratio = min(size / max(h, 1), size / max(w, 1))
    nw, nh = int(round(w * ratio)), int(round(h * ratio))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw = (size - nw) / 2.0
    dh = (size - nh) / 2.0
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    out = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(242, 237, 244)
    )
    return out, ratio, dw, dh


def xywh_to_xyxy(boxes):
    boxes = np.asarray(boxes, dtype=np.float32)
    out = np.empty_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


def iou(one, many):
    if len(many) == 0:
        return np.empty((0,), dtype=np.float32)
    x1 = np.maximum(one[0], many[:, 0])
    y1 = np.maximum(one[1], many[:, 1])
    x2 = np.minimum(one[2], many[:, 2])
    y2 = np.minimum(one[3], many[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    a1 = max(0, one[2] - one[0]) * max(0, one[3] - one[1])
    a2 = np.maximum(0, many[:, 2] - many[:, 0]) * np.maximum(0, many[:, 3] - many[:, 1])
    return inter / (a1 + a2 - inter + 1e-7)


def nms(boxes, scores, class_ids, threshold, max_det):
    keep = []
    for cls in np.unique(class_ids):
        idx = np.where(class_ids == cls)[0]
        idx = idx[np.argsort(scores[idx])[::-1]]
        while len(idx):
            cur = int(idx[0])
            keep.append(cur)
            if len(idx) == 1:
                break
            rest = idx[1:]
            idx = rest[iou(boxes[cur], boxes[rest]) <= threshold]
    keep.sort(key=lambda i: float(scores[i]), reverse=True)
    return keep[:int(max_det)]


def unicode_to_character(value):
    """Convert model Unicode/class output into the actual Lycian character."""
    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, str):
        s = value.strip()
        if s in LYCIAN_CHARS:
            return s
        m = re.fullmatch(r"U\+([0-9A-Fa-f]+)", s)
        if m:
            try:
                return chr(int(m.group(1), 16))
            except ValueError:
                return "?"
        m = re.fullmatch(r"0x([0-9A-Fa-f]+)", s)
        if m:
            try:
                return chr(int(m.group(1), 16))
            except ValueError:
                return "?"
        if s.isdigit():
            value = int(s)

    if isinstance(value, (int, np.integer)):
        v = int(value)
        if 0 <= v < len(LYCIAN_CHARS):
            return LYCIAN_CHARS[v]
        if v in UNICODE_TO_LYCIAN:
            return UNICODE_TO_LYCIAN[v]

    return "?"


def model_class_to_unicode(class_id):
    cid = int(class_id)
    if 0 <= cid < len(LYCIAN_CHARS):
        return f"U+{ord(LYCIAN_CHARS[cid]):04X}"
    return "UNKNOWN"


def group_lines(detections):
    lines = []
    for det in sorted(detections, key=lambda d: d["cy"]):
        placed = False
        for line in lines:
            avg_y = np.mean([x["cy"] for x in line])
            avg_h = np.mean([x["height"] for x in line])
            if abs(det["cy"] - avg_y) <= max(10, avg_h * 0.6):
                line.append(det)
                placed = True
                break
        if not placed:
            lines.append([det])
    lines.sort(key=lambda line: np.mean([x["cy"] for x in line]))
    return lines


def detections_to_text(detections):
    """Post-processing: Unicode/class output -> real Lycian Unicode characters."""
    output_lines = []
    for line in group_lines(detections):
        line = sorted(line, key=lambda d: d["cx"])
        text = []
        for det in line:
            if int(det["class_id"]) == WORD_DIVIDER_CLASS:
                text.append(" ")
            else:
                text.append(unicode_to_character(det["unicode"]))
        value = re.sub(r"\s+", " ", "".join(text)).strip()
        if value:
            output_lines.append(value)
    return "\n".join(output_lines)


def decode_output(raw, frame_shape, conf_threshold, iou_threshold, max_det, model_size=256):
    if raw is None:
        return []
    arr = np.asarray(raw, dtype=np.float32)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        return []

    if arr.shape[0] in (4 + len(LYCIAN_CHARS), 5 + len(LYCIAN_CHARS)):
        arr = arr.T
    if arr.shape[1] not in (4 + len(LYCIAN_CHARS), 5 + len(LYCIAN_CHARS)):
        return []

    has_obj = arr.shape[1] == 5 + len(LYCIAN_CHARS)
    boxes = arr[:, :4].copy()
    start = 5 if has_obj else 4
    cls_scores = arr[:, start:]

    if np.min(cls_scores) < 0 or np.max(cls_scores) > 1:
        cls_scores = 1 / (1 + np.exp(-np.clip(cls_scores, -50, 50)))

    classes = np.argmax(cls_scores, axis=1).astype(np.int32)
    scores = cls_scores[np.arange(len(classes)), classes]

    if has_obj:
        obj = arr[:, 4]
        if np.min(obj) < 0 or np.max(obj) > 1:
            obj = 1 / (1 + np.exp(-np.clip(obj, -50, 50)))
        scores *= obj

    keep = scores >= float(conf_threshold)
    if not np.any(keep):
        return []

    boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
    boxes = xywh_to_xyxy(boxes)

    if np.max(np.abs(boxes)) <= 2:
        boxes[:, [0, 2]] *= model_size
        boxes[:, [1, 3]] *= model_size

    h, w = frame_shape[:2]
    _, ratio, dw, dh = letterbox(np.zeros((h, w, 3), dtype=np.uint8), model_size)
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / max(ratio, 1e-9)
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / max(ratio, 1e-9)
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h - 1)

    selected = nms(boxes, scores, classes, iou_threshold, max_det)
    result = []

    for idx in selected:
        x1, y1, x2, y2 = boxes[idx]
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        cid = int(classes[idx])
        unicode_value = model_class_to_unicode(cid)
        result.append({
            "class_id": cid,
            "unicode": unicode_value,
            "character": unicode_to_character(unicode_value),
            "confidence": float(scores[idx]),
            "x1": float(x1), "y1": float(y1),
            "x2": float(x2), "y2": float(y2),
            "cx": float((x1 + x2) / 2),
            "cy": float((y1 + y2) / 2),
            "width": float(x2 - x1),
            "height": float(y2 - y1),
        })
    return result


def draw_detections(frame, detections, show_boxes=True, show_labels=True,
                    show_confidence=True, thickness=3, label_size=18):
    out = frame.copy()
    rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    pil = PILImage.fromarray(rgb)
    draw = ImageDraw.Draw(pil, "RGBA")
    font = get_font(label_size)

    for det in detections:
        x1, y1, x2, y2 = [int(round(det[k])) for k in ("x1", "y1", "x2", "y2")]
        ch = det["character"]
        conf = int(round(det["confidence"] * 100))

        if show_boxes:
            draw.rounded_rectangle(
                [x1, y1, x2, y2],
                radius=6,
                outline=(224, 102, 178, 255),
                width=max(1, int(thickness))
            )

        parts = []
        if show_labels:
            parts.append(ch)
        if show_confidence:
            parts.append(f"{conf}%")
        if parts:
            label = "  ".join(parts)
            bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            pad = 7
            top = max(0, y1 - th - pad * 2)
            draw.rounded_rectangle(
                [max(0, x1), top, x1 + tw + pad * 2, top + th + pad * 2],
                radius=8,
                fill=(117, 73, 125, 225)
            )
            draw.text((max(0, x1) + pad, top + pad), label,
                      font=font, fill=(255, 255, 255, 255))

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def make_interpreter():
    if not MODEL_PATH.exists():
        print("[Lycian OCR] Missing model:", MODEL_PATH)
        return None
    try:
        if TFLiteInterpreter is not None:
            itp = TFLiteInterpreter(model_path=str(MODEL_PATH), num_threads=2)
        elif tf is not None:
            itp = tf.lite.Interpreter(model_path=str(MODEL_PATH), num_threads=2)
        else:
            return None
        itp.allocate_tensors()
        return itp
    except Exception as exc:
        print("[Lycian OCR] Model load error:", repr(exc))
        return None


def infer(interpreter, frame, size=256):
    padded, _, _, _ = letterbox(frame, size)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    info = interpreter.get_input_details()[0]
    shape = tuple(int(x) for x in info["shape"])
    tensor = np.expand_dims(rgb, 0) if shape[-1] == 3 else np.expand_dims(np.transpose(rgb, (2, 0, 1)), 0)

    if info["dtype"] == np.float32:
        inp = tensor.astype(np.float32)
    elif info["dtype"] in (np.uint8, np.int8):
        scale, zero = info.get("quantization", (0, 0))
        inp = np.round(tensor / scale + zero).astype(info["dtype"])
    else:
        inp = tensor.astype(info["dtype"])

    interpreter.set_tensor(info["index"], inp)
    interpreter.invoke()

    out_info = interpreter.get_output_details()[0]
    output = interpreter.get_tensor(out_info["index"])
    if out_info["dtype"] in (np.uint8, np.int8):
        scale, zero = out_info.get("quantization", (0, 0))
        if scale:
            output = (output.astype(np.float32) - zero) * scale
    return output


class LycianOCRApp(MDApp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.interpreter = None
        self.capture = None
        self.running = True
        self.lock = threading.Lock()

        self.source = "camera"
        self.cached_image = None
        self.latest_frame = None
        self.frozen_frame = None

        self.last_text = ""
        self.last_count = 0

        self.conf = DEFAULT_CONF
        self.iou_threshold = DEFAULT_IOU
        self.max_det = DEFAULT_MAX_DET
        self.fps = DEFAULT_FPS
        self.box_thickness = DEFAULT_BOX_THICKNESS
        self.label_size = DEFAULT_LABEL_SIZE

        self.show_boxes = True
        self.show_labels = True
        self.show_confidence = True
        self.settings_dialog = None
        self.text_dialog = None

    def build(self):
        self.theme_cls.theme_style = "Light"
        self.theme_cls.primary_palette = "Pink"
        self.title = "Lycian OCR"

        self.interpreter = make_interpreter()

        screen = MDScreen(md_bg_color=(0.98, 0.95, 0.98, 1))
        root = MDBoxLayout(
            orientation="vertical",
            padding=dp(12),
            spacing=dp(10)
        )

        header = MDCard(
            size_hint_y=None, height=dp(82),
            radius=[dp(24)], elevation=3,
            padding=dp(12),
            md_bg_color=(1.0, 0.96, 0.99, 1)
        )
        h = MDBoxLayout(orientation="horizontal", spacing=dp(8))
        title_box = MDBoxLayout(orientation="vertical")
        title_box.add_widget(MDLabel(
            text="Lycian OCR",
            font_style="H6",
            bold=True,
            theme_text_color="Custom",
            text_color=(0.30, 0.17, 0.30, 1),
            adaptive_height=True
        ))
        title_box.add_widget(MDLabel(
            text="AI Vision for the Lycian Script",
            font_style="Caption",
            theme_text_color="Custom",
            text_color=(0.55, 0.43, 0.57, 1),
            adaptive_height=True
        ))
        h.add_widget(title_box)

        self.status = MDLabel(
            text="READY",
            halign="center",
            size_hint_x=None, width=dp(110),
            bold=True,
            theme_text_color="Custom",
            text_color=(0.75, 0.29, 0.55, 1)
        )
        h.add_widget(self.status)

        sbtn = MDRoundFlatIconButton(icon="cog-outline", text="Settings")
        sbtn.bind(on_release=self.open_settings)
        h.add_widget(sbtn)
        header.add_widget(h)
        root.add_widget(header)

        self.viewer_card = MDCard(
            radius=[dp(26)], elevation=3,
            padding=dp(8), md_bg_color=(1, 1, 1, 1)
        )
        self.viewer = Image(allow_stretch=True, keep_ratio=True)
        self.viewer_card.add_widget(self.viewer)

        info = MDBoxLayout(
            orientation="horizontal",
            size_hint_y=None, height=dp(34)
        )
        self.count = MDLabel(
            text="0 detected",
            bold=True,
            theme_text_color="Custom",
            text_color=(0.44, 0.31, 0.45, 1)
        )
        self.preview = MDLabel(
            text="Ready for a Lycian image",
            halign="right",
            shorten=True,
            shorten_from="left",
            theme_text_color="Custom",
            text_color=(0.55, 0.44, 0.56, 1)
        )
        info.add_widget(self.count)
        info.add_widget(self.preview)
        self.viewer_card.add_widget(info)
        root.add_widget(self.viewer_card)

        filters = MDCard(
            size_hint_y=None, height=dp(54),
            radius=[dp(20)], elevation=1,
            padding=dp(6), md_bg_color=(1.0, 0.98, 1.0, 1)
        )
        fr = MDBoxLayout(orientation="horizontal", spacing=dp(5))

        b = MDRoundFlatIconButton(icon="vector-square", text="Boxes")
        b.bind(on_release=lambda *_: self.toggle("boxes"))
        l = MDRoundFlatIconButton(icon="alphabetical", text="Characters")
        l.bind(on_release=lambda *_: self.toggle("labels"))
        c = MDRoundFlatIconButton(icon="percent-outline", text="Confidence")
        c.bind(on_release=lambda *_: self.toggle("confidence"))
        t = MDRoundFlatIconButton(icon="text-box-outline", text="Text")
        t.bind(on_release=self.open_text_view)

        for x in (b, l, c, t):
            fr.add_widget(x)
        filters.add_widget(fr)
        root.add_widget(filters)

        actions = MDCard(
            size_hint_y=None, height=dp(82),
            radius=[dp(24)], elevation=3,
            padding=dp(10), md_bg_color=(0.995, 0.98, 0.995, 1)
        )
        ar = MDBoxLayout(orientation="horizontal", spacing=dp(8))

        cap = MDFillRoundFlatIconButton(
            icon="camera", text="Capture",
            md_bg_color=(0.82, 0.44, 0.72, 1),
            text_color=(1, 1, 1, 1)
        )
        cap.bind(on_release=self.capture_and_analyze)

        gal = MDFillRoundFlatIconButton(
            icon="image-multiple", text="Gallery",
            md_bg_color=(0.55, 0.39, 0.68, 1),
            text_color=(1, 1, 1, 1)
        )
        gal.bind(on_release=self.open_gallery)

        live = MDFillRoundFlatIconButton(
            icon="camera-wireless-outline", text="Live",
            md_bg_color=(0.97, 0.78, 0.91, 1),
            text_color=(0.40, 0.22, 0.38, 1)
        )
        live.bind(on_release=self.switch_to_camera)

        for x in (cap, gal, live):
            ar.add_widget(x)
        actions.add_widget(ar)
        root.add_widget(actions)

        screen.add_widget(root)

        threading.Thread(target=self.inference_loop, daemon=True).start()
        return screen

    def update_status(self):
        label = "LIVE CAMERA" if self.source == "camera" else ("IMAGE" if self.source == "image" else "ANALYZED")
        if self.interpreter is None:
            label += " • MODEL MISSING"
        self.status.text = label

    def toggle(self, name):
        with self.lock:
            if name == "boxes":
                self.show_boxes = not self.show_boxes
            elif name == "labels":
                self.show_labels = not self.show_labels
            elif name == "confidence":
                self.show_confidence = not self.show_confidence

    def open_gallery(self, *_):
        if filechooser is None:
            return
        try:
            filechooser.open_file(
                title="Choose a Lycian image",
                filters=[("Images", "*.jpg"), ("Images", "*.jpeg"), ("Images", "*.png"), ("Images", "*.bmp")],
                on_selection=self.gallery_result
            )
        except Exception as exc:
            print("[Lycian OCR] gallery:", exc)

    def gallery_result(self, selection):
        if not selection:
            return
        image = read_image_unicode(selection[0])
        if image is None:
            return
        with self.lock:
            self.cached_image = image
            self.frozen_frame = None
            self.source = "image"
            self.last_text = ""
            self.last_count = 0

    def switch_to_camera(self, *_):
        with self.lock:
            self.source = "camera"
            self.cached_image = None
            self.frozen_frame = None
            self.last_text = ""

    def capture_and_analyze(self, *_):
        with self.lock:
            if self.source == "camera" and self.latest_frame is not None:
                self.frozen_frame = self.latest_frame.copy()
                self.source = "freeze"
            elif self.source in ("image", "freeze"):
                pass

    def inference_loop(self):
        while self.running:
            start = time.time()
            with self.lock:
                source = self.source
                conf = self.conf
                iou_th = self.iou_threshold
                max_det = self.max_det
                thickness = self.box_thickness
                label_size = self.label_size
                show_boxes = self.show_boxes
                show_labels = self.show_labels
                show_conf = self.show_confidence
                cached = None if self.cached_image is None else self.cached_image.copy()
                frozen = None if self.frozen_frame is None else self.frozen_frame.copy()

            frame = None
            if source == "camera":
                if self.capture is None or not self.capture.isOpened():
                    try:
                        self.capture = cv2.VideoCapture(0)
                    except Exception:
                        self.capture = None
                if self.capture is not None:
                    ok, f = self.capture.read()
                    if ok and f is not None:
                        frame = f
                        with self.lock:
                            self.latest_frame = f.copy()
            elif source == "freeze":
                frame = frozen
            else:
                frame = cached

            if frame is None:
                frame = np.full((480, 720, 3), 248, dtype=np.uint8)
                cv2.putText(frame, "Lycian OCR", (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                            (145, 100, 140), 3, cv2.LINE_AA)
                cv2.putText(frame, "Open an image or enable the camera",
                            (30, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (165, 145, 165), 2, cv2.LINE_AA)
                detections = []
            else:
                detections = []
                if self.interpreter is not None:
                    try:
                        raw = infer(self.interpreter, frame, 256)
                        detections = decode_output(
                            raw, frame.shape, conf, iou_th, max_det
                        )
                    except Exception as exc:
                        print("[Lycian OCR] inference:", repr(exc))

            text = detections_to_text(detections)
            rendered = draw_detections(
                frame, detections,
                show_boxes, show_labels, show_conf,
                thickness, label_size
            )

            with self.lock:
                self.last_text = text
                self.last_count = len(detections)

            Clock.schedule_once(
                lambda dt, img=rendered, n=len(detections), txt=text: self.render_ui(img, n, txt),
                0
            )

            if source == "camera":
                delay = max(0.01, 1 / max(1, self.fps) - (time.time() - start))
            else:
                delay = 0.08
            time.sleep(delay)

    def render_ui(self, frame, count, text):
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            texture = Texture.create(
                size=(frame.shape[1], frame.shape[0]),
                colorfmt="rgb"
            )
            texture.blit_buffer(rgb.tobytes(), colorfmt="rgb", bufferfmt="ubyte")
            texture.flip_vertical()
            self.viewer.texture = texture
            self.count.text = f"{count} detected"
            self.preview.text = text if text else "No Lycian characters detected"
            self.update_status()
        except Exception:
            pass

    def open_text_view(self, *_):
        with self.lock:
            text = self.last_text

        if not text:
            body = MDLabel(
                text="No recognized Lycian text is available yet.",
                halign="center",
                adaptive_height=True
            )
        else:
            holder = MDBoxLayout(
                orientation="vertical",
                size_hint_y=None,
                spacing=dp(8),
                padding=dp(8)
            )
            holder.bind(minimum_height=holder.setter("height"))

            for i, line in enumerate(text.splitlines(), 1):
                row = MDCard(
                    size_hint_y=None, height=dp(68),
                    radius=[dp(16)], elevation=1,
                    padding=dp(8),
                    md_bg_color=(1.0, 0.97, 1.0, 1)
                )
                box = MDBoxLayout(orientation="horizontal", spacing=dp(8))
                box.add_widget(MDLabel(
                    text=f"{i}",
                    size_hint_x=None, width=dp(28),
                    halign="center", bold=True
                ))
                box.add_widget(MDLabel(
                    text=line,
                    font_size="24sp",
                    theme_text_color="Custom",
                    text_color=(0.28, 0.17, 0.31, 1)
                ))
                cp = MDRoundFlatIconButton(icon="content-copy", text="Copy")
                cp.bind(on_release=lambda *_b, x=line: self.copy_text(x))
                box.add_widget(cp)
                row.add_widget(box)
                holder.add_widget(row)

            scroll = MDScrollView(size_hint=(1, None), height=dp(330))
            scroll.add_widget(holder)
            body = scroll

        self.text_dialog = MDDialog(
            title="Recognized Lycian Text",
            type="custom",
            content_cls=body,
            buttons=[
                MDFlatButton(text="Copy All", on_release=lambda *_: self.copy_text(text)),
                MDFlatButton(text="Close", on_release=lambda *_: self.text_dialog.dismiss()),
            ],
        )
        self.text_dialog.open()

    def copy_text(self, text):
        Clipboard.copy(text or "")

    def open_settings(self, *_):
        with self.lock:
            vals = (
                self.conf, self.iou_threshold, self.fps,
                self.max_det, self.box_thickness, self.label_size
            )

        content = MDBoxLayout(
            orientation="vertical",
            spacing=dp(9),
            padding=dp(8),
            size_hint_y=None,
            height=dp(490)
        )

        widgets = []
        specs = [
            ("Confidence", 0.05, 0.95, vals[0], "conf"),
            ("IoU", 0.05, 0.95, vals[1], "iou"),
            ("Camera FPS", 1, 30, vals[2], "fps"),
            ("Max detections", 5, 100, vals[3], "det"),
            ("Box thickness", 1, 6, vals[4], "thick"),
            ("Character size", 10, 32, vals[5], "font"),
        ]

        for title, mn, mx, value, key in specs:
            label = MDLabel(text=f"{title}: {value}", adaptive_height=True)
            slider = Slider(min=mn, max=mx, step=0.05 if isinstance(mn, float) else 1, value=value)
            slider.bind(
                value=lambda inst, val, k=key, l=label, t=title: self.setting_changed(k, val, l, t)
            )
            content.add_widget(label)
            content.add_widget(slider)

        scroll = MDScrollView(size_hint=(1, None), height=dp(410))
        scroll.add_widget(content)

        self.settings_dialog = MDDialog(
            title="Lycian OCR Settings",
            type="custom",
            content_cls=scroll,
            buttons=[
                MDFlatButton(text="Reset", on_release=self.reset_settings),
                MDFlatButton(text="Done", on_release=lambda *_: self.settings_dialog.dismiss()),
            ]
        )
        self.settings_dialog.open()

    def setting_changed(self, key, value, label, title):
        with self.lock:
            if key == "conf":
                self.conf = float(value); shown = f"{self.conf:.2f}"
            elif key == "iou":
                self.iou_threshold = float(value); shown = f"{self.iou_threshold:.2f}"
            elif key == "fps":
                self.fps = int(value); shown = str(self.fps)
            elif key == "det":
                self.max_det = int(value); shown = str(self.max_det)
            elif key == "thick":
                self.box_thickness = int(value); shown = str(self.box_thickness)
            else:
                self.label_size = int(value); shown = str(self.label_size)
        label.text = f"{title}: {shown}"

    def reset_settings(self, *_):
        with self.lock:
            self.conf = DEFAULT_CONF
            self.iou_threshold = DEFAULT_IOU
            self.max_det = DEFAULT_MAX_DET
            self.fps = DEFAULT_FPS
            self.box_thickness = DEFAULT_BOX_THICKNESS
            self.label_size = DEFAULT_LABEL_SIZE
        if self.settings_dialog:
            self.settings_dialog.dismiss()

    def on_stop(self):
        self.running = False
        if self.capture is not None:
            try:
                self.capture.release()
            except Exception:
                pass
        self.capture = None
        self.interpreter = None


if __name__ == "__main__":
    LycianOCRApp().run()
