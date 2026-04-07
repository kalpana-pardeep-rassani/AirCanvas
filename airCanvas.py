"""
AirCanvas - Draw in the air with your index finger!
Compatible with Python 3.10+ and mediapipe 0.10+

HOW TO USE (read the on-screen guide too):
  1. Show your hand to the camera, palm facing the screen.
  2. Point your INDEX FINGER at a toolbar button and HOLD for ~0.4s to activate it.
  3. Point your INDEX FINGER below the toolbar and move to DRAW.
  4. Close your fist (put finger down) to STOP drawing / lift the pen.

Keyboard shortcuts (click the AirCanvas window first):
  H        -> Toggle help overlay
  S        -> Save drawing to Desktop
  C        -> Clear canvas
  ESC / Q  -> Quit
"""

import cv2
import numpy as np
import mediapipe as mp
import os
import time
import urllib.request
from datetime import datetime

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarksConnections

# ── Model auto-download ────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")

def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        print()
        print("  [AirCanvas] First-time setup: downloading AI model (~2 MB)...")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("  [AirCanvas] Model downloaded successfully!")
        except Exception as e:
            print("  [ERROR] Could not download model:", e)
            print("  Please manually download:")
            print("  " + MODEL_URL)
            print("  and save it as 'hand_landmarker.task' in the same folder as airCanvas.py")
            input("  Press Enter to exit...")
            raise SystemExit(1)

# ── Layout constants ───────────────────────────────────────────────────────────
TOOLBAR_H    = 95       # toolbar height (px)
DEAD_ZONE    = 15       # gap below toolbar where drawing doesn't start
HOVER_FRAMES = 11       # frames to hold finger on button to activate (~0.37s at 30fps)

# ── Color palette (name, BGR) ──────────────────────────────────────────────────
COLORS = [
    ("RED",    (0,   0,   220)),
    ("GREEN",  (0,   190, 0  )),
    ("BLUE",   (220, 60,  0  )),
    ("YELLOW", (0,   210, 235)),
    ("PURPLE", (180, 0,   180)),
    ("WHITE",  (240, 240, 240)),
    ("ORANGE", (0,   130, 255)),
    ("CYAN",   (200, 210, 0  )),
]

BRUSH_OPTIONS = [("S", 3), ("M", 7), ("L", 14), ("XL", 22)]

# ── Hand landmark indices ──────────────────────────────────────────────────────
IDX_TIP, IDX_PIP = 8,  6   # index finger tip / proximal
HAND_CONNECTIONS = HandLandmarksConnections.HAND_CONNECTIONS


# ── Drawing helper ─────────────────────────────────────────────────────────────
def _rounded_rect(img, pt1, pt2, color, radius=7, border_color=None, border_w=2):
    x1, y1 = pt1
    x2, y2 = pt2
    r = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    if r == 0:
        cv2.rectangle(img, pt1, pt2, color, -1)
        return
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
    for cx, cy in [(x1+r, y1+r), (x2-r, y1+r), (x1+r, y2-r), (x2-r, y2-r)]:
        cv2.circle(img, (cx, cy), r, color, -1)
    if border_color:
        cv2.rectangle(img, (x1+r, y1),   (x2-r, y2),   border_color, border_w)
        cv2.rectangle(img, (x1, y1+r),   (x2, y2-r),   border_color, border_w)
        for cx, cy in [(x1+r, y1+r), (x2-r, y1+r), (x1+r, y2-r), (x2-r, y2-r)]:
            cv2.circle(img, (cx, cy), r, border_color, border_w)


def _text_centered(img, text, cx, cy, scale, color, thick=1):
    """Draw text horizontally centered around (cx, cy)."""
    tsz = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0]
    tx  = cx - tsz[0] // 2
    ty  = cy + tsz[1] // 2
    cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _semi_rect(img, pt1, pt2, color, alpha=0.55):
    """Draw a semi-transparent filled rectangle."""
    x1, y1 = pt1
    x2, y2 = pt2
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return
    overlay        = roi.copy()
    overlay[:]     = color
    img[y1:y2, x1:x2] = cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0)


# ══════════════════════════════════════════════════════════════════════════════
class AirCanvas:

    def __init__(self):
        self.draw_color   = COLORS[0][1]         # default Red
        self.brush_size   = BRUSH_OPTIONS[1][1]  # default M
        self.eraser_mode  = False
        self.canvas       = None
        self.prev_pt      = None
        self.hover_count  = {}
        self.feedback_msg = ""
        self.feedback_ttl = 0
        self.buttons      = []
        self.show_help    = True   # help overlay on by default for beginners
        self.save_dir     = os.path.join(os.path.expanduser("~"), "Desktop")
        self._quit        = False
        self._no_hand_frames = 0  # consecutive frames without a hand

    # ── Canvas ────────────────────────────────────────────────────────────────
    def _ensure_canvas(self, h, w):
        if self.canvas is None or self.canvas.shape[:2] != (h, w):
            self.canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # ── Build buttons ─────────────────────────────────────────────────────────
    def _build_buttons(self, fw):
        all_items = []
        for name, color in COLORS:
            all_items.append(("color",  name[:3], color,    color))
        all_items.append(    ("eraser", "ERASE",  None,     (65, 65, 65)))
        all_items.append(    ("clear",  "CLEAR",  None,     (35, 35, 35)))
        all_items.append(    ("save",   "SAVE",   None,     (0, 115, 0)))
        all_items.append(    ("quit",   "QUIT",   None,     (130, 0, 0)))
        for name, val in BRUSH_OPTIONS:
            all_items.append(("brush",  name,     val,      (50, 50, 130)))

        n   = len(all_items)
        GAP = 4
        PAD = 8
        bw  = max(34, (fw - 2 * PAD - GAP * (n - 1)) // n)
        y1, y2 = 6, TOOLBAR_H - 6

        buttons = []
        x = PAD
        for atype, label, aval, bg in all_items:
            x2c = min(x + bw, fw - PAD)
            buttons.append({
                "id":     atype + "_" + label,
                "label":  label,
                "rect":   (x, y1, x2c, y2),
                "action": (atype, aval),
                "bg":     bg,
            })
            x += bw + GAP

        self.buttons = buttons

    # ── Toolbar rendering ──────────────────────────────────────────────────────
    def _draw_toolbar(self, frame):
        w = frame.shape[1]

        # Dark background
        cv2.rectangle(frame, (0, 0), (w, TOOLBAR_H), (22, 22, 22), -1)
        cv2.line(frame, (0, TOOLBAR_H), (w, TOOLBAR_H), (80, 80, 80), 2)

        for btn in self.buttons:
            x1, y1, x2, y2 = btn["rect"]
            atype = btn["action"][0]
            aval  = btn["action"][1]
            label = btn["label"]
            bg    = btn["bg"]

            is_active = (
                (atype == "color"  and aval == self.draw_color and not self.eraser_mode) or
                (atype == "eraser" and self.eraser_mode) or
                (atype == "brush"  and aval == self.brush_size)
            )
            bcol = (255, 230, 50) if is_active else (75, 75, 75)
            bw2  = 3 if is_active else 1

            _rounded_rect(frame, (x1, y1), (x2, y2), bg, border_color=bcol, border_w=bw2)

            mid_y = (y1 + y2) // 2
            bw_px = x2 - x1

            if atype == "color":
                # Color swatch circle on the left, text on the right
                dot_r = min(9, bw_px // 4)
                dot_x = x1 + dot_r + 3
                cv2.circle(frame, (dot_x, mid_y), dot_r, aval, -1)
                cv2.circle(frame, (dot_x, mid_y), dot_r, (160, 160, 160), 1)
                tx = dot_x + dot_r + 4
            else:
                tx = x1 + 4

            fs   = 0.42
            fw3  = 1
            tsz  = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, fw3)[0]
            ty   = y1 + (y2 - y1 + tsz[1]) // 2
            tx   = min(tx, x2 - tsz[0] - 2)
            tx   = max(tx, x1 + 2)
            cv2.putText(frame, label, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), fw3, cv2.LINE_AA)

            # Hover progress arc
            prog = self.hover_count.get(btn["id"], 0)
            if 0 < prog < HOVER_FRAMES:
                cx    = (x1 + x2) // 2
                cy    = (y1 + y2) // 2
                angle = max(5, int(360 * prog / HOVER_FRAMES))
                ov    = frame.copy()
                cv2.ellipse(ov, (cx, cy), (20, 20), -90, 0, angle, (255, 230, 50), 3, cv2.LINE_AA)
                cv2.addWeighted(ov, 0.85, frame, 0.15, 0, frame)

    # ── Gesture logic ─────────────────────────────────────────────────────────
    @staticmethod
    def _index_up(lm):
        """True when index finger tip is above its knuckle (finger is raised)."""
        return lm[IDX_TIP].y < lm[IDX_PIP].y

    # ── Button hover dwell ────────────────────────────────────────────────────
    def _check_buttons(self, fx, fy):
        triggered = None
        for btn in self.buttons:
            x1, y1, x2, y2 = btn["rect"]
            bid = btn["id"]
            if x1 <= fx <= x2 and y1 <= fy <= y2:
                cnt = self.hover_count.get(bid, 0) + 1
                self.hover_count[bid] = cnt
                if cnt >= HOVER_FRAMES:
                    self.hover_count[bid] = 0
                    triggered = btn["action"]
            else:
                old = self.hover_count.get(bid, 0)
                if old > 0:
                    self.hover_count[bid] = old - 1
        return triggered

    def _decay_hovers(self, amount=2):
        for bid in list(self.hover_count):
            self.hover_count[bid] = max(0, self.hover_count[bid] - amount)

    # ── Apply button action ────────────────────────────────────────────────────
    def _apply(self, action, h, w):
        atype, aval = action
        if atype == "color":
            self.draw_color  = aval
            self.eraser_mode = False
            name = next((n for n, c in COLORS if c == aval), "color")
            self._feedback(name + " selected")
        elif atype == "eraser":
            self.eraser_mode = not self.eraser_mode
            self._feedback("Eraser ON  (hover ERASE again to turn OFF)" if self.eraser_mode else "Eraser OFF")
        elif atype == "brush":
            self.brush_size = aval
            name = next((n for n, v in BRUSH_OPTIONS if v == aval), "?")
            self._feedback("Brush size: " + name + "  (" + str(aval) + " px)")
        elif atype == "clear":
            self.canvas = np.zeros((h, w, 3), dtype=np.uint8)
            self._feedback("Canvas cleared!")
        elif atype == "save":
            self._save()
        elif atype == "quit":
            self._quit = True

    # ── Save ──────────────────────────────────────────────────────────────────
    def _save(self):
        try:
            os.makedirs(self.save_dir, exist_ok=True)
            ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = "aircanvas_" + ts + ".png"
            path  = os.path.join(self.save_dir, fname)
            cv2.imwrite(path, self.canvas)
            self._feedback("Saved to Desktop: " + fname)
            print("[AirCanvas] Drawing saved to:", path)
        except Exception as e:
            self._feedback("Save failed: " + str(e))

    # ── Feedback banner ───────────────────────────────────────────────────────
    def _feedback(self, msg, duration=85):
        self.feedback_msg = msg
        self.feedback_ttl = duration

    def _draw_feedback(self, frame):
        if self.feedback_ttl <= 0:
            return
        h, w = frame.shape[:2]
        text = self.feedback_msg
        fs   = 0.62
        fw   = 2
        tsz  = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, fw)[0]
        tx   = (w - tsz[0]) // 2
        ty   = h - 16
        pad  = 6
        _semi_rect(frame,
                   (tx - pad, ty - tsz[1] - pad),
                   (tx + tsz[0] + pad, ty + pad),
                   (0, 0, 0), alpha=0.6)
        cv2.putText(frame, text, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 230, 230), fw, cv2.LINE_AA)
        self.feedback_ttl -= 1

    # ── Status bar (bottom strip) ─────────────────────────────────────────────
    def _draw_status(self, frame, mode_label, hand_found):
        h, w = frame.shape[:2]
        bar_y   = h - 36
        bar_h   = 36
        # Semi-transparent dark bar
        _semi_rect(frame, (0, bar_y), (w, h), (10, 10, 10), alpha=0.70)

        if not hand_found:
            txt = "  NO HAND DETECTED  --  Show your hand to the camera and raise your INDEX FINGER"
            cv2.putText(frame, txt, (10, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 140, 255), 1, cv2.LINE_AA)
            return

        # Mode badge
        badge_colors = {
            "DRAW":    (0, 210, 0),
            "ERASE":   (255, 100, 0),
            "SELECT":  (255, 220, 0),
            "IDLE":    (140, 140, 140),
        }
        badge_col = badge_colors.get(mode_label, (140, 140, 140))
        btext     = "  " + mode_label + "  "
        btsz      = cv2.getTextSize(btext, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)[0]
        bx1, by1  = 8, bar_y + 4
        bx2, by2  = bx1 + btsz[0] + 4, bar_y + bar_h - 4
        _rounded_rect(frame, (bx1, by1), (bx2, by2), badge_col, radius=4)
        cv2.putText(frame, btext, (bx1 + 2, by2 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 2, cv2.LINE_AA)

        hint_x = bx2 + 14
        if mode_label == "DRAW":
            hint = "Move finger to draw  |  Lower finger = lift pen  |  Point UP into toolbar = select tools"
        elif mode_label == "ERASE":
            hint = "Move finger to erase  |  Hover ERASE button again to switch back to draw"
        elif mode_label == "SELECT":
            hint = "Hold finger over a button for ~0.4s to activate it"
        else:
            hint = "Raise your INDEX FINGER to draw  |  Move into toolbar to select tools"

        cv2.putText(frame, hint, (hint_x, h - 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)

    # ── Help overlay ──────────────────────────────────────────────────────────
    def _draw_help(self, frame):
        if not self.show_help:
            return
        h, w  = frame.shape[:2]
        pw, ph = 340, 230
        px, py = 10, TOOLBAR_H + 10

        # Panel background
        _semi_rect(frame, (px, py), (px + pw, py + ph), (10, 10, 30), alpha=0.82)
        cv2.rectangle(frame, (px, py), (px + pw, py + ph), (80, 200, 255), 1)

        lines = [
            (" HOW TO USE  [H to hide]", (80, 220, 255), 0.50, 2),
            ("", (0,0,0), 0.0, 0),
            (" 1. RAISE your INDEX FINGER", (220, 220, 220), 0.42, 1),
            ("    pointing at the camera.", (180, 180, 180), 0.38, 1),
            ("", (0,0,0), 0.0, 0),
            (" 2. Move finger INTO TOOLBAR", (220, 220, 220), 0.42, 1),
            ("    (top bar) to select tools.", (180, 180, 180), 0.38, 1),
            ("    Hold ~0.4s to activate.", (180, 180, 180), 0.38, 1),
            ("", (0,0,0), 0.0, 0),
            (" 3. Move finger BELOW toolbar", (220, 220, 220), 0.42, 1),
            ("    to DRAW on the canvas.", (180, 180, 180), 0.38, 1),
            ("", (0,0,0), 0.0, 0),
            (" 4. LOWER finger = lift pen.", (220, 220, 220), 0.42, 1),
            ("", (0,0,0), 0.0, 0),
            (" Keys: S=Save  C=Clear  H=Help  ESC=Quit",
             (140, 200, 140), 0.36, 1),
        ]
        cy = py + 22
        for text, col, scale, thick in lines:
            if scale == 0.0:
                cy += 6
                continue
            cv2.putText(frame, text, (px + 6, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick, cv2.LINE_AA)
            tsz = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0]
            cy += tsz[1] + 7

    # ── Cursor ────────────────────────────────────────────────────────────────
    def _draw_cursor(self, frame, fx, fy, in_toolbar):
        if in_toolbar:
            cv2.circle(frame, (fx, fy), 13, (255, 230, 50), 2, cv2.LINE_AA)
            cv2.circle(frame, (fx, fy),  3, (255, 230, 50), -1)
        elif self.eraser_mode:
            r = 24
            cv2.circle(frame, (fx, fy), r, (220, 220, 220), 2)
            cv2.line(frame, (fx - r, fy), (fx + r, fy), (220, 220, 220), 1)
            cv2.line(frame, (fx, fy - r), (fx, fy + r), (220, 220, 220), 1)
        else:
            cv2.circle(frame, (fx, fy), self.brush_size, self.draw_color, -1)
            cv2.circle(frame, (fx, fy), self.brush_size + 2, (210, 210, 210), 1)

    # ── Hand skeleton ─────────────────────────────────────────────────────────
    def _draw_skeleton(self, frame, lm_px):
        for c in HAND_CONNECTIONS:
            cv2.line(frame, lm_px[c.start], lm_px[c.end], (240, 170, 0), 1, cv2.LINE_AA)
        for pt in lm_px:
            cv2.circle(frame, pt, 2, (0, 255, 140), -1)

    # ── Main loop ─────────────────────────────────────────────────────────────
    def run(self):
        _ensure_model()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print()
            print("  [ERROR] Cannot open webcam!")
            print("  Make sure your camera is connected and not used by another app.")
            input("  Press Enter to exit...")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.50,
            min_hand_presence_confidence=0.50,
            min_tracking_confidence=0.50,
        )

        buttons_built    = False
        consecutive_fail = 0
        start_time       = time.time()
        window_ready     = False   # True after first imshow

        cv2.namedWindow("AirCanvas", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("AirCanvas", 1280, 720)

        with HandLandmarker.create_from_options(options) as landmarker:
            while not self._quit:

                # Only check for window-closed AFTER first frame is shown
                if window_ready:
                    try:
                        if cv2.getWindowProperty("AirCanvas", cv2.WND_PROP_VISIBLE) < 1:
                            break
                    except cv2.error:
                        break

                ret, frame = cap.read()
                if not ret or frame is None:
                    consecutive_fail += 1
                    if consecutive_fail > 30:
                        print("[ERROR] Camera stopped sending frames. Exiting.")
                        break
                    continue
                consecutive_fail = 0

                frame = cv2.flip(frame, 1)
                h, w  = frame.shape[:2]

                self._ensure_canvas(h, w)
                if not buttons_built:
                    self._build_buttons(w)
                    buttons_built = True

                # ── Hand detection ─────────────────────────────────────────
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                ts_ms  = int((time.time() - start_time) * 1000)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(mp_img, ts_ms)
                rgb.flags.writeable = True

                fx = fy = None
                mode_label = "IDLE"
                hand_found = bool(result.hand_landmarks)

                if hand_found:
                    self._no_hand_frames = 0
                    lm     = result.hand_landmarks[0]
                    fx     = int(lm[IDX_TIP].x * w)
                    fy     = int(lm[IDX_TIP].y * h)
                    idx_up = self._index_up(lm)

                    if idx_up:
                        in_toolbar = fy < TOOLBAR_H
                        in_draw    = fy > (TOOLBAR_H + DEAD_ZONE)

                        if in_toolbar:
                            mode_label = "SELECT"
                            action = self._check_buttons(fx, fy)
                            if action:
                                self._apply(action, h, w)
                            self.prev_pt = None

                        elif in_draw:
                            mode_label = "DRAW" if not self.eraser_mode else "ERASE"
                            color = (0, 0, 0)      if self.eraser_mode else self.draw_color
                            thick = 44             if self.eraser_mode else self.brush_size
                            if self.prev_pt is not None:
                                cv2.line(self.canvas, self.prev_pt, (fx, fy),
                                         color, thick, cv2.LINE_AA)
                            self.prev_pt = (fx, fy)

                        else:
                            # Dead zone — pen up, no selection
                            mode_label = "IDLE"
                            self.prev_pt = None
                    else:
                        mode_label = "IDLE"
                        self.prev_pt = None
                        self._decay_hovers(2)

                    lm_px = [(int(p.x * w), int(p.y * h)) for p in lm]
                    self._draw_skeleton(frame, lm_px)

                else:
                    self._no_hand_frames += 1
                    self.prev_pt = None
                    self._decay_hovers(2)

                # ── Composite drawing → webcam frame ───────────────────────
                gray     = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
                _, mask  = cv2.threshold(gray, 12, 255, cv2.THRESH_BINARY_INV)
                mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                frame    = cv2.bitwise_and(frame, mask_bgr)
                frame    = cv2.bitwise_or(frame, self.canvas)

                # ── Overlays (order matters) ───────────────────────────────
                self._draw_toolbar(frame)
                if fx is not None and fy is not None:
                    self._draw_cursor(frame, fx, fy, fy < TOOLBAR_H)
                self._draw_help(frame)
                self._draw_status(frame, mode_label, hand_found)
                self._draw_feedback(frame)

                cv2.imshow("AirCanvas", frame)
                window_ready = True

                # ── Keyboard shortcuts ─────────────────────────────────────
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q'), ord('Q')):
                    break
                elif key in (ord('s'), ord('S')):
                    self._save()
                elif key in (ord('c'), ord('C')):
                    self.canvas = np.zeros((h, w, 3), dtype=np.uint8)
                    self._feedback("Canvas cleared!")
                elif key in (ord('h'), ord('H')):
                    self.show_help = not self.show_help
                    self._feedback("Help: ON" if self.show_help else "Help: OFF  (press H to show again)")

        cap.release()
        cv2.destroyAllWindows()
        print("[AirCanvas] Closed. Goodbye!")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print()
    print("=" * 62)
    print("   AirCanvas  -  Draw in the Air with Your Finger!")
    print("=" * 62)
    print()
    print("  QUICK START:")
    print("  1) A window will open showing your webcam feed.")
    print("  2) Show your hand and raise your INDEX FINGER.")
    print("  3) Move your finger into the TOP TOOLBAR to pick a color.")
    print("     -> Hold your finger over a button for ~0.4s to select it.")
    print("  4) Move your finger BELOW the toolbar to start DRAWING.")
    print("  5) Lower your finger (or make a fist) to stop drawing.")
    print()
    print("  Keyboard (focus the AirCanvas window first):")
    print("    H  = toggle help panel    S = save    C = clear    ESC = quit")
    print()
    print("  A help panel will appear inside the window when it opens.")
    print("  Press H inside the window to hide/show it.")
    print()
    print("  Starting... (this may take a few seconds)")
    print()
    AirCanvas().run()
