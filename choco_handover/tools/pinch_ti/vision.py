"""頭部 HD 画像から金色の包みのチロルチョコを色で探す。中心 (u,v)・外接サイズ・面積を返す (見つからなければ None)。"""
import cv2, numpy as np
ROI = (440, 190, 860, 480)        # 右手の届く机の手前側だけを見る (チラシの黄色を除外)
def find_choco(img, roi=ROI):
    """金色の包みを種にして探す。包みが白っぽく写る部分は、金色の塊のすぐ隣の「周りより明るい画素」として取り込む。
    手の金属の反射は金色ではないので拾わない。"""
    x0, y0, x1, y1 = roi; sub = img[y0:y1, x0:x1]; hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    gold = cv2.inRange(hsv, (10, 95, 115), (35, 255, 255)); seed = cv2.morphologyEx(gold, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    th = cv2.morphologyEx(hsv[:, :, 2], cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))); bright = (th > 40).astype(np.uint8) * 255
    n, lab, st, cen = cv2.connectedComponentsWithStats(seed); best = None
    for i in range(1, n):
        a, w, h = st[i, cv2.CC_STAT_AREA], st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
        if a < 60 or w > 60 or h > 60 or w < 8 or h < 8: continue
        if st[i, 0] <= 1 or st[i, 1] <= 1 or st[i, 0] + w >= (x1 - x0) - 1 or st[i, 1] + h >= (y1 - y0) - 1: continue      # 枠にかかる塊は除く
        blob = (lab == i).astype(np.uint8) * 255; grown = cv2.bitwise_and(cv2.dilate(blob, np.ones((11, 11), np.uint8)), cv2.bitwise_or(gold, bright)); grown = cv2.bitwise_or(grown, blob)
        ys, xs = np.nonzero(grown); area = int(len(xs)); bw, bh = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
        if area > 2800 or bw > 66 or bh > 66: ys, xs = np.nonzero(blob); area = int(len(xs)); bw, bh = int(w), int(h)
        if best is None or a > best["gold"]:
            best = {"u": round(float(xs.mean() + x0), 1), "v": round(float(ys.mean() + y0), 1), "w": bw, "h": bh, "area": area, "gold": int(a)}
    return best
def find_in_jpg(jpg_bytes):
    return find_choco(cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_COLOR))
if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]: print(p.split("/")[-1], find_choco(cv2.imread(p)))

# --- 手 (手首の黒い金具) に対するチョコの相対位置。教示した接地時の関係を基準 (0,0) とする ---
import os as _os
_TPL = cv2.imread(_os.path.join(_os.path.dirname(__file__), "wrist_template.png"), cv2.IMREAD_GRAYSCALE)
_TPL_ORIGIN = (700, 325); _CHOCO_REF = (635.1, 360.3)
MM_PER_PX_X = -1.03   # 画像の下 (+v) へ 1px = 胴体の前後 x が -1.03mm
MM_PER_PX_Y = -0.86   # 画像の右 (+u) へ 1px = 胴体の左右 y が -0.86mm
def choco_rel_hand(img):
    """(du, dv, score, choco) を返す。du,dv は「教示時の手とチョコの関係」からのずれ [px]。"""
    c = find_choco(img)
    if c is None or _TPL is None: return None
    res = cv2.matchTemplate(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), _TPL, cv2.TM_CCOEFF_NORMED); _, mx, _, loc = cv2.minMaxLoc(res)
    sh = (loc[0] - _TPL_ORIGIN[0], loc[1] - _TPL_ORIGIN[1])
    return {"du": round(c["u"] - _CHOCO_REF[0] - sh[0], 1), "dv": round(c["v"] - _CHOCO_REF[1] - sh[1], 1), "score": round(float(mx), 3), "choco": c, "hand_shift": sh}
def rel_in_jpg(jpg_bytes):
    return choco_rel_hand(cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_COLOR))

# --- 上空の基準姿勢 (LIFT_TOP) から見たチョコの位置 → 教示した接地位置からのずれ [m] ---
_TPL_H = cv2.imread(_os.path.join(_os.path.dirname(__file__), "wrist_template_hover.png"), cv2.IMREAD_GRAYSCALE)
_TPL_H_ORIGIN = (740, 300); _CHOCO_REF_H = (635.6, 363.4)
def locate_from_hover(img):
    """基準姿勢 LIFT_TOP で撮った HD 画像から (dx, dy) [m] を返す。手首の金具の写り方で頭の向きのずれを打ち消す。"""
    c = find_choco(img)
    if c is None or _TPL_H is None: return None
    res = cv2.matchTemplate(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), _TPL_H, cv2.TM_CCOEFF_NORMED); _, mx, _, loc = cv2.minMaxLoc(res)
    sh = (loc[0] - _TPL_H_ORIGIN[0], loc[1] - _TPL_H_ORIGIN[1]); du = c["u"] - _CHOCO_REF_H[0] - sh[0]; dv = c["v"] - _CHOCO_REF_H[1] - sh[1]
    return {"dx": round(MM_PER_PX_X * dv / 1000.0, 4), "dy": round(MM_PER_PX_Y * du / 1000.0, 4), "du": round(du, 1), "dv": round(dv, 1), "score": round(float(mx), 3), "choco": c, "hand_shift": sh}
