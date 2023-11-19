import cv2
import numpy as np

def detect_shape(image: np.array) -> str:
    """画像に含まれる標識の形状を検出する

    Args:
        image (np.array): 2値画像

    Returns:
        str: 標識の形状
    """
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours[0]) > 5:
        # 楕円の外に標識がはみ出ていない場合は円と判定
        ellipse = cv2.fitEllipse(contours[0])
        mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.ellipse(mask, ellipse, 255, -1)
        rate = np.sum((image > 0) & (mask == 0))/np.sum(mask == 0)
        if rate < 0.05:
            return "circle", ellipse

    # other shapes
    tmp= cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = tmp[0] if len(tmp) == 2 else tmp[1]
    cnt = contours[0]
    epsilon = 0.1*cv2.arcLength(cnt,True)
    area = cv2.approxPolyDP(cnt,epsilon,True)
    area = sorted(area, key=lambda x: x[0][1])
    shape = ""
    if len(area) == 3:
        shape = "triangle"
        p = area
        if abs(p[0][0, 1]-p[1][0, 1]) < abs(p[1][0, 1]-p[2][0, 1]):
            shape = "inverted_" + shape
        return shape, area

    if len(area) == 4:
        return "rectangle", area

    return "others", None

def clone(src: np.array, tgt: np.array, mask: np.array, center: tuple[int, int]) -> np.array:
    """src を tgt の center に合成する

    Args:
        src (np.array): 合成する画像
        tgt (np.array): 合成される画像
        mask (np.array): src のマスク
        center (tuple[int, int]): 合成する位置

    Returns:
        np.array: 合成後の画像
    """
    # seamless ではなく、単に差し込む
    h, w = src.shape[:2]
    x, y = center
    x0, y0 = x - w//2, y - h//2
    img = tgt.copy()
    img[y0:y0+h, x0:x0+w] = src * (mask > 0)[..., np.newaxis] + tgt[y0:y0+h, x0:x0+w] * (mask == 0)[..., np.newaxis]
    return img

def blend_sign(bg_img: np.array, sign_img: np.array, annotation: list[float]) -> np.array:
    """標識をポアソンブレンディングで合成する

    Args:
        bg_img (np.array): 背景画像
        sign_img (np.array): 標識画像
        annotation (list[float]): 標識のアノテーション. YOLO 形式

    Returns:
        np.array: 合成後の画像
    """
    height, width = bg_img.shape[:2]
    hs, ws = sign_img.shape[:2]
    c, x, y, w, h = annotation
    # 矩形領域に収まるサイズ標識をリサイズする、その際にアスペクト比を維持する
    scale = min(w*width / ws, h*height / hs)
    sign_img = sign_img.copy()
    sign_img = cv2.resize(sign_img, (0, 0), fx=scale, fy=scale)
    hs, ws = sign_img.shape[:2]
    # 矩形の中心に sign_img を配置する
    mask = np.zeros((hs, ws), dtype=np.uint8)
    mask[sign_img[:, :, 3] > 0] = 255
    center = (int(x*width), int(y*height))
    sign_img = sign_img[:, :, :3]
    # return clone(sign_img, bg_img, mask, center)
    return cv2.seamlessClone(sign_img, bg_img, mask, p=center, flags=cv2.NORMAL_CLONE)


def get_threepoints(area: np.array) -> np.array:
    # 左上、右上、左下の順に並べる
    p1, p2 = area[:2]
    p1, p2 = sorted([p1, p2], key=lambda x: x[0, 0])
    p3 = sorted(area[2:], key=lambda x: x[0, 0])[0]
    return np.array([p1, p2, p3])

def blend_polygon(bg_img: np.array, sign_img: np.array, area: np.array, area_sign: np.array, annotation: list[float]) -> np.array:
    area = get_threepoints(area)
    area_sign = get_threepoints(area_sign)
    # bbox の中心を求める
    height, width = bg_img.shape[:2]
    _, cx, cy, w, h = annotation
    sx = int((cx - w/2) * width)
    sy = int((cy - h/2) * height)
    # bg の適切な形に sign を変形する
    h, w = sign_img.shape[:2]
    affine = cv2.getAffineTransform(area_sign.astype(np.float32), area.astype(np.float32))
    warp_img = cv2.warpAffine(sign_img, affine, (w, h), cv2.INTER_LANCZOS4)
    # warp_img の不透明部分の幅と高さ、左上の座標を求める
    h, w = np.where(warp_img[:, :, 3] > 0)
    x0, y0 = min(w), min(h)
    h = max(h) - min(h)
    w = max(w) - min(w)
    warp_img = warp_img[y0:y0+h, x0:x0+w]

    # bg 上のサイズに変換する
    x = [p[0][0] for p in area]
    y = [p[0][1] for p in area]
    # center = (sum(x)//len(x), sum(y)//len(y))
    center = ((max(x)+min(x))//2 + sx, (max(y)+min(y))//2 + sy)
    w = max(x) - min(x)
    h = max(y) - min(y)
    warp_img = cv2.resize(warp_img, (w, h), interpolation=cv2.INTER_LANCZOS4)

    # マスクを作成
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[warp_img[:, :, 3] > 0] = 255

    warp_img = warp_img[:, :, :3]
    img = cv2.seamlessClone(warp_img, bg_img, mask, center, cv2.NORMAL_CLONE)
    return img
    