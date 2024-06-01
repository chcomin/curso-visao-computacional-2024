import numpy as np
import cv2

def find_object(img_scene, img_obj, kp_obj, des_object, sift):
    """Identifica a posição de um objeto em uma outra imagem.

    Args:
        img_scene: Imagem global na qual o objeto será buscado
        img_obj: Imagem do objeto a ser buscado
        kp_obj: Pontos salientes do objeto previamente calculados
        des_object: Descritores dos pontos salientes do objeto
        sift: Instância do detector SIFT

    Returns:
        Coordenada do bounding box de `img_obj` na imagem `img_scene`
    """

    # Detecta os pontos salientes e realiza o casamento
    kp_scene, des_scene = sift.detectAndCompute(img_scene, None)
    bf = cv2.BFMatcher(crossCheck=True)
    matches = bf.knnMatch(des_object, des_scene, k=1)

    # Mantém apenas os pontos que possuem uma correspondência na outra imagem
    good = []
    for m in matches:
        if len(m)!=0:
            good.append(m[0])

    if len(good)<=10:
        obj_bounds_in_scene = cv2.drawMatches(
            img_obj, kp_obj, img_scene, kp_scene, good, None)
    else:
        obj_pts = np.zeros((len(good), 2), dtype=np.float32)
        scene_pts = np.zeros((len(good), 2), dtype=np.float32)
        for i, m in enumerate(good):
            obj_pts[i] = kp_obj[m.queryIdx].pt
            scene_pts[i] = kp_scene[m.trainIdx].pt

        # Mapeia os pontos do objeto na imagem global
        t_matrix, _ = cv2.findHomography(
            obj_pts, scene_pts, cv2.RANSAC, ransacReprojThreshold=3.0)
        h, w = img_obj.shape
        obj_bounds = ((0, 0), (0, h-1), (w-1, h-1), (w-1, 0))
        obj_bounds = np.array(obj_bounds, dtype=np.float32).reshape(-1,1,2)
        obj_bounds_in_scene = cv2.perspectiveTransform(obj_bounds, t_matrix)

    return obj_bounds_in_scene

def draw_bbox(img_scene, obj_bounds):
    """Desenha a bounding box do objeto em uma imagem"""

    img_scene_obj = cv2.polylines(
        img_scene.copy(), [np.int32(obj_bounds)], True, 255, 3, cv2.LINE_AA)

    return img_scene_obj
