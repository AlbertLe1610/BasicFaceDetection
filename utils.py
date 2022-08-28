from torchvision import transforms
import cv2


def draw_box(img, bounding_boxes=[], color=(0, 0, 255)):
    img_copy = img.copy()
    for [x1, y1, x2, y2] in bounding_boxes:
        cv2.rectangle(img_copy, (int(x1), int(y1)), (int(x2), int(y2)), color, 10)
    return img_copy

def ResizePadding(img, image_size=224):
    ratio = image_size / max(img.shape)
    img_resize = cv2.resize(img, None, fx=ratio, fy=ratio)
    k = abs(img_resize.shape[0] - img_resize.shape[1])  # padd bao nhieu hang cot
    if img_resize.shape[0] < img_resize.shape[1]:  # kiem tra xem padd hang hay pad cot
        imgPadding = cv2.copyMakeBorder(img_resize, 0, k, 0, 0, cv2.BORDER_CONSTANT, None, value=0)
    else:
        imgPadding = cv2.copyMakeBorder(img_resize, 0, 0, 0, k, cv2.BORDER_CONSTANT, None, value=0)
    return imgPadding

def load_input(path):
    """

    :param path: image path
    :return: image tensor, original image shape
    """
    img = cv2.imread(path)

    resize_padd_img = ResizePadding(img)
    img_tensor = transforms.ToTensor()(resize_padd_img)
    img_tensor.unsqueeze_(0)
    return img_tensor, img, img.shape