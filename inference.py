
import torch
from torchvision import transforms
from model.network import ResNet
import cv2
import matplotlib.pyplot as plt
from utils import draw_box

# def calculate_accuracy(fx, y):
#     preds = fx.argmax(1, keepdim=True)
#     correct = preds.eq(y.argmax(1, keepdim=True).view_as(preds)).sum()
#     acc = correct.float() / preds.shape[0]
#     return acc


def load_model(weight_path, device):

    print("device:", device)
    # load model
    print("Load model")
    net = ResNet()
    net.load_state_dict(torch.load(weight_path))
    return net


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





if __name__ == '__main__':
    img_path = r"D:\AI programme\Machine Learning final project\Dataset\Test Image\WIDER_test\images\1--Handshaking\1_Handshaking_Handshaking_1_725.jpg"
    weight_path = "model/weight/ResNet50_best.pt"

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # load model
    net = load_model(weight_path, device)
    net.to(device)

    # load input
    ip, original_img, original_shape = load_input(img_path)
    ip = ip.to(device)

    # run to get output
    op = net(ip)  # ip must be: 4 dims

    # post-processing
    h, w, _ = original_shape
    op[:, ::2] *= w
    op[:, 1::2] *= h
    po_list = op.cpu().tolist()  # -> [[x1, y1, x2, y2]]
    print(po_list)
    predict_image = draw_box(original_img, po_list)
    plt.imshow(predict_image[..., ::-1])
    plt.show()




