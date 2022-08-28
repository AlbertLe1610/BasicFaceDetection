
import os
import torch
from torch.optim import Adam
from dataLoader import CustomImageDataset
from model.network import ResNet
from torch.utils.data import DataLoader
from model.loss import RetinaNetBoxLoss
import cv2
import utils
import matplotlib.pyplot as plt

def calculate_accuracy(fx, y):
    preds = fx.argmax(1, keepdim=True)
    correct = preds.eq(y.argmax(1, keepdim=True).view_as(preds)).sum()
    acc = correct.float() / preds.shape[0]
    return acc

def train(model, device, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for (y, x) in iterator:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        fx = model(x)

        loss = criterion(fx, y)
        # acc = calculate_accuracy(fx, y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        # epoch_acc += acc.item()

    # return epoch_loss / len(iterator), epoch_acc / len(iterator)
    return epoch_loss / len(iterator)


def evaluate(model, device, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()
    with torch.no_grad():
        for (y, x) in iterator:
            x = x.to(device)
            y = y.to(device)

            fx = model(x)

            loss = criterion(fx, y)
            # acc = calculate_accuracy(fx, y)

            epoch_loss += loss.item()
            # epoch_acc += acc.item()

    # return epoch_loss / len(iterator), epoch_acc / len(iterator)
    return epoch_loss / len(iterator)

def run():
    epoch = 50
    save_dir = "model/weight"
    use_gpu = True
    learning_rate = 10e-3
    pretrain = False
    batch_size = 64
    shuffle = True
    num_workers = 0
    # device = "cuda" if torch.cuda.is_available() else "cpu" --> check for gpu
    device = torch.device("cuda") if torch.cuda.is_available() and use_gpu else torch.device("cpu")
    print("device:", device)

    # model, loss_fn, resample data, use weight, version
    prex = '%s' % ("ResNet50")
    LATEST_MODEL_SAVE_PATH = os.path.join(save_dir, '%s_latest.pt' % prex)
    BEST_MODEL_SAVE_PATH = os.path.join(save_dir, '%s_best.pt' % prex)

    if not os.path.isdir(f'{save_dir}'):
        os.makedirs(f'{save_dir}')

    # Dataloader
    data_train = CustomImageDataset("D:\AI programme\Machine Learning final project\Dataset\Train Image\WIDER_train\images",
                           "D:\AI programme\Machine Learning final project\Dataset\Annotation\wider_face_split\wider_face_train_bbx_gt.txt")
    dataloader_train = DataLoader(data_train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    print('TOTAL training images:', len(data_train))
    print('-----------------------------------------')

    data_val = CustomImageDataset("D:\AI programme\Machine Learning final project\Dataset\Validation Image\WIDER_val\images",
                         "D:\AI programme\Machine Learning final project\Dataset\Annotation\wider_face_split\wider_face_val_bbx_gt.txt")
    dataloader_val = DataLoader(data_val, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    print('TOTAL validating images:', len(data_val))
    print('-----------------------------------------')

    # load model
    if os.path.exists(LATEST_MODEL_SAVE_PATH) and pretrain:
        print("Load model")
        net = ResNet()
        net.load_state_dict(torch.load(LATEST_MODEL_SAVE_PATH))
        net.to(device)
    else:
        net = ResNet().to(device)

    if not os.path.exists('train_output'):
        os.mkdir('train_output')
    ip, original_img, original_shape = utils.load_input('1_Handshaking_Handshaking_1_848.jpg')
    ip = ip.to(device)
    loss_fn = RetinaNetBoxLoss()
    optimizer = Adam(net.parameters(), learning_rate)
    best_valid_loss = float('inf')
    for epoch in range(epoch):
        # train_loss, train_acc = train(net, device, dataloader_train, optimizer, loss_fn)
        # valid_loss, valid_acc = evaluate(net, device, dataloader_val, loss_fn)
        train_loss = train(net, device, dataloader_train, optimizer, loss_fn)
        valid_loss = evaluate(net, device, dataloader_val, loss_fn)
        torch.save(net.state_dict(), LATEST_MODEL_SAVE_PATH)



        # run to get output
        op = net(ip)  # ip must be: 4 dims

        # post-processing
        h, w, _ = original_shape
        op[:, ::2] *= w
        op[:, 1::2] *= h
        po_list = op.cpu().tolist()  # -> [[x1, y1, x2, y2]]
        predict_image = utils.draw_box(original_img, po_list)
        cv2.imwrite('train_output/epoch_%i.jpg' %(epoch), predict_image)




        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(net.state_dict(), BEST_MODEL_SAVE_PATH)

        # print(f'| Epoch: {epoch + 1:02} | '
        #       f'Train Loss: {train_loss:.3f} | '
        #       f'Train Acc: {train_acc * 100:05.2f} | '
        #       f'Val. Loss: {valid_loss:.3f} | '
        #       f'Val. Acc: {valid_acc * 100:05.2f}% |')
        print(f'| Epoch: {epoch + 1:02} | '
              f'Train Loss: {train_loss:.3f} | '
              f'Val. Loss: {valid_loss:.3f} | ')


    # net.load_state_dict(torch.load(BEST_MODEL_SAVE_PATH))
    #
    # test_loss, test_acc = evaluate(net, device, dataloader_test, loss_fn_test)
    # torch.save(net.state_dict(), os.path.join(config.save_dir, '%s_best_%.4f.pt' % (prex, test_loss)))
    # print(f'| Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:05.2f}% |')


if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model", default="resnet18")
    # parser.add_argument("--train_dir", required=True)
    # parser.add_argument("--test_dir", required=True)
    # parser.add_argument("--label_path", required=True)
    # parser.add_argument("--pretrain", default=True)
    # parser.add_argument("--version", default="floss")
    # args = parser.parse_args()


    run()




