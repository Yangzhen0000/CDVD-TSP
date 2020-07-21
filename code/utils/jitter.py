import cv2
import numpy as np
import math


def psnr(img1, img2, range=65535):
    mse = np.mean((img1/range - img2/range) ** 2 )
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def jitter(img, step, factor=0.1, range=65535):
    up = np.zeros_like(img)
    down = np.zeros_like(img)
    left = np.zeros_like(img)
    right = np.zeros_like(img)

    if img.ndim == 3:
        up[1:, :, :] = img[1:, :, :] - img[:-1, :, :]
        down[:-1, :, :] = img[:-1, :, :] - img[1:, :, :]
        left[:, 1:, :] = img[:, 1:, :] - img[:, :-1, :]
        right[:, :-1, :] = img[:, :-1, :] - img[:, 1:, :]
    else:
        up[1:, :] = img[1:, :] - img[:-1, :]
        down[:-1, :] = img[:-1, :] - img[1:, :]
        left[:, 1:] = img[:, 1:] - img[:, :-1]
        right[:, :-1] = img[:, :-1] - img[:, 1:]

    pos_up_weight = (up == step).astype(int)
    neg_up_weight = (up == -step).astype(int)
    pos_down_weight = (down == step).astype(int)
    neg_down_weight = (down == -step).astype(int)
    pos_left_weight = (left == step).astype(int)
    neg_left_weight = (left == -step).astype(int)
    pos_right_weight = (left == step).astype(int)
    neg_right_weight = (left == -step).astype(int)

    pos_weight = (pos_up_weight + pos_down_weight + pos_left_weight + pos_right_weight) / 4
    neg_weight = (neg_up_weight + neg_down_weight + neg_left_weight + neg_right_weight) / 4

    disturb = np.random.normal(0, step, img.shape)*factor
    img = img + pos_weight*disturb - neg_weight*disturb
    return np.clip(img, 0, range)


def generate_mask(img, step):
    up = np.zeros_like(img)
    left = np.zeros_like(img)

    if img.ndim == 3:
        up[1:, :, :] = img[1:, :, :] - img[:-1, :, :]
        left[:, 1:, :] = img[:, 1:, :] - img[:, :-1, :]
    else:
        up[1:, :] = img[1:, :] - img[:-1, :]
        left[:, 1:] = img[:, 1:] - img[:, :-1]
    mask = (up == step) | (up == -step) | (left == step) | (left == -step)
    return mask.astype(int)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    hbd_path = "D:\\SDR4k\\10bit_10099858_p057\\001.png"
    lbd_path = "D:\\SDR4k\\4bit_10099858_p057\\001.png"
    #
    hbd_img = cv2.imread(hbd_path, cv2.IMREAD_UNCHANGED)  # uint16
    lbd_img = cv2.imread(lbd_path, cv2.IMREAD_UNCHANGED)  # uint16
    # hbd_img = hbd_img.astype(np.int32)
    # lbd_img = lbd_img.astype(np.int32)
    #
    step = 16
    # hbd_img = cv2.imread(hbd_path)
    # lbd_img = hbd_img // step * step
    fig, axes = plt.subplots(2, 2)
    axes[0, 0].imshow(hbd_img)
    axes[0, 1].imshow(lbd_img)
    # lbd_img = jitter(lbd_img, step, factor=0.1)
    # cv2.imwrite("../tmp.png", lbd_img.astype(np.uint16))
    # psnr2 = psnr(hbd_img, lbd_img)
    # print("Before jittering, PSNR={:4f}, After jittering, PSNR={:4f}".format(psnr1, psnr2))
    # a = np.array([[1, 4, 3], [2, 5, 6], [9, 8, 1]])
    mask = generate_mask(np.mean(lbd_img, axis=2), step)
    axes[1, 0].imshow(mask)
    mask = np.expand_dims(mask, 2)
    mask = np.concatenate((mask, mask, mask), axis=-1)
    smooth = cv2.blur(lbd_img, (20, 20))
    print(smooth)
    # smooth = cv2.GaussianBlur(lbd_img, (5, 5), step)
    # blend = smooth*mask + lbd_img*(1-mask)  # 只改变了单个像素值，有点少，应该把这个windows里面的都改变
    blend = smooth
    axes[1, 1].imshow(blend)
    plt.show()
    psnr1 = psnr(hbd_img, lbd_img, range=255)
    psnr2 = psnr(hbd_img, blend, range=255)
    print(psnr1, psnr2)

