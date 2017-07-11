import cv2
import numpy as np


def main():
    frame1 = cv2.imread('data/raw/random_logs/val/image_file_log_fc5d9e_0000030087.png')
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    frame2 = cv2.imread('data/raw/random_logs/val/image_file_log_fc5d9e_0000030103.png')
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite('opencv_vis/%s' % 'check.png', bgr)
    # k = cv2.waitKey(30) & 0xff
    # if k == 27:
    #     break
    # elif k == ord('s'):
    #     cv2.imwrite('opticalfb.png', frame2)
    #     cv2.imwrite('opticalhsv.png', bgr)
    # prvs = next
    # cap.release(),
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


