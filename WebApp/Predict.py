import cv2
import numpy as np
import torch


def predict(face, model):
    face = cv2.resize(face, (200, 200))
    face = np.transpose(face, (2, 0, 1))
    face = face / 255.
    pred = model(torch.Tensor(np.array([face]))).detach().numpy()
    return pred[0]

    # del last_ages[0]
    # last_ages.append(CLASS_NAMES[pred_class])
    # # print(pred)
    # pred = pred[0]
    # pred -= min(pred)
    # pred /= sum(pred)
    # # print(pred)
    #
    # dominant_class = max(set(last_ages), key=last_ages.count)
    # print(last_ages)
    # print(sorted(last_ages))
    #
    # img = cv2.putText(frame, CLASS_NAMES[pred_class] + 'Prob: {:.2f}'.format(max(pred)), (x, y), font,
    #                   fontScale, color, thickness, cv2.LINE_AA)