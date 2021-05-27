import torch
import numpy as np
from network import C3D_model,R2Plus1D_model
import cv2

torch.backends.cudnn.benchmark = True


def change_Size(frame):
    frame = frame[8:120, 30:142, :]
    return np.array(frame).astype(np.uint8)

def fetch_class_names():
    with open('./dataloaders/labels.txt', 'r') as f:
        class_names = f.readlines()
        f.close()
    return class_names

def set_inputs(fps,device):
    inputs = np.expand_dims(np.array(fps).astype(np.float32), axis=0)
    inputs = torch.autograd.Variable(torch.from_numpy(np.transpose(inputs, (0, 4, 1, 2, 3))),
                                     requires_grad=False).to(device)
    return inputs

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # flag="image"

    class_names = fetch_class_names()
    modelName = "C3D"
    # init model
    if modelName=="C3D":
        model = C3D_model.C3D(num_classes=4)
    else:
        model = R2Plus1D_model.R2Plus1DClassifier(num_classes=4, layer_sizes=(2, 2, 2, 2))
    checkpoint = torch.load('run\\run_0\\models\\'+modelName+'-ucf101_epoch-19.pth.tar', map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    videoName = 'fixing_repairs_9.mp4'
    video = 'dataset\\Failure\\Fixing_repairs\\'+videoName
    cap = cv2.VideoCapture(video)
    ret = True
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter('out_'+modelName+'-'+videoName, fourcc, 20.0, (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = []
    while ret:
        ret, frame = cap.read()
        if not ret and frame is None:
            continue
        changed_frames = change_Size(cv2.resize(frame, (171, 128)))
        changed_frames = changed_frames - np.array([[[90.0, 98.0, 102.0]]])
        fps.append(changed_frames)
        if len(fps) == 16:
            inputs=set_inputs(fps,device)
            with torch.no_grad():
                outputs = model(inputs)

            probs = torch.nn.Softmax(dim=1)(outputs)

            preds = torch.max(probs, 1)[1].detach().cpu().numpy()[0]

            print(cap.get(cv2.CAP_PROP_FRAME_WIDTH),cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cv2.putText(frame, class_names[preds].split(' ')[-1].strip(), (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1)
            cv2.putText(frame, "prob: %.4f" % probs[0][preds], (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)
            fps.pop(0)
        out.write(frame)
        cv2.imshow('show action label', frame)
        out.write(frame)
        cv2.waitKey(10)

    cap.release()
    out.release()
    cv2.destroyAllWindows()



main()