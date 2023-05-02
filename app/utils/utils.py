import torch
from facenet_pytorch import MTCNN
import numpy as np
from PIL import Image
import cv2
import os
from utils.my_models import *
from torchvision import transforms
import torch.nn.functional as nnf


FE_MODEL_PATH = 'app/model_weights/efficientnet_affectnet.pt'
ENSEMBLE_PATH = 'app/model_weights/best_ensemble.pt'
VA_MODEL_PATH = 'app/model_weights/VA_0.469680.pt'
EX_MODEL_PATH = 'app/model_weights/EX_0.370579.pt'
AU_MODEL_PATH = 'app/model_weights/AU_0.507221.pt'
PHOTOS_PATH = 'app/photos'
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

MTCNN = MTCNN(keep_all=False, post_process=False, min_face_size=40, device=DEVICE)
FE_MODEL = torch.load(FE_MODEL_PATH)
VA_MODEL = ValenceArousal().to(device=DEVICE)
VA_MODEL.load_state_dict(torch.load(VA_MODEL_PATH, map_location=DEVICE))
EX_MODEL = Expression().to(device=DEVICE)
EX_MODEL.load_state_dict(torch.load(EX_MODEL_PATH, map_location=DEVICE))
AU_MODEL = ActionUnit().to(device=DEVICE)
AU_MODEL.load_state_dict(torch.load(AU_MODEL_PATH, map_location=DEVICE))
ENSEMBLE = Ensemble(model_va=VA_MODEL, model_ex=EX_MODEL, model_au=AU_MODEL)
IMG_SIZE = 224

class MTLModel:
    def __init__(self, FE_MDEOL_PATH=FE_MODEL_PATH, VA_MODEL_PATH=VA_MODEL_PATH,
                 EX_MODEL_PATH=EX_MODEL_PATH, AU_MODEL_PATH=AU_MODEL_PATH, device='cpu'):
        self.device = device
        self.is_mtl = True
        self.va_model_path = VA_MODEL_PATH
        self.ex_model_path = EX_MODEL_PATH
        self.au_model_path = AU_MODEL_PATH

        self.idx_to_class = {0: 'Neutral', 1: 'Anger', 2: 'Disgust', 3: 'Fear', 4: 'Happiness', 
                             5: 'Sadness', 6: 'Surprise', 7: 'Other'}
        self.idx_to_au = {0: 'AU1', 1: 'AU2', 2: 'AU4', 3: 'AU6', 4: 'AU7', 5: 'AU10,', 6: 'AU12',
                          7: 'AU15', 8: 'AU23', 9: 'AU24', 10: 'AU25', 11: 'AU26'}
        self.img_size=224
        self.test_transforms = transforms.Compose(
            [
                transforms.Resize((self.img_size,self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            ])
        
        if device == 'cpu':
            model=torch.load(FE_MODEL_PATH, map_location=torch.device('cpu'))
        else:
            model=torch.load(FE_MODEL_PATH)
        if isinstance(model.classifier,torch.nn.Sequential):
            self.classifier_weights=model.classifier[0].weight.cpu().data.numpy()
            self.classifier_bias=model.classifier[0].bias.cpu().data.numpy()
        else:
            self.classifier_weights=model.classifier.weight.cpu().data.numpy()
            self.classifier_bias=model.classifier.bias.cpu().data.numpy()
        
        model.classifier = torch.nn.Identity()
        model = model.to(device)
        self.model = model.eval()
        self.ensemble = self.init_ensemble()

    def init_ensemble(self):
        va_model = ValenceArousal().to(device=DEVICE)
        va_model.load_state_dict(torch.load(self.va_model_path, map_location=DEVICE))
        ex_model = Expression().to(device=DEVICE)
        ex_model.load_state_dict(torch.load(self.ex_model_path, map_location=DEVICE))
        au_model = ActionUnit().to(device=DEVICE)
        au_model.load_state_dict(torch.load(self.au_model_path, map_location=DEVICE))

        ensemble = Ensemble(model_va=va_model, model_ex=ex_model, model_au=au_model).to(device=DEVICE)
        return ensemble

    def get_probab(self, features, logits=True):
        xs = np.dot(features, np.transpose(self.classifier_weights)) + self.classifier_bias

        if logits:
            return xs
        else:
            e_x = np.exp(xs - np.max(xs, axis=1)[:,np.newaxis])
            return e_x / e_x.sum(axis=1)[:, None]
    
    def extract_features(self,face_img):
        img_tensor = self.test_transforms(Image.fromarray(face_img))
        img_tensor.unsqueeze_(0)
        features = self.model(img_tensor.to(self.device))
        features = features.data.cpu().numpy()
        return features
        
    def get_output(self, face_img):
        features = self.extract_features(face_img)
        scores = self.get_probab(features)
        model_input = np.concatenate((features, scores), axis=1)
        model_input = torch.tensor(data=model_input, dtype=torch.float, device=DEVICE)

        va_output, ar_output, expression_output, action_unit_output = self.ensemble(model_input)
        return va_output, ar_output, expression_output, action_unit_output

mtlmodel = MTLModel()

def detect_face(frame):
    bounding_boxes, probs = MTCNN.detect(frame, landmarks=False)
    if probs is None:
        print('!failed to detect face')
        return frame
    bounding_boxes = bounding_boxes[probs > .9]
    return bounding_boxes


def get_outputs_via_camera(camera, ensemble):
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            print('!failed to grab frame')
        img_name = os.path.join(PHOTOS_PATH, 'snap.jpg')
        if not cv2.imwrite(img_name, frame):
            raise Exception("!couldn't write an image")
        else:
            break
    img = cv2.imread(img_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bounding_boxes = detect_face(img)

    for bbox in bounding_boxes:
        box = bbox.astype(int)
        x1, y1, x2, y2 = box[0:4]
        face_img = img[y1:y2, x1:x2, :]
        # face_img = torch.tensor(data=face_img, dtype=torch.float, device=DEVICE)

        va_output, ar_output, expression_output, action_unit_output = mtlmodel.get_output(face_img)
        expression_prob = nnf.softmax(expression_output, dim=1)
        top_expression_prob, top_expression_class = expression_prob.topk(5, dim = 1)

        action_unit_prob = nnf.softmax(action_unit_output, dim=1)
        top_action_unit_prob, top_action_unit_class = action_unit_prob.topk(5, dim = 1)
        
        os.remove(img_name)
        return va_output.item(), ar_output.item(), top_expression_class.squeeze(0), top_expression_prob.squeeze(0), \
            top_action_unit_class.squeeze(0), top_action_unit_prob.squeeze(0)


def get_outputs_via_image(img_name, ensemble):
    img = cv2.imread(img_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bounding_boxes = detect_face(img)

    for bbox in bounding_boxes:
        box = bbox.astype(int)
        x1, y1, x2, y2 = box[0:4]
        face_img = img[y1:y2, x1:x2, :]
        # face_img = torch.tensor(data=face_img, dtype=torch.float, device=DEVICE)

        va_output, ar_output, expression_output, action_unit_output = mtlmodel.get_output(face_img)
        expression_prob = nnf.softmax(expression_output, dim=1)
        top_expression_prob, top_expression_class = expression_prob.topk(5, dim = 1)

        action_unit_prob = nnf.softmax(action_unit_output, dim=1)
        top_action_unit_prob, top_action_unit_class = action_unit_prob.topk(5, dim = 1)
        
        os.remove(img_name)
        return va_output.item(), ar_output.item(), top_expression_class.squeeze(0), top_expression_prob.squeeze(0), \
            top_action_unit_class.squeeze(0), top_action_unit_prob.squeeze(0)
        