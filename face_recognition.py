import os
import sys
import cv2
import numpy as np
from datetime import datetime
from utils import text_spary, is_image_file

config_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 'config'
)
config_file_name = 'haarcascade_frontalface_default.xml'
config_file_path = os.path.join(config_dir, config_file_name)

class CaptureFace():
    '''
    얼굴 정보를 캡처하기위해 만들었음.
    '''
    timestamp = str(datetime.now()).replace(':', '').replace(' ', '_').split('.')[0]
    sample_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'faces',
        timestamp
    )
    sample_max = 100

    def __init__(self):
        '''
        설정정보 초기화
        '''
        if not os.path.isdir(config_dir):
            os.makedirs(config_dir)
        if not os.path.isdir(self.sample_dir):
            os.makedirs(self.sample_dir)
        if not os.path.isfile(config_file_name):
            import wget
            from utils import bar_progress
            '''
            check xml file from here
            https://github.com/opencv/opencv/tree/master/data/haarcascades
            '''
            url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
            text_spary('Configuration.. ')
            wget.download(url, bar=bar_progress)
            os.rename(config_file_name, config_file_path)
            text_spary('Configuration.. Done')

    def get_face_classifier(self):
        '''
        얼굴정보에 대한 분류기를 클래스에 초기화하고
        분류기를 반환하는 기능
        '''
        self.face_classifier = cv2.CascadeClassifier(str(config_file_path))
        return self.face_classifier
        
    def face_extractor(self, img, classifier):
        '''
        획득한 영상에서 얼굴만 잘라서 원하는 크기로 자르는 기능
        '''
        empty_face = ()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = classifier.detectMultiScale(gray, 1.3, 5)

        if faces == empty_face:
            return None

        for (x,y,w,h) in faces:
            cropped_face = img[y:y+h, x:x+w]

        return cropped_face
    
    def get_sample(self, max_count=100) -> str:
        '''
        얼굴 샘플 획득용
        '''
        try:
            face_classifier = self.face_classifier
        except Exception as e:
            '''
            NameError, AttributeError
            and the others
            '''
            face_classifier = self.get_face_classifier()
        cap = cv2.VideoCapture(0)
        count = 0

        while count < max_count:
            ret, frame = cap.read()
            if self.face_extractor(frame, face_classifier) is not None:
                count += 1
                face = cv2.resize(
                    self.face_extractor(frame, face_classifier),
                    (200, 200)
                )
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                file_name_path = os.path.join(self.sample_dir,
                                                '{}.jpg'.format(count))
                cv2.imwrite(file_name_path, face)
                print('{} / {} >> captured'.format(
                    count,
                    max_count
                ))
            else:
                print("Face not Found")

        cap.release()
        cv2.destroyAllWindows()
        text_spary('Sampling >> Done')
        return self.sample_dir


class ModelManager():
    '''
    About train and inference
    학습과 검증을 위한 클래스
    '''
    train_set, labels = list(), list()

    def __init__(self, path):
        self.data_path = path
        self.only_images = [
            f for
            f in
            os.listdir(self.data_path)
            if is_image_file(
                os.path.join(
                    self.data_path,
                    f
                )
            )
        ]
    
    def get_face_classifier(self):
        '''
        얼굴정보에 대한 분류기를 클래스에 초기화하고
        분류기를 반환하는 기능
        '''
        self.face_classifier = cv2.CascadeClassifier(str(config_file_path))
        return self.face_classifier

    def train(self, path):
        '''
        face 모델 학습을 위한 기능
        '''
        for index, files in enumerate(self.only_images):
            image_path = os.path.join(path, self.only_images[index])
            images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            self.train_set.append(np.asarray(images, dtype=np.uint8))
            self.labels.append(index)

        self.labels = np.asarray(self.labels, dtype=np.int32)

        model = cv2.face.LBPHFaceRecognizer_create()
        model.train(np.asarray(self.train_set), np.asarray(self.labels))

        text_spary("Model trained")
        return model
    
    def detect(self, frame, size=0.5):
        '''
        face 모델 검증을 위한 기능
        '''

        empty_face = ()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        try:
            face_classifier = self.face_classifier
        except Exception as e:
            '''
            NameError, AttributeError
            and the others
            '''
            face_classifier = self.get_face_classifier()
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        if faces is empty_face:
            return frame, []

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            roi = frame[y:y + h, x:x + w]
            roi = cv2.resize(roi, (200, 200))
            
        return frame, roi


if __name__ == '__main__':
    '''
    쉘에서 python 스크립트 실행시에만 작동하는 부분
    모듈로 사용할땐 실행되지 않음.
    '''
    text_spary('Please check your webcam and network status')
    while True:
        flag = input("You wanna try? (Y/n)")
        if flag == '' or flag == 'y' or flag == 'Y':
            break
        else:
            sys.exit(1)
    
    c = CaptureFace()
    path = c.get_sample()
    m = ModelManager(path)
    model = m.train(path)

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    image, face = m.detect(frame)
    
    while True:
        ret, frame = cap.read()
        image, face = m.detect(frame)
        try:
            face = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            result = model.predict(face)

            if result[1] < 500:
                confidence = int(100*(1-(result[1])/300))
                display_string = str(confidence)+'% 확률로 일치함'
            cv2.putText(image,display_string,(50, 50), cv2.FONT_HERSHEY_COMPLEX,1,(250,120,255),2)


            if confidence > 75:
                cv2.putText(image, "Hello", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Face detector', image)
            else:
                cv2.putText(image, "Lock", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Face detector', image)

        except:
            cv2.putText(image, "Face Not Found", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('Face detector', image)
            pass

        if cv2.waitKey(1) == 13 or cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
        

