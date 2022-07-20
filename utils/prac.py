from deepface import DeepFace
import pandas as pd

project_path='/home/ljj/pixel2style2pixel/'

result=[]
for i in range(18):
    result.append(DeepFace.verify(img1_path=project_path+'datas/1.jpg',img2_path=project_path+'result/1/'+str(i)+'.jpg', enforce_detection=False))

distance=[]
verified=[]
for i in range(18):
    distance.append(result[i]['distance'])
    verified.append(result[i]['verified'])

result = pd.DataFrame(distance,verified)
print(result)