

import os
import re
os.environ["CUDA_VISIBLE_DEVICES"]="0"     #select the first GPU
from keras.models import model_from_json
import numpy as np
from keras.preprocessing import image


# load json and create model
json_file = open('CTC_neg_pos_150epoch_vgg16.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights('CTC_neg_pos_150epoch_vgg16.h5')

_nsre = re.compile('([0-9]+)')
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]

eval_class = 'positive'
class_dir = 'test_20191022_10_groups _2/G1/' + eval_class + '/'
all_pics = (os.listdir(class_dir))
all_pics.sort(key=natural_sort_key)

#ptm1 = []
ptm =[]
for ap in range(len(all_pics)):
#for ap in range(2,2260):
 #  img_path = class_dir + "2nd_stage_mitdb100_"+ str(ap)+".jpg"
 #  ptm1.append(img_path)
 #img_path = str(sys.argv[1])
    img_path = class_dir + all_pics[ap]
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preds = loaded_model.predict(x)
    pred_max = np.argmax(preds)
    ptm.append(pred_max)

    
print(ptm)
#print(all_pics)

s_0 = 0
s_1 = 0
#s_2 = 0
#s_3 = 0
#s_4 = 0
#s_5 = 0

for si in range(len(ptm)):
    if ptm[si] == 0:
        s_0 = s_0 + 1
    elif ptm[si] == 1:
        s_1 = s_1 + 1
  #  elif ptm[si] == 2:
      #  s_2 = s_2 + 1
    #elif ptm[si] == 3:
   #     s_3 = s_3 + 1
    #elif ptm[si] == 4:
     #   s_4 = s_4 + 1
    #elif ptm[si] == 5:
        #s_5 = s_5 + 1

        
print(s_0,s_1)
#sum_ptm = [s_0,s_1,s_2,s_3]
#path = '19_04_2nd_stage/test_results/mitdb200', ptm, '\n', sum_ptm
#np.savetxt(path)
