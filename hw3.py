import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

###
class AdaptiveThreshold:
    def __init__(self,blockSize,C):
        self.filters = tf.ones((blockSize,blockSize,1,1),dtype=tf.float32)/blockSize**2
        self.C       = tf.constant(-C,dtype=tf.float32)
    def __call__(self,inputs): 
        # hint: tf.nn.conv2d, tf.where
        x = inputs-tf.nn.conv2d(inputs,self.filters,strides=(1,1),padding='SAME')
        x = tf.where(x>=self.C,tf.constant([[[1.]]],dtype=tf.float32),tf.constant([[[0.]]],dtype=tf.float32))
        return x
    
class Thinning:
    
    def __init__(self):   
        self.filters1,self.filters2 = self._surface_patterns() 

    @staticmethod
    def _surface_patterns():
        # generate the filters for rules 1 and 2
        pattern1 = []
        pattern2 = []
        image   = np.zeros((9,))    
        image[4]= 1
        clockwise=[(0,0),(0,1),(0,2),(1,2),(2,2),(2,1),(2,0),(1,0),(0,0)]
        for i in range(256):
            n = 0
            for j in range(4):
                image[j] = 2*(i//(2**j) % 2)-1
                if image[j] == 1:
                    n += 1
            for j in range(4,8):
                image[j+1] = 2*(i//(2**j) % 2)-1
                if image[j+1] == 1:
                    n += 1
            if n >= 2 and n <= 6:
                a = np.reshape(image,(3,3))
                sp = 0
                for j in range(8):
                    if a[clockwise[j][0],clockwise[j][1]]==-1 and a[clockwise[j+1][0],clockwise[j+1][1]]==1:
                        sp += 1
                if sp == 1:
                    if (a[0,1]==-1 or a[1,2]==-1 or a[2,1]==-1) and (a[1,2]==-1 or a[2,1]==-1 or a[1,0]==-1):
                        pattern1.append(np.reshape(a.copy(),(3,3,1,1)))

                    if (a[0,1]==-1 or a[1,0]==-1 or a[1,2]==-1) and (a[0,1]==-1 or a[2,1]==-1 or a[1,0]==-1):
                        pattern2.append(np.reshape(a.copy(),(3,3,1,1)))
                        
        #return filters1, filters2 # for rules 1 and 2
        return tf.constant(np.concatenate(pattern1,axis=-1),dtype=tf.float32),tf.constant(np.concatenate(pattern2,axis=-1),dtype=tf.float32)

 
    def __call__(self,inputs):
        #  do thinning
        #  padding is required
        x = tf.pad(inputs,tf.constant([[0,0],[1,1],[1,1],[0,0]]),constant_values=-1.0)
        while True:
            # tf.nn.conv2d, tf.math.reduce_max, tf.where 
            # add your code for rule 1
            z = tf.nn.conv2d(x, self.fliters1, strides=(1,1),padding='SAME')
            z = tf.math.reduce_max(z, axis=-1,keepdims=True)
            c1=tf.where(z==9,tf.constant([[[1.]]],dtype=tf.float32),tf.constant([[[0.]]],dtype=tf.float32))
            x =tf.where(z==9,tf.constant([[[-1.]]],dtype=tf.float32),x)

            # add your code for rule 2
            z = tf.nn.conv2d(x, self.fliters2, strides=(1,1),padding='SAME')
            z = tf.math.reduce_max(z, axis=-1,keepdims=True)
            c2=tf.where(z==9,tf.constant([[[1.]]],dtype=tf.float32),tf.constant([[[0.]]],dtype=tf.float32))
            x =tf.where(z==9,tf.constant([[[-1.]]],dtype=tf.float32),x)

            # if no pixels are changed from 1 to -1, break this loop            
            if tf.reduce_sum(c1) == 0 and tf.reduce_sum(cw)==0:
                    break;
        outputs = x[:,1:-1,1:-1,:]
        return outputs


#下載測試影像
url      = 'https://evatronix.com/images/en/offer/printed-circuits-board/Evatronix_Printed_Circuits_Board_01_1920x1080.jpg'
testimage= tf.keras.utils.get_file('pcb.jpg',url)

#讀入測試影像
inputs   = cv2.imread(testimage)

#轉成灰階影像
inputs   = cv2.cvtColor(inputs,cv2.COLOR_BGR2GRAY)

#顯示測試影像
plt.figure(figsize=(20,15))
plt.imshow(inputs,cmap='gray')
plt.axis(False)
plt.show()

#轉換影像表示方式成四軸張量(sample,height,width,channel)，以便使用卷積運算。
inputs = tf.convert_to_tensor(inputs,dtype=tf.float32)
inputs = inputs[tf.newaxis,:,:,tf.newaxis]

#使用卷積運算製作AdatpiveThresholding
binary = AdaptiveThreshold(61,-8)(inputs)

#存下AdaptiveThresholding結果
outputs = tf.where(tf.squeeze(binary)>0,tf.constant([[255]],dtype=tf.uint8),tf.constant([[0]],dtype=tf.uint8))
cv2.imwrite('pcb_threshold.png',outputs.numpy())

#顯示AdaptiveThresholding結果
plt.figure(figsize=(20,15))    
plt.imshow(tf.squeeze(binary).numpy()*255,cmap='gray')
plt.axis(False)
plt.show()

#使用卷積運算製作Thinning (改成-1代表0,1代表1會比較好算)
binary  = binary*2-1
outputs = tf.where(tf.squeeze(Thinning()(binary))>0,tf.constant([[255]],dtype=tf.uint8),tf.constant([[0]],dtype=tf.uint8))
outputs = tf.squeeze(outputs)

#存下細線化結果
cv2.imwrite('pcb_thinning.png',outputs.numpy())

#注意由於螢幕解析度，同學在螢幕上看到的細線化結果可能不是真正結果，此時必須看存下來的結果影像。
plt.figure(figsize=(20,15))        
plt.imshow(outputs.numpy(),cmap='gray')
plt.axis(False)
plt.show()