# random-noise
It is the code of our article "What's the relationship between CNNs and communication systems?"

The random noised images can be made by "create_alpha.py", the hyperparameters can be changed inside it. 

The classification results can be calculated by "classify_alpha_0.9.py","classify_alpha_1.5.py" and "classify_alpha_2.py", the model used can be changed inside these codes.

The dataset can be easily downloaded by tensorflow 2.0, the code is "tf.keras.utils.get_file(origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz', fname='flower_photos', untar=True)"

This dataset need to be convert to 224*224 to test resnet50, vgg16 and vgg19.


Some examples:

salt-pepper noise(alpha=0.9):
![image](https://github.com/ghghgh0001/random-noise/tree/master/Example_Images/alpha0.9/1.jpg)
![image](https://github.com/ghghgh0001/random-noise/tree/master/Example_Images/alpha0.9/2.jpg)
![image](https://github.com/ghghgh0001/random-noise/tree/master/Example_Images/alpha0.9/3.jpg)
![image](https://github.com/ghghgh0001/random-noise/tree/master/Example_Images/alpha0.9/4.jpg)

alpha=1.5:
![image](https://github.com/ghghgh0001/random-noise/blob/master/Example_Images/alpha1.5/1.jpg)
![image](https://github.com/ghghgh0001/random-noise/blob/master/Example_Images/alpha1.5/2.jpg)
![image](https://github.com/ghghgh0001/random-noise/blob/master/Example_Images/alpha1.5/3.jpg)
![image](https://github.com/ghghgh0001/random-noise/blob/master/Example_Images/alpha1.5/4.jpg)

Gause noise(alpha=2.0):
![image](https://github.com/ghghgh0001/random-noise/blob/master/Example_Images/alpha2.0/1.jpg)
![image](https://github.com/ghghgh0001/random-noise/blob/master/Example_Images/alpha2.0/2.jpg)
![image](https://github.com/ghghgh0001/random-noise/blob/master/Example_Images/alpha2.0/3.jpg)
![image](https://github.com/ghghgh0001/random-noise/blob/master/Example_Images/alpha2.0/4.jpg)

triangle:
![image](https://github.com/ghghgh0001/random-noise/blob/master/Example_Images/triangle/1.jpg)
![image](https://github.com/ghghgh0001/random-noise/blob/master/Example_Images/triangle/2.jpg)
![image](https://github.com/ghghgh0001/random-noise/blob/master/Example_Images/triangle/3.jpg)
![image](https://github.com/ghghgh0001/random-noise/blob/master/Example_Images/triangle/4.jpg)

rhombus:
![image](https://github.com/ghghgh0001/random-noise/blob/master/Example_Images/rhombus/1.jpg)
![image](https://github.com/ghghgh0001/random-noise/blob/master/Example_Images/rhombus/2.jpg)
![image](https://github.com/ghghgh0001/random-noise/blob/master/Example_Images/rhombus/3.jpg)
![image](https://github.com/ghghgh0001/random-noise/blob/master/Example_Images/rhombus/4.jpg)

square:
![image](https://github.com/ghghgh0001/random-noise/blob/master/Example_Images/square/1.jpg)
![image](https://github.com/ghghgh0001/random-noise/blob/master/Example_Images/square/2.jpg)
![image](https://github.com/ghghgh0001/random-noise/blob/master/Example_Images/square/3.jpg)
![image](https://github.com/ghghgh0001/random-noise/blob/master/Example_Images/square/4.jpg)

