# Citation:
# Benjamin Shi (Bin Shi)
# April 18, 2020
#
# Noise2Noise for converting noisy SEM images to clean SEM images
# Following the codes in GitHub and make some small changes
https://github.com/NVlabs/noise2noise

(1) add folder "datasets"
	- kodak: validation datasets
	- test: test datasets
	- train.tfrecords: training datasets. 200 images with size 	  	  	  512*512. The 1st channel (in RGB) is edge image, the 2nd 
	  is noisy gray image, the 3rd is clean gray image.
	- validate.tfrecords: validation datasets in the form of 	  tfrecords. We use the images in kodak instead of this one.

(2) add folder "testresult" shows some test results in test datasets.

(3) config.py: change some long training params

(4) datasets.py: change the add_noise function. We have already add perlin noises, sparks, enhance contrast in the SEM datasets. The clean/dirty image pairs are read from train.tfrecords in the folder "datasets"

(5) train.py: change the loss function and data inread mode.

(6) add trained net model: network_SEMdenoising.pickle
    you can use that for test.

(6) add the trained model 
 
(5) validation.py: change the validation data inread mode.