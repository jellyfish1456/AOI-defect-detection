# AOI-defect-detection

Automated Optical Inspection(AOI) is a critical technique which is used in the manufacture and test of electronics printed circuit boards, PCBs and so on. AOI defect detection allows us to inspect is there any defects on electronics assemblies or in particular PCBs fastly and accurately. It is the method to ensure that the quality of product can be built correctly and without manufacturing faults.

I use VGG16 and Densenet121 model to train the network.


# Content

  * [Environment](#Environment)
  * [Method](#Method)
  * [Honor](#Honor)
  * [Notice](#Notice)
  * [Reference](#Reference)
  

# Environment

   * Python: 3.6.5
   * Keras: 2.2.5
   * Tensorflow: 1.14.0
   
# Method
  1. Here we use the data from [Industrial Technology Research Institute - Aidea](https://aidea-web.tw/topic/a49e3f76-69c9-4a4a-bcfc-c882840b3f27) to classify the defect. Unzip the file, it includes:
  
      * train_images.zip： 2528 images.
      * test_images.zip：10142 images.
      * train.csv：two columns, ID and Label respectively.
      * test.csv：two columns, ID and Label respectively.
      * ID is for the name of the png file. Label is for the class（0: normal, 1: void, 2: horizontal defect, 3: vertical defect, 4: edge   defect, 5: particle）
  
  2. Create a folder. Put the file inside the floder. And create
  
      * Train_image
      * Test_image
      
  3. Run the py file.

# Honor

In private score gets  99.42500 % in accuracy.

# Notice

In order not to violate the rules of the competition, the code provided here is just the example of the code. Not the exact code which is uploaded in th competition. You can change the network structure on your own.

# Reference

https://aidea-web.tw/topic/a49e3f76-69c9-4a4a-bcfc-c882840b3f27

https://keras.io/applications/

https://keras.io/applications/#densenet
