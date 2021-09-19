# Dataset of Perceived Intoxicated Faces for Drunk Person Identification
This repository contains the code for the deep learning models in the paper Dataset of Perceived Intoxicated Faces for Drunk Person Identification by Vineet Mehta, Devendra Pratap Yadav, Sai Srinadhu Katta and Abhinav Dhall.

This repository contains the code for the feature extraction and the experiments. For dataset sharing and creation, check our website-[https://sites.google.com/view/difproject/home](https://sites.google.com/view/difproject/home)
## Major libraries used
1. **Common Libraires**
    1. tqdm
    2. numpy
    3. opencv
    4. matplotlib
3. **Specific Libraries**
    1. **Audio**
        1. keras
        2. openSMILE- Command line- [https://audeering.github.io/opensmile/get-started.html](https://audeering.github.io/opensmile/get-started.html)
    3. **CNN_RNN**
        1. keras
    4. **3D CNN and  variants**
        1. Pytorch
   
## Files/Folders description
1. **Audio_models** - This folder contains the code for the audio feature extraction and training code for all audio models.
2. **CNN_RNN** - This folder contains the code for the visual feature extraction and the training code for the CNN_RNN models.
3. **3D_CNN_and_variants** - This folder contains the code for the training of 3D CNN models and it's various variants.
4. **split.csv** - Data split used in the experiments. 
5 **test.ipynb** - This jupyter notebook contains the testing code for the best models and the ensemble startegy as discussed in the paper. The final hyperparameter setting and the model configuration can be seen in this notebook. 
6. **3D_pred.csv** - This file contains the predictions by the best 3D CNN varaints. 



