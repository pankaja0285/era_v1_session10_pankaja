### Purpose: Deep dive into coding and applying different blocks in 7 steps.

## Based on MNIST dataset
### Create a simple Convolutional Neural Network model and predict

### Project Setup:
Clone the project as shown below:-

```bash
$ git clone git@github.com:pankaja0285/era_v1_session7_pankaja.git
$ cd era_v1_session7_pankaja
```
About the file structure</br>
|__era1_S7_0_BasicSetup.ipynb<br/>
|__era1_S7_1_BasicSkeleton.ipynb<br/>
|__era1_S7_2_Batch_Normalization.ipynb<br/>
|__era1_S7_3_Dropout.ipynb<br/>
|__era1_S7_4_Fully_Connected_layer.ipynb<br/>
|__era1_S7_5_Augmentation.ipynb<br/>
|__era1_S7_6_LRScheduler.ipynb<br/>
|__model.py<br/>
|__README.md<br/>
|__requirements.txt<br/>
|__utils.py<br/>

**NOTE:** List of libraries required: ***torch*** and ***torchsummary***, ***tqdm*** for progress bar, which are installed using requirements.txt<br/>

One of 2 ways to run any of the notebooks, for instance **era1_S7_0_BasicSetup.ipynb** notebook:<br/>
1. Using Anaconda prompt - Run as an **administrator** start jupyter notebook from the folder ***era_v1_session7_pankaja*** and run it off of your localhost<br/>
**NOTE:** Without Admin privileges, the installs will not be correct and further import libraries will fail. <br/>
```
jupyter notebook
```
2. Upload the notebook folder ***era_v1_session7_pankaja*** to google colab at [colab.google.com](https://colab.research.google.com/) and run it on colab<br/>

###
**Context:** The drill down analysis starts from Step 0 through Step 6 and the analysis is laid out as Target, Results and Analysis. <br />
- Target: what is the target we are setting up as
- Results: what results we are getting
- Analysis: basically analyzing what we are doing in the model in this step, what is the algorithm etc.,

### Step 0:
**File used: era1_S7_0_BasicSetup.ipynb**
<p>
Target:
- create a model structure set up

Results:
- Total parameters are of the order of ~6 Million
- Train accuracy of  and test accuracy of 

Analysis:
- This base model Model_1 is just so we establish a structure set up and not be concerned<br />
with the train and test results.
</p>

### Step 1:
**File used: era1_S7_1_BasicSkeleton.ipynb**
<p>
Target:
- Establish the basic skeleton in terms of convolution and placement of transition blocks such as max pooling, 1x1's
- Attempting to reduce the number of parameters as low as possible
- Adding GAP and remove the last BIG kernel.

Results:
- Total parameters: 4572
- Best Training accuracy: 98.22
- Best Test accuracy: 98.43

Analysis:
- Structured the model as a new model class 
- The model is lighter with less number of parameters 
- The performace is reduced compared to previous models. Since we have reduced model capacity, this is expected, the model has capability to learn.   
- Next, we will be tweaking this model further and increase the capacity to push it more towards the desired accuracy.
</p>

### Step 2:
**File used: era1_S7_2_Batch_Normalization.ipynb**
<p>
Target:
- Add Batch-norm to increase model efficiency.

Results:
- Parameters: 5,088
- Best Train Accuracy: 99.02%
- Best Test Accuracy: 99.03%

Analysis:
- There is slight increase in the number of parameters, as batch norm stores a specific mean and std deviation for each layer.
- Model overfitting problem is rectified to an extent. But, we have not reached the target test accuracy 99.40%.
</p>

### Step 3:
**File used: era1_S7_3_Dropout.ipynb**
<p>
Target:
- Add Batch-norm to increase model efficiency.

Results:
- Parameters: 5,088
- Best Train Accuracy: 99.02%
- Best Test Accuracy: 99.03%

Analysis:
- There is slight increase in the number of parameters, as batch norm stores a specific mean and std deviation for each layer.
- Model overfitting problem is rectified to an extent. But, we have not reached the target test accuracy 99.40%.
</p>

### Step 4:
**File used: era1_S7_4_Fully_Connected_layer.ipynb**
<p>
Target:
- Add Regularization Dropout to each layer except last layer.

Results:
- Parameters: 5,088
- Best Train Accuracy: 97.94%
- Best Test Accuracy: 98.64%

Analysis:
- There is no overfitting at all. With dropout training will be harder, because we are droping the pixels randomly.
- The performance has droppped, we can further improve it.
- But with the current capacity,not possible to push it further.We can possibly increase the capacity of the model by adding a layer after GAP.
</p>

### Step 5:
**File used: era1_S7_5_Augmentation.ipynb**
<p>
Target:
- Add various Image augmentation techniques, image rotation, randomaffine, colorjitter

### Results:
- Parameters: 6124
- Best Training Accuracy: 97.61
- Best Test Accuracy: 99.32%

### Analysis:
- he model is under-fitting, that should be ok as we know we have made our train data harder. 
- However, we haven't reached 99.4 accuracy yet.
- The model seems to be stuck at 99.2% accuracy, seems like the model needs some additional capacity towards the end.
</p>

### Step 6:
**File used: era1_S7_6_LRScheduler.ipynb**
<p>
Target:
- Add some capacity (additional FC layer after GAP) to the model and added LR Scheduler

Results:
- Parameters: 6720
- Best Training Accuracy: 99.43
- Best Test Accuracy: 99.53

Analysis:
- The model parameters have increased
- The model is under-fitting. This is fine, as we know we have made our train data harder.  
- LR Scheduler and the additional capacity after GAP helped getting to the desired target 99.4, Onecyclic LR is being used, this seemed to perform better than StepLR to achieve consistent accuracy in last few layers
</p>

### Python script files - details:
**model.py** - This has Model_1, Model_2, Model_3, Model_4, Model_5, Model_6, Model_7 <br />
in all 7 models to achieve an train accuracy = 99.5 and test accuracy = 98.6

*The illustration below shows how many layers are in this Convolution Neural Network that we have based upon and its details:-*
![CNN diagram used](cnn_28_x_28.png)

**utils.py** - This file contains the following main functions
* get_device() - checks for device availability for cuda, if not gives back cpu as set device
* plot_sample_data() - plots a sample grid of random 12 images from the training data
* plot_metrics() - plots the metrics - train and test - losses and accuracies respectively
* show_summary() - displays the model summary with details of each layer
* download_train_data() - downloads train data from MNIST
* download_test_data() - downloads test data from MNIST
* create_data_loader() - common data loader function using which we create both train_loader and test_loader by appropriately passing required arguments
* train_and_predict() - trains the CNN model on the training data and uses the trained model to predict on the test data

### More resources:
Some useful resources on MNIST and ConvNet:

- [MNIST](http://yann.lecun.com/exdb/mnist/)
- [Colah's blog](https://colah.github.io/posts/2014-10-Visualizing-MNIST/)
- [FloydHub Building your first ConvNet](https://blog.floydhub.com/building-your-first-convnet/)
- [How Convolutional Neural Networks work - Brandon Rohrer](https://youtu.be/FmpDIaiMIeA)
- [An Intuitive Explanation of Convolutional Neural Networks](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/)
- [Stanford CS231n](https://cs231n.github.io/convolutional-networks/)
- [Stanford CS231n Winter 2016 - Karpathy](https://youtu.be/NfnWJUyUJYU)

### Contributing:
For any questions, bug(even typos) and/or features requests do not hesitate to contact me or open an issue!
