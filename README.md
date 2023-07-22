### Purpose: Apply 3 different Normalization methods.

## Based on CiFAR 10 dataset
### Create 3 different one for each Normalization methods:- 
- Batch Normalization
- Group Normalization
- Linear Normalization

### Project Setup:
Clone the project as shown below:-

```bash
$ git clone git@github.com:pankaja0285/era_v1_session8_pankaja.git
$ cd era_v1_session8_pankaja
```
About the file structure</br>
|__config
   __config.yaml<br/>
|__data
|__data_analysis
|__data_loader
   __load_data.py<br/>
   __albumentation.py<br/>
|__models
   __model.py<br/>
|__utils
   __dataset.py<br/
   __engine.py<br/>
   __helper.py<br/>
   __plot_metrics.py<br/>
   __test.py<br/>
   __train.py<br/>
|__S8.ipynb<br/>
|__README.nd<br/>

**NOTE:** List of libraries required: ***torch*** and ***torchsummary***, ***tqdm*** for progress bar, which are installed using requirements.txt<br/>

One of 2 ways to run any of the notebooks, for instance **S8.ipynb** notebook:<br/>
1. Using Anaconda prompt - Run as an **administrator** start jupyter notebook from the folder ***era_v1_session8_pankaja*** and run it off of your localhost<br/>
**NOTE:** Without Admin privileges, the installs will not be correct and further import libraries will fail. <br/>
```
jupyter notebook
```
2. Upload the notebook folder ***era_v1_session8_pankaja*** to google colab at [colab.google.com](https://colab.research.google.com/) and run it on colab<br/>

###
**Context:** The drill down analysis starts from Step 0 through Step 6 and the analysis is laid out as Target, Results and Analysis. <br />
- Target: what is the target we are setting up as
- Results: what results we are getting
- Analysis: basically analyzing what we are doing in the model in this step, what is the algorithm etc.,

### Batch Normalization:
**File used: models/model.py, model with Net1 Class**
<p>
Target:
- create a model with Batch Normalization as the normalization method

Results:
- Total parameters: 52,576
- Train accuracy of  and test accuracy of 77.69

Analysis:
- To see how the accuracy is using Batch Normalization method.
</p>

### Group Normalization:
**File used: models/model.py, model with Net2 Class**
<p>
Target:
- create a model with Group Normalization as the normalization method

Results:
- Total parameters: 52,576
- Train accuracy of  and test accuracy of 77.69

Analysis:
- To see how the accuracy is using Group Normalization method.
</p>

### Linear Normalization:
**File used: models/model.py, model with Net3 Class**
<p>
Target:
- create a model with Linear Normalization as the normalization method

Results:
- Total parameters: 52,576
- Train accuracy of  and test accuracy of 77.69

Analysis:
- To see how the accuracy is using Linear Normalization method.
</p>
- [An Intuitive Explanation of Convolutional Neural Networks](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/)
- [Stanford CS231n](https://cs231n.github.io/convolutional-networks/)
- [Stanford CS231n Winter 2016 - Karpathy](https://youtu.be/NfnWJUyUJYU)

### Contributing:
For any questions, bug(even typos) and/or features requests do not hesitate to contact me or open an issue!
