# **Trade Promotion Deductions**

## Introduction
According to Investopedia [Accounts receivable (AR)](https://www.investopedia.com/terms/a/accountsreceivable.asp) are the balance of money due to the company for goods delivered but not yet paid for by its clients. It represents a credit extended by the company to the client and normally has terms that require payments due within a time period. Account Receivable is an important aspect of businesses since it represents an asset, so it measures a company's liquidity or ability to cover short-term obligations without additional cash. One of the critical financial processes for a consumer products company is AR’s resolution of Trade Promotion Deductions (a.k.a. claims); It is a complex process that involves many parties and systems and requires time to assess and validate. Using AI to accelerate the resolution of AR deductions will directly impact the ability of consumer product companies to access cash.

## **Table of Contents**
 - [Purpose](#Purpose)
 - [Reference Solution](#proposed-solution)
 - [Reference Implementation](#Reference-Implementation)
 - [Intel® Implementation](#optimizing-the-E2E-solution-with-Intel®-oneAPI)
 - [Performance Observations](#performance-observations)

## Purpose
A Client (i.e. a Retailer) may have hundreds of promotions that a Consumer Product company may offer for them to use. Clients need to properly follow the promotion for them to get the deduction (a.k.a. claims) applied after the Consumer Product company validates. Getting, sorting and storing client’s claims is a significant time-consuming task that is part of the deduction validation process that the Consumer Product Company’s AR team conducts. After a client send a payment and applies a deduction, to validate the claim the AR Analyst will need to get claim information from different sources such as client website, client’s email or client’s claims systems where client’s documents their claims. The information will come in different formats which makes the documentation very challenging. 

To fully automate this process, Consumer Product companies would need to be able to address all the below pain points:
- Support extraction of claim documents multiple formats from different sources like Scanned Documents, emails, websites, images.
- Support claim content capture, reading, analysis of content.
- Support claim categorization, so claims can be stored and matched in a second stage.

For the context of this reference solution, we have limited the experiment only to scanned documents extraction and claim categorization

## Reference Solution
In this reference kit, A 2-stage AI solution has been proposed to extract the useful information from claim documents using the OCR and categorize the claims into a certain category. Two deep learning models have been built to perform text extraction from documents in first stage and for promotion claim classification in second stage. We also focus on below critical factors:
- Faster model development and 
- Performance efficient model inference and deployment mechanism.

Typically, in deep learning scenarios such as this type of use cases, GPUs are a natural choice since they achieve a higher FPS rate; however, they are also very expensive and memory consuming. Therefore, our experiment offers an alternative on CPU by applying Intel's model quantization which compresses the models using quantization techniques while maintaining the accuracy and speeding up the inference time of text extraction from documents and further classification of claims

The experiment consists of two connected parts, an OCR extraction module, and a classification module.
  1. In a typical deep learning based OCR solution, pretrained text detection model gives good accuracy on detecting text regions in the images but text recognition model is generally finetuned on the given dataset. Accuracy of text recognition model is dependent on the different properties of text like font, font-size, font color and handwritten or printed text etc. Hence it is advised to finetune the text recognition model on given dataset instead of using pretrained model to observe more accuracy in OCR pipeline.
  In this reference solution, the OCR extraction module consists of a pre-trained text detection model EasyOCR python library (installation steps can be found in this link https://pypi.org/project/easyocr), used for Text Detection in the claim document images, and a custom CRNN model, which has been trained for Text Recognition, using a synthetically generated dataset. Hyper-parameter tuning has been used to optimize the CRNN model with different learning rate for checking how quickly the model is adapted to the problem to increase the model performance. 
  To demonstrate the reusability and adaptability of our reference kits on helping build more advanced solutions, for the OCR extraction module we have used another Reference Kit (github link pending) as a reference solution.
  2. The classification module consists of a CNN-based custom classification model, which has been trained for claim classification, using a synthetically generated dataset. Hyper-parameter tuning has been used to optimize the CNN model with different learning rate for checking how quickly the model is adapted to the problem to increase the model performance.

Finally, the end-to-end pipeline has been created by connecting the two modules to perform inference of the overall end-to-end pipeline, with further quantization to reduce the size of the models whilst maintaining the accuracy and speeding up the inference time. The time required for training the models, inference time and accuracy of the models are captured for multiple runs on the stock version as well on the Intel OneAPI version. The average of these runs is considered and the comparison has been provided. This sample code is implemented for CPU using the Python language and Intel® PyTorch Extension 1.12.100.

### **Key Implementation Details**

The reference kit implementation is a reference solution to the OCR text extraction use case that includes 

  1. A reference E2E architecture to arrive at an AI solution with custom CRNN (Convolutional Recurrent Neural Network) with PyTorch 1.12.0  for text extraction and  with CNN (Convolutional Neural Network) for claim classification with Tensorflow 2.8.0.
  from documents.
  2. An Optimized reference E2E architecture enabled with Intel® Extension for PyTorch 1.12.100 and Tensorflow 2.9.1
  3. e2e pipeline for trade promotion.

## **Reference Implementation**

### **Use Case E2E flow**

![Use_case_flow](assets/workflow_stock.png)

### Expected Input-Output

**Input**                                 | **Output** |
| :---: | :---: |
| Claim Document Image          | Claim Valid/Invalid|

**Example Input**                                 | **Example Output** |
| :---: | :---: |
|Invoice Image |Valid/Invalid  

### ***Software Requirements***

1. Python - 3.9.13
2. PyTorch - 1.12.0
3. Tensorflow - 2.8.0

### ***Solution Setup Prerequisite***
Note that this reference kit implementation already provides the necessary scripts to setup the software requirements. 
To utilize these environment scripts, first install Anaconda/Miniconda by following the instructions at the following link

https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
or
[Anaconda installation](https://docs.anaconda.com/anaconda/install/linux/)

### ***Solution Setup***
Clone the repository to desired location on your compute using the command given below:
```sh
git clone https://github.com/oneapi-src/invoice-to-cash-automation.git
cd invoice-to-cash-automation
```
> Note: It is assumed that the present working directory is the root directory of this code repository. You can verify the present working directory location using following command.
```sh
$pwd
```
```
output: 
<Absolute path to cloned directory>/invoice-to-cash-automation
```
Follow the below conda installation commands to setup the Stock environment along with the necessary packages for this model training and prediction.
```sh
conda env create --file env/stock/tradepromotion-stock.yml
```
This command utilizes the dependencies found in the `env/stock/tradepromotion-stock.yml` file to create an environment as follows:

**YAML file**                       | **Environment Name**         |  **Configuration** |
| :---: | :---: | :---: |
| `env/stock/tradepromotion-stock.yml`             | `tradepromotion-stock` | Python=3.9.13 with PyTorch=1.12.0 & Tensorflow=2.8.0

Use the following command to activate the environment that was created:
*Activate stock conda environment*
```sh
conda activate tradepromotion-stock
```
##  **Synthetic Dataset Generation**
In this reference kit, synthetic dataset is generated using a reference dataset and can be found at https://www.kaggle.com/datasets/PromptCloudHQ/flipkart-products. Instructions for downloading the data for use can be found under the data folder in the repo. \
Once the reference dataset is downloaded using above steps, synthetic datasets for classification model and OCR model can be generated using following steps :
```
usage: python src/dataset_gen.py [-i][-d][-n]
```

Arguments:<br>
```
--input_csv_path, -i       Path of reference input csv dataset (Kaggle Flipkart Dataset)
--output_dir_path, -d      Path of output directory to generate the dataset 
--number_of_invoices, -n   Number of invoice images to be generated (OCR Dataset)
```

<b>Example</b>: 
```sh
python src/dataset_gen.py -i ./data/flipkart_com-ecommerce_sample.csv -d ./data -n 300
```
The above command will generate the OCR and classification dataset inside the ./data folder(given as output directory).

##  **Reference Implementation for the OCR Module**

### Dataset
| **Use case** | OCR Text Extraction
| :--- | :---
| **Dataset** | Synthetically Created Dataset
| **Size** | Total 300 Labelled Images<br>
| **Train Images :** | 6000 (Text ROI Cropped Images)
| **Test Images :** | 100
| **Input Size** | 32x280

The dataset used for this reference kit has been created synthetically. In this dataset, each image has certain texts in it. The ground truth text file is created with each image path and its respective words in the image.

> The dataset is generated and moved to the data/ocr_module folder using the above given steps in dataset generation section. Dataset contains 
train.txt and test.txt files to be used for training and inference respectively.These file contains path to images and their respective labels.

### **Reference Sources**
*Case Study* *Github repo*: https://github.com/courao/ocr.pytorch.git<br>

### ***Solution Implementation***

#### **Model Building Process**
#### **Model Training**
The Python script given below to be used to start training in the active conda environment enabled using 
the given steps above. 

Pretrained model CRNN-1010.pth has to be downloaded from https://drive.google.com/drive/folders/1hRr9v9ky4VGygToFjLD9Cd-9xan43qID and saved to .models/ocr_module/ folder path. The below command can be used to download the model.
```
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=15TNHcvzNIjxCnoURkK1RGqe0m3uQAYxM' -O CRNN-1010.pth
mv CRNN-1010.pth ./models/ocr_module/CRNN-1010.pth
```
```
python src/ocr_module/ocr_train.py [-i ]
```
Arguments:<br>
```
--intel, -i       1 for enabling Intel PyTorch Optimizations and 0 for disabling Intel optimizations. The default value is 0.
```

<b>Example</b>: 
```sh
python src/ocr_module/ocr_train.py -i 0
```
In this example, as the environment is stock environment, -i parameter is set to 0.
The model generated will be saved in ./src/crnn_models folder with the prefix CRNN-1010_WOHP.

### ***Hyper-parameter Analysis***

In realistic scenarios, an analyst will train the text extraction model multiple times on the same dataset, scanning across different hyper-parameters.  
To capture this, we measure the total amount of time it takes to generate results across different hyper-parameters for a fixed algorithm, which we 
define as hyper-parameter analysis.  In practice, the results of each hyper-parameter analysis provides the analyst with many different models that 
they can take and further analyse to choose the best model.

The below table provide details about the hyper-parameters & values used for hyperparameter tuning in our benchmarking experiments:
| **Algorithm**                     | **Hyper-parameters**
| :---                              | :---
| Custom CRNN Architecture          | learning rate, epochs, batch size

 Using different learning rates and batch sizes to the model architecture on the dataset, also we increased the number of epochs to reach maximum accuracy on the training set. Hyperparameters considered for tuning are `Learning Rate, Epochs, batch-size.

#### **Hyper-parameter Tuning**
The Python script given below needs to be executed to start hyper-parameter tuned training

The model generated using regular training from previous steps can be used as pretrained model. Hyper-parameters considered for tuning are 
learning rate, epochs and batch size.

```
usage: python src/ocr_module/ocr_train_hp.py [-i ]
```

Arguments:<br>
```
--intel, -i        1 for enabling Intel PyTorch Optimizations & 0 for disabling Intel PyTorch Optimizations. The default value is 0.
```

<b>Example</b>: 
```sh
python src/ocr_module/ocr_train_hp.py -i 0
```
In this example, as the environment is stock environment, -i parameter is given a value of 0. 
The model gets tuned with hyperparameters and the generated model will be saved in ./src/crnn_models folder with the prefix CRNN-1010_HP.

##  **Reference Implementation for the Classification  Module**

### Dataset
| **Use case** | Claim Classification Dataset
| :--- | :---
| **Dataset** | Synthetically generated dataset
| **Size** | 16k rows<br>
| **No of columns for training:** | 6
| **No of classes :** | 10


Here invoice id, vendor code, GL code, invoice amount, discount amount column has been synthetically generated. Description and Claim Category has been taken from following dataset on Kaggle. 
https://www.kaggle.com/datasets/PromptCloudHQ/flipkart-products
Dataset is available in data/classification_module folder.

### **Reference Sources**
*Case Study* *Github repo*: https://www.kaggle.com/code/vonhammm/invoice-classification-cnn-tensorflow<br>

### ***Solution Implementation***

#### **Model Building Process**
#### **Model Training**
The Python script given below needs to be used to start training.

```
usage: python src/classification_module/tp_inv_cls_training.py [-hp][-l][-p][-o]
```
Arguments:<br>
```
--hp_tuning, -hp          1 for enabling Hyperparameter Tuning 
--logs, -l                Path where logs need to be saved.
--data_path,-p            Training dataset csv path
--model_output_path,-o    Model output path
```

<b>Example</b>: 
```sh
python src/classification_module/tp_inv_cls_training.py -hp 0 \
 -l logs/cf_model_train_logs_stock.txt \
 -p data/classification_module/classification_data.csv \
 -o models/classification_module
```

### ***Hyper-parameter Analysis***

In realistic scenarios, an analyst will train the classification model multiple times on the same dataset, scanning across different hyper-parameters.  
To capture this, we measure the total amount of time it takes to generate results across different hyper-parameters for a fixed algorithm, which we 
define as hyper-parameter analysis. In practice, the results of each hyper-parameter analysis provides the analyst with many different models that 
they can take and further analyse to choose the best model.

The below table provide details about the hyperparameters & values used for hyperparameter tuning in our benchmarking experiments:
| **Algorithm**                     | **Hyper-parameters**
| :---                              | :---
| Custom CNN Architecture           | epochs, batch size

 Introduced different epochs and batch sizes to the model architecture on the dataset, also we increased the number of epochs to reach maximum accuracy on the training set. Hyperparameters considered for tuning are Epochs and batch size.
 
#### **Hyperparameter Tuning**
The Python script given below needs to be used to start the training with hyper-parameter tuning.

The model generated using regular training from previous steps can be used as pretrained model. Hyper-parameters considered for tuning are 
epochs and batch size.

```
usage: python src/classification_module/tp_inv_cls_training.py [-hp][-l]
```

Arguments:<br>
```
--hp_tuning, -hp          1 for enabling Hyperparameter Tuning
--logs, -l                Path where logs need to be saved.
--data_path,-p            Training dataset csv path
--model_output_path,-o    Model output path
```

<b>Example</b>: 
```
python src/classification_module/tp_inv_cls_training.py -hp 1 \
 -l logs/cf_model_train_logs_stock_hp.txt \
 -p data/classification_module/classification_data.csv \
 -o models/classification_module
```

#### **Reference Implementation for End to End Inference Pipeline**
The Python script given below is to be used to execute end to end inference pipeline using OCR & Classification models generated in the above training steps. This pipeline is used to 
predict claim category on a given invoice image.The path/s of image/s, on 
which the inference has to be performed is given as input and the output is the claim category.

```
usage: python src/e2e_pipeline/inference.py [-i][-m][-l][-p][-b]
```

Arguments:<br>
```
  -i, --intel               Give 1 for enabling intel PyTorch optimizations, default is 0
  -m, --ocr_model_path      Absolute path to the ocr model  
  -l, --cls_model_path      Absolute path to the classification model
  -p, --test_data_path      Path of test dataset
  -b, --batch_size          batch size
```

<b>Example</b>:
```sh
python src/e2e_pipeline/inference.py -i 0 \
 -m ./models/ocr_module/CRNN-1010-WOHP_stock.pth \
 -l ./models/classification_module/tp_inv_cls_model_without_HP/tp_inv_cls_model_without_HP.pb \
 -p data/e2e_pipeline/invoice_dataset \
 -b 1
```
> As the dataset used in this reference kit is synthetically generated, test dataset has already been provided at data/e2e_pipeline path to run the 
end to end inference pipeline.\

## **Optimizing the E2E solution with Intel Optimizations for Trade Promotion Deductions**
Although AI delivers a solution to address Trade Promotion, on a production scale implementation with millions 
or billions of records demands for more compute power without leaving any performance on the table. Under this scenario, 
a text extraction models and the claim classification models are essential for identifying and extracting text and classifying the claims which will enable analyst to take
appropriate decisions. In order to derive the most insightful and beneficial actions to take, 
they will need to study and analyse the data generated through various feature sets and algorithms, thus requiring frequent 
re-runs of the algorithms under many different parameter sets. To utilize all the hardware resources efficiently, Software 
optimizations cannot be ignored.   
 
This reference kit solution extends to demonstrate the advantages of using the Intel® AI Analytics Toolkit on the task of 
building a pipeline for trade promotion deductions. The savings gained from using the Intel® optimizations for PyTorch and Intel® optimized Tensorflow can 
lead an analyst to more efficiently explore and understand data, leading to better and more precise targeted solutions.

#### Use Case E2E flow
![image](assets/workflow_intel.png)

### ***Software Requirements***
| **Package**                    | **Intel® Python**
| :---                           | :---
| Python                         | 3.9.13
| Intel® Extension for PyTorch   | 1.12.100
| Tensorflow                     | 2.9.1

### ***Optimized Solution Setup***
Follow the below conda installation commands to setup the Intel environment along with the necessary packages for this model training and prediction.
>Note: It is assumed that the present working directory is the root directory of this code repository

```shell
conda env create --file env/intel/tradepromotion-intel.yml
```
This command utilizes the dependencies found in the `env/intel/tradepromotion-intel.yml` file to create an environment as follows:

**YAML file**                                 | **Environment Name** |  **Configuration** |
| :---: | :---: | :---: |
| `env/intel/tradepromotion-intel.yml`             | `tradepromotion-intel` | Python=3.9.13 with Intel® Extension for PyTorch 1.12.100 & Tensorflow 2.9.1

For the workload implementation to arrive at first level solution we will be using the intel environment.

Use the following command to activate the environment that was created:
```shell
conda activate tradepromotion-intel
```
## **Optimizing the OCR Module with Intel Optimizations**
### **Optimized Software Components**
Intel® Extension for PyTorch (version 1.12.100) framework has been optimized using oneAPI Deep Neural Network Library (oneDNN) primitives, a popular performance 
library for deep learning applications. It provides accelerated implementations of numerous popular DL algorithms that optimize performance on 
Intel® hardware with only requiring a few simple lines of modifications to existing code.

**Intel® Extension for PyTorch** IPEX extends PyTorch with Optimizations for extra performance boost on Intel hardware
The below changes have been done to the stock PyTorch training code base to utilize the Intel® Extension for PyTorch* performance.
One can enable the intel flag to incorporate below Intel PyTorch optimizations.
```
import intel_extension_for_pytorch  as ipex

model, optimizer=ipex.optimize(crnn,optimizer=optimizer)
```

### ***Optimized Solution Implementation***

#### **Optimized Model Building Process**
#### **Model Training**
The Python script given below need to be executed to start training in intel conda environment.

```sh
python src/ocr_module/ocr_train.py [-i]
```
Arguments:<br>
```
--intel, -i               1 for enabling Intel® Extension for PyTorch Optimizations, default is 0
```

<b>Example</b>: 
```sh
python src/ocr_module/ocr_train.py -i 1
```
In this example, as the environment is intel environment, -i parameter is set to 1 to enable intel optimizations in the code.
The model generated will be saved in ./src/crnn_models folder with the prefix CRNN-1010_WOHP

#### **Hyperparameter Tuning**
The Python script given below need to be executed to start hyper-parameter tuned training in the active environment enabled by using the above steps. 

The model generated using regular training from previous steps will be used as pretrained model. Hyper-parameters considered for tuning are 
epochs and batch size. 

```
usage: python src/ocr_module/ocr_train_hp.py [-i][-b]
```

Arguments:<br>
```
--intel, -i               1 for enabling Intel® PyTorch Optimizations, default is 0
```

<b>Example</b>: 
```sh
python src/ocr_module/ocr_train_hp.py -i 1
```
In this example, as the environment is intel environment, -i parameter is set to 1 to enable intel optimizations in the code 
and the model is saved in ./src/crnn_models folder with the prefix CRNN-1010_HP.

## **Optimizing the Classification Module with Intel Optimizations**
### **Optimized Software Components**
Tensorflow (version 2.9.1) framework has been optimized using oneAPI Deep Neural Network Library (oneDNN) primitives, a popular performance 
library for deep learning applications. It provides accelerated implementations of numerous popular DL algorithms that optimize performance on 
Intel® hardware with only requiring a few simple lines of modifications to existing code.

### ***Optimized Solution Implementation***

#### **Optimized Model Building Process**
#### **Model Training**
The Python script given below to be executed to start training.

```sh
python src/classification_module/tp_inv_cls_training.py [-hp][-l][-p][-o]
```
Arguments:<br>
```
--hp tuning, -hp          1 for enabling hp tuning, default is 0
--logs, -l                Path where logs need to be saved.
--data_path,-p            Training dataset csv path
--model_output_path,-o    Model output path
```

<b>Example</b>: 
```sh
python src/classification_module/tp_inv_cls_training.py -hp 0 \
 -l logs/cf_model_train_logs.txt \
 -p data/classification_module/classification_data.csv \
 -o models/classification_module
```

#### **Hyperparameter Tuning**
The Python script given below need to be executed to start training with hyper-parameter tuninig training. 

The model generated using regular training from previous steps can be used as pretrained model. Hyper-parameters considered for tuning are 
epochs and batch size. 

```
python src/classification_module/tp_inv_cls_training.py [-hp][-l][-p][-o]
```

Arguments:<br>
```
--hp tuning, -hp          1 for enabling hp tuning, default is 0
--logs, -l                Path where logs need to be saved.
--data_path,-p            Training dataset csv path
--model_output_path,-o    Model output path
```

<b>Example</b>: 
```sh
python src/classification_module/tp_inv_cls_training.py -hp 1 \
 -l logs/cf_model_training_logs_intel.txt \
 -p data/classification_module/classification_data.csv \
 -o models/classification_module
```
In this example, as the environment is Intel environment, the model gets tuned with hyperparameters & is saved in the models folder.

#### **Intel Optimized Implementation for End to End Inference Pipeline**
The Python script given below is to be used to execute end to end inference pipeline using OCR & Classification models generated in the above training steps. This pipeline is used to 
predict claim catagory on a given invoice image.The path/s of image/s, on 
which the inference has to be performed is given as input and the output is the claim category.

```
usage: python src/e2e_pipeline/inference.py [-i][-m][-l][-p][-b]
```

Arguments:<br>
```
  -i, --intel               Give 1 for enabling intel PyTorch optimizations, default is 0
  -m, --ocr_model_path      Absolute path to the ocr model  
  -l, --cls_model_path      Absolute path to the classification model
  -p, --test_data_path      Path of test dataset
  -b, --batch_size          batch size
```

<b>Example</b>:
```sh
python src/e2e_pipeline/inference.py -i 1 \
 -m ./models/ocr_module/CRNN-1010-WOHP_intel.pth \
 -l ./models/classification_module/tp_inv_cls_model_without_HP/tp_inv_cls_model_without_HP.pb \
 -p data/e2e_pipeline/invoice_dataset \
 -b 1
```
In this example, -i is set 1 to enable intel optimizations in the pipeline.
> As dataset used in this reference kit is synthetically generated, test dataset has already been provided at data/e2e_pipeline path to run the 
end to end inference pipeline.\

#### **Model Conversion using Intel® Neural Compressor**
Intel® Neural Compressor is used to quantize the FP32 Model to the INT8 Model.
Intel® Neural Compressor supports many optimization methods. Here, we have used post training optimization with `Default quanitization mode` method to 
quantize the FP32 model.

The below script is used to convert the FP32 model to INT8 quantized model for OCR Intel® Neural Compressor. Run the below script after activating the Intel conda environment.
```
usage: python src/ocr_module/neural_compressor_conversion.py [-m modelpath] [-o output_path]
```

Arguments:
```
  -m, --modelpath                 path of the FP32 model
  -o, --output_path               output path for int8 model
```

<b>Example</b>
```sh
python src/ocr_module/neural_compressor_conversion.py -m ./models/ocr_module/CRNN-1010-WOHP_intel.pth -o models/ocr_module
```

The below script is used to convert the FP32 model to INT8 quantized model for classification module using Intel® Neural Compressor. Run the below script after activating the Intel environment.
```
usage: python src/classification_module/inc_quantizer.py [-p] [-m] [-o][-l]
```

Arguments:
```
  -p, --test_dataset_path                 input csv file path
  -m, --fp32_model_path                   already trained FP32 model path
  -o, --output_model_path                 output quantized Int8 model path
  -l, --logfile                           logfile path
```

<b>Example</b>
```sh
python src/classification_module/inc_quantizer.py -p data/classification_module/classification_data.csv \
-m ./models/classification_module/tp_inv_cls_model_without_HP/tp_inv_cls_model_without_HP.pb \
-o ./models/classification_module \
-l ./logs/log.txt
```

#### **Optimized Implementation of End to End Inference Pipeline using Intel® Neural Compressor Quantized Int8 Models**
The Python script given below needs to be executed to perform inference using the E2E pipeline with quantized OCR and classification models.

``
usage: python src/e2e_pipeline/inc_inference.py  [i][-o][-q][-c][-t][-b]
``

Arguments:
```
  -i, --intel                            Give 1 for enabling intel PyTorch optimizations, default is 0
  -o, --ocr_fp32_model_path              FP32 ocr model path
  -q, --ocr_int8_model_path              quantized OCR model path
  -c, --cls_int8_model_path              quantized classification model path
  -t, --test_data_path                   test data path

```

<b>Example</b>
```sh
python src/e2e_pipeline/inc_inference.py -i 1 \
 -o ./models/ocr_module/CRNN-1010-WOHP_intel.pth \
 -q ./models/ocr_module/best_model.pt \
 -c ./models/classification_module.pb \
 -t data/e2e_pipeline/invoice_dataset \
 -b 1
```
The above script gives the average time taken for inference with the quantized model.
> As dataset used in this reference kit is synthetically generated, test dataset has already been provided at data/e2e_pipeline path to run the 
end to end inference pipeline.\

## **Performance Observations**
This section covers the training time and inference time comparison between Stock PyTorch version and Intel PyTorch Extension for OCR model and time comparison between Tensorflow 2.8.0 and Tensorflow 2.9.1 for Classification module. The results are captured for regular training and hyperparameter tuned training models which includes 
training time and inference time. The results are used to calculate the performance gain achieved by using Intel One API packages over 
stock version of similar packages.

![image](assets/e2e_inf.png)
<br>**Takeaway**<br>Intel® PyTorch Extension and Tensorflow 2.9.1 with Intel Optimizations and INC collectively offers batch inference time speed-up of 
2.38x using trained model compared to stock versions without Intel Optimizations for end to end pipeline.

#### **Conclusion**
To build Trade Promotion Deduction pipeline using CRNN (Convolutional Recurrent Neural Network) transfer learning approach and CNN(Convolutional Neural Network), 
at scale, Data Scientist will need to train models for substantial datasets and run inference more frequently. The ability 
to accelerate training will allow them to train more frequently and achieve better accuracy. Besides training, faster speed 
in inference will allow them to run prediction in real-time scenarios as well as more frequently.  This task of Trade Promotion Deductions pipeline building requires 
a lot of training and retraining, making the job tedious. The ability to get it faster speed will accelerate the ML pipeline. 
This reference kit implementation provides performance-optimized guide for Trade Promotion deduction use cases that can be 
easily scaled across similar use cases.

### Notices & Disclaimers
Performance varies by use, configuration and other factors. Learn more on the [Performance Index site](https://edc.intel.com/content/www/us/en/products/performance/benchmarks/overview/). 
Performance results are based on testing as of dates shown in configurations and may not reflect all publicly available updates.  See backup for configuration details.  No product or component can be absolutely secure. 
Your costs and results may vary. 
Intel technologies may require enabled hardware, software or service activation.<br>
© Intel Corporation. Intel, the Intel logo, and other Intel marks are trademarks of Intel Corporation or its subsidiaries.  Other names and brands may be claimed as the property of others.  

**Date Testing Performed**: December 2022 

**Configuration Details and Workload Setup**: Azure D8v5 (Intel® Xeon® Platinum 8370C CPU @ 2.80GHz), 1 Socket, 4 Cores per Socket, 2 Threads per Core, Turbo: On, Total Memory: 32 GB, OS: Ubuntu 20.04, Kernel: Linux 5.15.0-1019-azure. Framework/Toolkit: PyTorch 1.12.0, Intel® Extension for PyTorch 1.12.100, Intel® Neural Compressor 1.14.0, Tensorflow 2.8.0, Tensorflow 2.9.1, Python 3.9.13. Dataset size: 6k word images for OCR model training, ~16k products and 10 categories for classification model training. Model: EasyOCR model for text detection, custom CRNN model for text recognition and custom CNN model for classification. 

**Testing performed by** Intel Corporation

To the extent that any public or non-Intel datasets or models are referenced by or accessed using tools or code on this site those datasets or models are provided by the third party indicated as the content source. Intel does not create the content and does not warrant its accuracy or quality. By accessing the public content, or using materials trained on or with such content, you agree to the terms associated with that content and that your use complies with the applicable license.
 
Intel expressly disclaims the accuracy, adequacy, or completeness of any such public content, and is not liable for any errors, omissions, or defects in the content, or for any reliance on the content. Intel is not liable for any liability or damages relating to your use of public content.


## Appendix

### **Experiment setup**

| Platform                          | Ubuntu 20.04
| :---                              | :---
| Hardware                          | Azure Standard_D8_V5 (Icelake)
| Software                          | Intel® Distribution for Python 3.9.13, Intel® Extension for PyTorch 1.12.100,Tensorflow 2.9.1.
| What you will learn               | Advantage of using components in Intel® Extension for PyTorch over the stock versions and advantage of using TF 2.9.1 with oneDNN optimizations over TF 2.8.0

### Troubleshooting
We have listed some of the issues that might occur during the environment creation or running the experiment scripts. Also, a resolution is provided against the error. 
If the users encounter these errors, they can refer to this section to resolve the issues by themselves:

|Step	|Error |	Resolution
| :---                       | :---                | :---  
|Intel Environment Setup | error: command 'gcc' failed: No such file or directory  [end of output]  note: This error originates from a subprocess, and is likely not a problem with pip. ERROR: Failed building wheel for pycocotools ERROR: Could not build wheels for pycocotools, which is required to install pyproject.toml-based projects failed  CondaEnvException: Pip failed | sudo apt install gcc
|Running Training Script in tradepromotion-stock Environment| TypeError: Descriptors cannot not be created directly. If this call came from a _pb2.py file, your generated code is out of date and must be regenerated with protoc >= 3.19.0. If you cannot immediately regenerate your protos, some other possible workarounds are: 1. Downgrade the protobuf package to 3.20.x or lower. 2. Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python (but this will use pure-Python parsing and will be much slower). | 1. Downgrade the protobuf package to 3.20.x or lower. 2. Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python (but this will use pure-Python parsing and will be much slower).
|Running Training Script in tradepromotion-intel Enviornment|File "/miniconda/envs/tradepromotion-intel/lib/python3.9/site-packages/nltk/data.py", line 583, in find raise LookupError(resource_not_found) Resource punkt not found. Please use the NLTK Downloader to obtain the resource:| Try running following commands on python prompt >>> import nltk >>> nltk.download('punkt') nltk.download('averaged_perceptron_tagger') nltk.download('wordnet') nltk.download('omw-1.4')
