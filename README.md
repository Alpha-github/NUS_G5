# NUS Internship Project: FreshLite

&nbsp;
[![Tensorflow](https://miro.medium.com/max/500/1*37N7BHNaEsXPaerNQ8wBdA.png)](https://www.tensorflow.org/)[![Keras](https://static.javatpoint.com/tutorial/keras/images/keras.png)](https://keras.io/)

[![Numpy](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTeXkEA3c2hxTcrZWwVnniXAFiqai51196osD9FWL0_D_Ca7fOT)](https://www.numpy.org/)[![OpenCV](https://developer.nvidia.com/sites/default/files/akamai/cuda/images/product_logos/OpenCV_Logo_340.jpg)](https://opencv.org/)
[![Flask](https://cdn-images-1.medium.com/max/1150/1*0G5zu7CnXdMT9pGbYUTQLQ.png)](http://flask.pocoo.org/)[![Python](https://pbs.twimg.com/media/FDWpCsXVQAQFuUw.jpg)](https://www.python.org/)
[![Azure](https://techgenix.com/tgwordpress/wp-content/uploads/2016/12/How-to-Deploy-a-Windows-Server-2016-as-an-Azure-VM.png)](https://azure.microsoft.com/)[![Docker](https://blog.knoldus.com/wp-content/uploads/2017/12/docker_facebook_share.png)](https://www.docker.com/)
&nbsp;

## Introduction:

---

##### FreshLite is a Deep Learning Solution for Fruit and Vegetable health detection

Built on Tensorflow - Keras using Python, this project adopts Ensemble Learning of Two Custom Models and One Pre-trained Model.
The Web App Interface is Created using Flask, Containerized using Docker and Hosted on Azure Web Services.
&nbsp;

##### Click on the image below to Try It Out !!

[![freshlite](static/img/app_interface.png)](https://gaip-group-5.azurewebsites.net/)
&nbsp;

## Table Of Contents

- #### [Introduction](#introduction)
- #### [Technologies](#technologies-1)
- #### [Datasets](#datasets-1)
- #### [Data Preprocessing](#data-preprocessing-1)
- #### [Neural Network Architecture](#neural-network-architecture-1)
- #### [Setup](#setup-1)
- #### [Docker Deployment](#docker-deployment-1)
- #### [Additional Resources](#additional-resources-1)
- #### [Downloads](#downloads-1)

## Technologies

FreshLite utilizes several libraries and platforms (mostly Open Source) to operate seemlessly:

- [Python] - Python is a high-level, interpreted, general-purpose programming language. It was designed for readability and is a cross-platform, interpreted, object-oriented programming language that is perfectly suited for Rapid Application Development, scripting, and connecting existing components together.
- [Tensorflow][Tensorflow] - TensorFlow is an end-to-end platform that makes it easy for you to build and deploy ML models. TensorFlow offers multiple levels of abstraction so you can choose the right one for your needs.
- [Keras](https://keras.io/) - Keras is an open-source software library that provides a Python interface for artificial neural networks. Keras acts as an interface for the TensorFlow library.
- [Flask](http://flask.pocoo.org/) - Flask is a small and lightweight Python web framework that provides useful tools and features that make creating web applications in Python easier. It gives developers flexibility and is a more accessible framework for new developers since you can build a web application quickly using only a single Python file.
- [OpenCV][OpenCV] - OpenCV is a great tool for image processing and performing computer vision tasks. It is an open-source library that can be used to perform tasks like face detection, objection tracking, landmark detection, and much more.
- [Numpy](https://www.numpy.org/) - NumPy is a library for the Python programming language which provides additional support for processing large, multi-dimensional arrays and matrices.
- [Azure][Azure] - Azure is Microsoft's public cloud computing platform, with solutions including Infrastructure as a Service (IaaS), Platform as a Service (PaaS), and Software as a Service (SaaS) that can be used for services such as analytics, virtual computing, storage, networking, and much more.
- [Docker](https://www.docker.com/) - Docker is a set of platform as a service products that use OS-level virtualization to deliver software in packages called containers. Docker enables you to separate your applications from your infrastructure so you can deliver software quickly.

## Datasets

| Source                | Links                                             |
| --------------------- | ------------------------------------------------- |
| Bing Image Search API | [Web Scrapper Program]                            |
| Kaggle Datasets       | [Fresh and Stale Images of Fruits and Vegetables] |
|                       | [Fruits fresh and rotten for classification]      |
| Custom Images         | Images taken using Camera                         |

## Data Preprocessing

> ##### Step 1: Split the dataset into Train and Test datasets
>
> Program: [Preprocess_1]

> ##### Step 2: Merge the multiple labels into Fresh and Rotten
>
> Program: [Preprocess_2]

Note: Remember to enter the path of dataset folders in place of double quotations (" ").

## Neural Network Architecture

&nbsp;

#### 1. [Custom Models] :

![Custom Models](/static/img/Custom_Model.png)

##### Layers:

- **Convolution Layer**: Convolution is an orderly procedure where two sources of information are intertwined; it’s an operation that changes a function into something else. Convolutions are used to perform operations such as enhancing edges and embossing, to _extract features from the image_.
- **Conv2D**: A filter or a kernel in a conv2D layer “slides” over the 2D input data,also called striding, performing an elementwise multiplication. As a result, it will be summing up the results into a single output pixel. The kernel will perform the same operation for every location it slides over, transforming a 2D matrix of features into a different 2D matrix of features.
- **MaxPooling2D**: This layer downsamples the input along its spatial dimensions (height and width) by taking the maximum value over an input window for each channel of the input. The window is shifted by strides along each dimension. It essentially compresses the image. This method is helpful to extract features with high importance or which are high-lighted in the image.
- **Flatten**: Flattening is converting the data into a 1-dimensional array for inputting it to the next layer. We flatten the output of the convolutional layers to create a single long feature vector and it is connected to the final classification model, which is called a fully-connected layer (Dense Layers).
- **Dense Layer/ Fully Connected Layer**: Dense Layer is simple layer of neurons in which each neuron receives input from all the neurons of previous layer, thus called as dense. Dense Layer is used to classify image based on output from convolutional layers.
- **[Dropout]**: Large neural nets trained on relatively small datasets can overfit the training data.
  Ensembles of neural networks with different model configurations are known to reduce overfitting, but require the additional computational expense of training and maintaining multiple models. A single model can be used to simulate having a large number of different network architectures by randomly dropping out nodes during training. This is called dropout and offers a very computationally cheap and remarkably effective regularization method to reduce overfitting and improve generalization error in deep neural networks of all kinds. During training, some number of layer outputs are randomly ignored or “dropped out”.

**_Check It Out_:**

- [Introduction To Convolution Neural Network]
- [Droput For Regularizing Deep Neural Networks]

&nbsp;

#### 2. Pretrained Model : MobileNet V2

![MobileNet-V2](/static/img/mobilenetv2.png)

[MobileNet-V2] is a convolutional neural network that is 53 layers deep. It is a light-weight model, that uses Depthwise Separable Convolution Layers to reduce the complexity cost, which is suitable to be used on mobile devices or devices with low computation power.
You can load a pretrained version of the network trained on more than a million images from the ImageNet database. However for our use, we have trained the model on our custom dataset with 2 output labels instead of 1000.

## Setup

##### This project was built on Windows 11

FreshLite requires [Python] v3.8+ to run.

##### Click below to visit Python 3.9.13 Download Page

[![Python](https://www.twine.net/blog/wordpress/wp-content/uploads/2022/07/pythonlogo.png)](https://www.python.org/downloads/release/python-3913/)

**NOTE:** Ensure you have **pip** installed, else follow the given [Pip Installation]
Install the following dependencies on your local machine and run [app.py].

###### Install [Flask](http://flask.pocoo.org/) using pip

```sh
$ pip install Flask
```

###### Install [OpenCV][OpenCV] using pip

```sh
$ pip install opencv-python
```

###### Install [Numpy] using pip

```sh
$ pip install numpy
```

###### Install [Tensorflow][Tensorflow] using pip

> Follow the instructions in the link for [Installation Of Tensorflow and GPU Setup](https://www.tensorflow.org/install/pip)

## Docker Deployment

FreshLite can be easily deployed using Docker.

> File : [Dockerfile]

By default, the Docker will expose port 8080, so change this within the
Dockerfile if necessary. When ready, simply use the Dockerfile to
build the image.

Assuming that all files are stored in folder: **NUS_G5_Internship**

```sh
cd NUS_G5_Internship
docker build NUS_G5_Internship --tag {name of image}
```

This will create the docker image.
Be sure to swap out `{name of image}` with the name you want to prvide.

Once done, run the Docker image and map the port to whatever you wish on
your host. In this example, we simply map port 8000 of the host to
port 8080 of the Docker (or whatever port was exposed in the Dockerfile):

```sh
docker run -p 8000:8080 {name of image}
```

Verify the deployment by navigating to your server address in
your preferred browser.

```sh
127.0.0.1:8000
```

## Additional Resources

> Refer the following resource for Deployment of Docker Image With Azure Web Services:
> https://docs.microsoft.com/en-us/learn/modules/deploy-run-container-app-service/

## Downloads

###### To Download this Repo using GIT BASH:

```sh
$ git clone https://github.com/Alpha-github/NUS_G5_Internship.git
```

###### To Download this Repo using GIT CLI:

```sh
gh repo clone Alpha-github/NUS_G5_Internship
```

## License

Apache License 2.0

[Keras]:[https://keras.io/](https://keras.io/)
[Python]: https://www.python.org/
[Flask]:[http://flask.pocoo.org/](http://flask.pocoo.org/)
[Docker]:[https://www.docker.com/](https://www.docker.com/)
[Numpy]:[https://www.numpy.org/](https://www.numpy.org/)
[MobileNet-V2]:[https://towardsdatascience.com/review-mobilenetv2-light-weight-model-image-classification-8febb490e61c](https://towardsdatascience.com/review-mobilenetv2-light-weight-model-image-classification-8febb490e61c)
[Web Scrapper Program]:<crawler.py>
[Preprocess_1]:<preprocess.py>
[Preprocess_2]:<preprocess2.py>
[Custom Models]:<mod3.ipynb>
[app.py]:<app.py>
[Dockerfile]:`<Dockerfile>`
[Introduction To Convolution Neural Network]:[https://towardsdatascience.com/introduction-to-convolutional-neural-network-cnn-de73f69c5b83#:~:text=Dense%20Layer%20is%20simple%20layer,multiple%20number%20of%20such%20neurons.](https://towardsdatascience.com/introduction-to-convolutional-neural-network-cnn-de73f69c5b83#:~:text=Dense%20Layer%20is%20simple%20layer,multiple%20number%20of%20such%20neurons.)
[Dropout]:[https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/](https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/)
[Droput For Regularizing Deep Neural Networks]:[https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/](https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/)
[Pip Installation]:[https://www.geeksforgeeks.org/how-to-install-pip-on-windows/#:~:text=Step%201%3A%20Download%20the%20get,where%20the%20above%20file%20exists.&amp;text=Step%204%3A%20Now%20wait%20through%20the%20installation%20process.](https://www.geeksforgeeks.org/how-to-install-pip-on-windows/#:~:text=Step%201%3A%20Download%20the%20get,where%20the%20above%20file%20exists.&text=Step%204%3A%20Now%20wait%20through%20the%20installation%20process.)
[Fresh and Stale Images of Fruits and Vegetables]:[https://www.kaggle.com/datasets/raghavrpotdar/fresh-and-stale-images-of-fruits-and-vegetables](https://www.kaggle.com/datasets/raghavrpotdar/fresh-and-stale-images-of-fruits-and-vegetables)
[Fruits fresh and rotten for classification]:[https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification](https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification)
[Click to go to Python 3.9.13 Download Page]:[https://www.python.org/downloads/release/python-3913/](https://www.python.org/downloads/release/python-3913/)

[Tensorflow]: https://www.tensorflow.org/
[Azure]: https://azure.microsoft.com/

[OpenCV]: https://opencv.org/
