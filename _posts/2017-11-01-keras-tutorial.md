---
layout: post
title: 'Simplifying Deep Learning Programming with Keras'
categories:
 - ai
 - tutorial
---

[Last post,]({{site.url}}/2017/09/12/intro-to-tensorflow) I gave an introduction
into programming a deep neural network with TensorFlow. The model worked quite
well (98% accuracy on the test set) with only 150 lines of code, but it was
arguably a bit complex.

The problem was we had to really dig into the nitty-gritty details of how we
wanted our model to work. But a lot of times, we do not need to deal with that
level of detail and the complexity that comes with it. This kind of problem
occurs often in software engineering, and it is generally solved with a
convenient library.

[Keras](https://keras.io) is such a library: it does a great job of taking the
complexity out of building a neural network, so you can focus on the interesting
parts of training and utilizing the model. In this post I'll walk through some
of the basics of Keras and we will rebuild our MNIST handwritten-digit
classifier in a much simpler program.

## Keras: The Model Abstraction

In Keras, the fundamental abstraction is the `Model` object. We can design,
train, and evaluate the `Model` without necessarily knowing the exact details.
In this example, TensorFlow will be the backend that Keras will utilize behind
the scenes, but Keras can actually function agnostic of its specific backend and
run with [TensorFlow][tf], [Theano][th], or [CNTK][cntk].

There are a few different model types, but the one we will utilize is the
`Sequential` model. The `Sequential` model view the network architecture as a
sequence of layers strung together, one after another. This is exactly the
architecture we used in our previous convolutional neural network. With Keras,
we can stack our network layers like individual building blocks to create our
overall model.

{% highlight python %}
from keras.models import Sequential
model = Sequential()
{% endhighlight %}

## Adding Layers

To add new layers to a Keras model, we simply call the `add()` function and pass
in the layer we want to use. To recreate our previous convnet, we'll need main
kinds of layers: `Dense` for our fully connected layers, and `Conv2D` for our
two dimensional convolutional layers. We'll also need `MaxPool2D` and `Dropout`
layers to utilize max pooling and dropout. Finally, a `Flatten` layer will be
used to convert between our convolutional and fully connected layers.

{% highlight python %}
from keras.layers import Dense, Conv2D, Dropout, Flatten
model.add(Conv2D(32, (5, 5), padding='same'), input_shape=(28, 28, 1))
model.add(MaxPooling2D(padding='same'))
model.add(Conv2D(64, (5, 5), padding='same'))
model.add(MaxPooling2D(padding='same'))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
{% endhighlight %}

There's a few things we should not here. The first layer we `add()` needs to
take an additional argument: `input_shape`. This tells Keras the size of the
inputs that we will feed into our model (in our case, a 28x28 pixel image for
MNIST). For the `Conv2D` layers, the first argument represents the number of
filters, followed by the dimensions of our convolution. The `Dense` layers take
an argument that represents the number of neurons in that layer. We can also
specify the activation function we want to use by a keyword argument, as we did
here. Alternatively, we could have added an `Activation` layer.

## Compiling and Training the Model

Now that we have defined what our convnet will look like by stacking all of our
layers into our `model`, we can get ready to start training our model on the
data set. However, first we need to `compile` the model. Since Keras serves as a
high-level wrapper of other machine learning libraries, it needs to convert our
Keras-defined model into a model of our backend. Additionally, we will need to
specify some other attributes of our training procedure.

{% highlight python %}
from keras.optimizers import Adadelta
model.compile(optimizer=Adadelta(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
{% endhighlight %}

Here we specify that training will use the [Adadelta optimizer][adadelta], our
loss function is defined by the cross entropy of the output (since this is a
classification task), and we want to optimize over the accuracy of the model.

Next, we can get our training data ready. Luckily, Keras even has some common
data sets built in.

{% highlight python %}
from keras.dataset import mnist
from keras.utils import to_categorical
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Convert to values between 0. and 1.
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
{% endhighlight %}

For the inputs, we need to convert the arrays into the right shape, and scale
the values between $$[0, 1]$$. The outputs get converted to binary one-hot
vectors by Kera's `to_categorical` utility function.

Now we can finally train our model. This is done by the `Model`'s `fit` method:

{% highlight python %}
model.fit(x_train, y_train, epochs=10, batch_size=50)
{% endhighlight %}

The `epochs` argument will determine how many passes through the data training
will make. The `batch_size` determines how many samples to train with for each
weight update. Keras will output its progress as it works, updating you on which
epoch is running, approximately how long it will take, and the current loss in
the model.

## Evaluating the Results

Once our model is trained, we can see how accurate it is at predicting on novel
data. To see how our model stacks up against the test set, use the `evaluate`
method:

{% highlight python %}
loss, accuracy = model.evaluate(x_test, y_test)
print('Accuracy of {:.2f}%'.format(accuracy * 100))
{% endhighlight %}

So in just a few lines of Python, we were able to create a high performing MNIST
classifier! Using Keras is really straightforward, and allows us to avoid the
nitty-gritty details of programming complex deep neural networks. Instead, we
can work on other interesting aspects of our models and keep the implementation
from hindering our ideas. And when Keras is too high level, we can even use it
as a [simplified interface to TensorFlow][keras-to-tensorflow]. As a deep
learning researcher, Keras takes a lot of the hassle out of programming deep
neural networks.


[tf]:tensorflow.org
[th]:http://www.deeplearning.net/software/theano/
[cntk]:https://www.microsoft.com/en-us/cognitive-toolkit/
[adadelta]:https://arxiv.org/abs/1212.5701
[keras-to-tensorflow]:https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html

