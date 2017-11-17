---
layout: post
title: 'Dynamic Routing Between Capsules'
categories:
  - ai
crosspost_to_medium: true
---

Convolutional neural networks have dominated the computer vision landscape ever
since [AlexNet won the ImageNet challenge in 2012][alexnet], and for good
reason. Convolutions create a spatial dependency inside the network that
functions as an effective prior for image classification and segmentation.
Weight sharing reduces the number of parameters, and efficient accelerated
implementations are readily available. These days, convolutional neural networks
(or "convnets" for short) are the de facto architecture for almost any computer
vision task.

However, despite the enormous success of convnets in recent years, the question
begs to be asked, _"Can we do better?"_ Are there underlying assumptions built
into the fundamentals of convnets that makes them in some ways deficient?

__Capsule networks__ are novel one architecture that attempts to supersede
traditional convnets. Geoffrey Hinton, who helped develop the [backpropagation
algorithm][backprop] and has pioneered neural networks and deep learning, has
[talked][capsule-talk] about capsule networks for some time, but until very
recently no work on the idea had been publicly published. Just a few weeks ago,
[_Dynamic Routing Between Capsules_][capsule-paper] by Sara Sabour, Nicholas
Frosst and Geoffrey Hinton was made available, explaining what capsule networks
are and the details of their functionality. Here, I'll walk through the paper
and give a high level review as to how capsules function, the routing algorithm
described in the paper, and the results of using a capsule network for image
classification and segmentation.

## Why Typical Convnets are Doomed

Before we dive into how capsules solve the problems of convnets, we first need
to establish what capsules are trying to solve. After all, modern convnets can
be trained for near-perfect accuracy over a million images with a thousand
classes, so how bad can they be? While traditional convnets are great at
classifying images like ImageNet, they fall short of perfect in some key ways.

### Sub-sampling loses precise spatial relationships

Convolutions are great because they create a spatial dependency in our models
(see [my previous post][what-is-dl] for a high level overview of convolutions, or [these
lecture notes][stanford-cnn] for an in-depth explanation of convnets), but they
have a key failure. Commonly, a convolutional layer is followed by a (max)
__pooling__ layer. The pooling layer sub-samples the extracted features by
sliding over patches and pulling out the maximum or average value. This has the
benefit of reducing the dimensionality (making it easier for other parts of our
network to work with), but also __loses precise spatial relationships__.

By precise spatial relationships, I mean the exact ways that the extracted
features relate to one another. For instance, consider the following image of a
kitten:

<figure class="image" >
<img style="display:block;margin:0 auto;" src="{{ site.url }}/images/capsules/kitten.jpeg" alt="A picture of a normal looking kitten">
<figcaption style="display:block;margin:0 auto;text-align:center">Just an
ordinary, though cute, kitten</figcaption>
</figure>

A pretrained [ResNet50][resnet] convnet classifies as a tabby cat, which is
obviously correct. Convnets are excellent at detecting specific features within
an image, such as the cats ears, nose, eyes, paws, etc. and combining them to
form a classification. However, sub-sampling via pooling loses the exact
relationship that those features share with each other: e.g. the eyes should be
level and the mouth should be underneath them. Consider what happens when I edit
some of the spatial relationships, and create a kitten image more in the style
of Pablo Picasso:

<figure class="image" >
<img style="display:block;margin:0 auto;" src="{{ site.url }}/images/capsules/kitten-picasso.jpg" alt="A picture of a normal looking kitten">
<figcaption style="display:block;margin:0 auto;text-align:center">A slightly
less ordinary kitten</figcaption>
</figure>

When this image is fed to our convnet, we __still__ get a tabby classification
with similar confidence. That's completely incorrect! Any person can
immediately tell by looking at the image something isn't right, but the convnet
plugs along as if the two images are almost identical.

### Convnets are _invariant_, not _equivariant_

Another shortcoming of typical convnets is that the explicitly strive to be
invariant to change. By invariant, I mean that the entire classification
procedure (the hidden layer activations and the final prediction) are nearly
identical to small changes in the input (such as shift, tilt, zoom). This is
effective for the classification task, but it ultimately limits our convents.
Consider what happens when I flip the previous image of the kitten upside-down:

<figure class="image">
<img style="display:block;margin:0 auto;" src="{{ site.url }}/images/capsules/kitten-rotated-180.jpg" alt="A picture of a normal looking kitten">
<figcaption style="display:block;margin:0 auto;text-align:center">The view of a
kitten if you were hanging from the ceiling</figcaption>
</figure>

This time, our ResNet __thinks our kitten is a guinea pig__! The problem is that
while convnets are invariant to small changes, they don't react well to large
changes. Even though all the features are still in the image, the lack of
spatial knowledge within the convnet means it can't make head or tail of such a
transformation.

Rather than invariance that's built into traditional convnets by design in
pooling layers, what we should really strive for is __equivariance__: the model
will still produce a similar classification, but the internal activations
transform along with the image transformations. Rather than ignoring
transformations, we should adjust alongside them.

### A note on sub-sampling

Before we proceed, I feel it necessary to point out that Hinton points out these
problems with sub-sampling in convnets, __not the convolution operation
itself__. The sub-sampling in typical convnets (usually max-pooling) is largely
to blame for these deficiencies in convnets. The convolution operation itself is
quite useful, and is even utilized in the capsule networks presented.

## Capsules to the Rescue

These kinds of problems (lack of precise spatial knowledge and invariance to
transformations) are exactly what capsules try to solve. Most simply, __a
capsule is just a group of neurons__. A typical neural network layer has some
number of neurons, each of which is a floating point number. A _capsule layer_,
on the other hand, is a layer that has some number of capsules, each of which is
a grouping of floating point neurons. In this work, a capsule is a single
vector, though [later work][matrix-capsules] utilizes matrices for their
capsules.

The key idea is that by grouping neurons into capsules, we can encode more
information about the entity (i.e. feature or object) that we're detecting. This
extra information could be size, shape, position, or a host of other things. The
framework of capsules leaves this open, and its up to the implementation to
define and enforce these encoding principles.

There's a few things we need to be careful about before we can get started using
capsules. First, since capsules contain extra information, we need to be a
little more nuanced about how we connect capsule layers and utilize them in our
network. Typical convnets only care about the existence of a feature/object, so
their layers can be fully connected without problem, but we don't get that
luxury when we start encoding extra properties with capsules. We also have to be
smart about how we connect capsules so that we can appropriately manage the
dimensionality without having to resort to pooling.

## Connecting Capsule Layers: Dynamic Routing by Agreement

The central algorithm presented in the [capsules paper][capsule-paper] that came
out recently is one that describes how capsule layers can be connected to one
another. The authors chose an algorithm that encourages "routing by agreement":
capsules in an earlier layer that cause a greater output in the subsequent layer
should be encouraged to send a greater portion of their output to that capsule
in the subsequent layer.

The routing procedure happens for every forward pass through the network, both
during testing and training. The image below visually describes the effect of
the routing procedure.

<figure class="image">
<img style="display:block;margin:0 auto;" src="{{ site.url }}/images/capsules/routing-visualized.jpeg"
alt="A visual explaination of the effect of the routing procedure">
<figcaption style="display:block;margin:0 auto;text-align:center">A visual
description of the before and after of the routing procedure. Arrow widths
correspond to the strength of the output. The lines coming out of the capsules
in the second layer correspond to the portions of the output that come from the
capsule in the previous layer.</figcaption>
</figure>

Before the routing procedure, every capsule in the earlier layer spreads its
output _evenly_ to every capsule in the subsequent layer (initial couplings can
be learned like weights, but this isn't done in the paper). During each
iteration of the dynamic routing algorithm, strong outputs from capsules in the
subsequent layer are used to encourage capsules in the previous layer to send a
greater portion of their output. Note how `caps_21` has a large portion of its
output influenced by `caps_11` (denoted by the thick arrow coming out on top).
After the routing procedure, `caps_11` sends much more of its output toward
`caps_21` than any of the other capsules in the second layer.

The mathematical details of this procedure are excellently explained in the
[original paper][capsule-paper], but for brevity I will omit a complete
explanation.

## CapsNet: A Shallow Network with Deep Results

Now let's look at the actual capsule network utilized int the
[paper][capsule-paper], known as __CapsNet__.

<figure class="image">
<img style="display:block;margin:0 auto;" src="{{ site.url }}/images/capsules/capsnet-arch.jpeg" alt="The CapsNet Architecture">
<figcaption style="display:block;margin:0 auto;text-align:center">The CapsNet
architecture</figcaption>
</figure>

Our input images are the MNIST data set: 28x28 grayscale pictures of handwritten
digits (later we'll see how capsules perform on other, more complex data sets).
These get fed into a convolutional layer (256 9x9 filters, stride of 1 and no
padding). That layer passes through another set of convolutions to become the
__PrimaryCaps__ layer, the first capsule layer. Each capsule inside PrimaryCaps
is a size 8 vector. There are 32x6x6 = 1152 of the capsule in the PrimaryCaps
layer. In the paper, they construct their capsule architecture such that the
_length_ of a capsule (i.e. the value after putting the vector through the
Euclidean norm) represents the likelihood an entity exists, and the
_orientation_ (i.e. how the values are distributed within a capsule) represents
all other spatial properties of the entity.

The PrimaryCaps layer is connected to the DigitCaps layer. _This is the only
place in the CapsNet architecture where routing takes place_. DigitCaps is a
layer of 10 capsules (one corresponding to each potential digit), each a size
16 vector. Since the length of the vector corresponds to its existence, all we
need to do is norm the capsules in DigitCaps to get our logits, which can be fed
into a softmax layer to get prediction probabilities.

Since each digit could appear independently, and in some tests multiple digits,
the authors used a custom loss function that penalized each digit independently
during training. Additionally, they wanted to ensure that each DigitCaps was
learning a good representation within the capsule. To do this they appended a
fully connected network to function as a decoder and attempt to reconstruct the
original input from the DigitCaps layer.

<figure class="image">
<img style="display:block;margin:0 auto;" src="{{ site.url }}/images/capsules/capsnet-reconstruction.jpeg" alt="The CapsNet Architecture">
<figcaption style="display:block;margin:0 auto;text-align:center">Reconstructing
the original input from the DigitCaps layer</figcaption>
</figure>

Only the capsule value from the correct label was used to reconstruct (the
others were masked out). The reconstruction error was used to regularize CapsNet
during training, forcing the DigitCaps layer to learn more about how the digit
appeared in the image.

## Results

With only three layers, the CapsNet architecture performed remarkably well. The
authors report a __0.25%__ test error rate on MNIST, which is close to state of
the art and not possible with a similarly shallow convnet.

They also performed so experiments on a MultiMNIST data set: two images from
MNIST overlapping each other by up to 80%. Since CapsNet understands more about
size, shape, and position, it should be able to use that knowledge to untangle
the overlapping words.

<figure class="image">
<img style="display:block;margin:0 auto;" src="{{ site.url }}/images/capsules/multi-mnist.jpeg" alt="The CapsNet Architecture">
<figcaption style="display:block;margin:0 auto;text-align:center">Reconstructing
from severely overlapping digits</figcaption>
</figure>

The image above represents CapsNet reconstructing its predictions twice, one for
each digit in the image. The red corresponds to one digit, and green for another
(yellow is where they overlap). CapsNet is extremely good at this kind of
segmentation task, and the authors suggest is in part because the routing
mechanism serves as a form of attention.

CapsNet is also performant on several other data sets. On CIFAR-10, it has a
10.6% error rate (with an ensemble and some minor architecture modifications),
which is roughly the same as when convnets were first used on the data set.
CapsNet attain 2.7% error on the smallNORB data set, and 4.3% error on a subset
of Street View Housing Numbers (SVHN).

## Conclusion

Convnets are extremely performant architectures for computer vision tasks. Their
resurgence has marked the recent AI renaissance currently unfolding with the
advent of deep learning. However, they suffer from some serious flaws that make
them unlikely to take us all the way to general intelligence.

Capsules are a novel enhancement that go beyond typical convnets by encoding
extra information about detected objects and retain precise spatial
relationships by avoiding sub-sampling. The simple capsule architecture
presented in this paper, CapsNet, is able to get incredible results considering
its small size. Additionally, CapsNet understands more about the images its
classifying, like their position and size.

Although CapsNet doesn't necessarily outperform convnets, they are able to match
their accuracy out-of-the-box. This is really promising for the future role of
capsules in computer vision. There's still a huge amount to research to be done
into improving capsules and scaling them to larger data sets. As a computer
vision researcher, this is an extremely exciting time to be working in the
field!

[alexnet]: http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks
[backprop]: https://www.wikipedia.org/en/Backpropagation
[capsule-talk]: https://www.youtube.com/watch?v=rTawFwUvnLE&t=602s
[capsule-paper]: https://arxiv.org/abs/1710.09829
[what-is-dl]: {{site.url}}/2017/09/09/what-is-deep-learning/
[stanford-cnn]: https://cs231n.github.io/convolutional-networks/
[resnet]: https://arxiv.org/abs/1512.03385
[matrix-capsules]: https://openreview.net/pdf?id=HJWLfGWRb
