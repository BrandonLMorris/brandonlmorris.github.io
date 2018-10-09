---
layout: home
title: "My home"
permalink: /
---

# Welcome

<figure class="image" style="float:right; width:30%;border-radius:20%; margin-left:10pt">
<img src="/images/avatar.jpeg">
<figcaption style="text-align:center;"><a href="mailto:mail@brandonlmorris.com" style="color:black;">mail [AT] brandonlmorris.com</a></figcaption>
</figure>

I am an Ph.D. student of Computer Science with a focus on Artificial
Intelligence at Arizona State University. I currently study deep learning and
with a particular focus on multimodal models combining computer vision and
natural language processing. I currently study under [Dr. Yezhou Yang][yang] in
the [Active Perception Group][apg].

## Research Interests

Deep learning is an incredibly powerful and popular machine learning technique,
but it is not without its own set of [limitations][marcus]. I'm interested in
understanding these shortcomings and devising novel approaches to overcome them.
This includes working on problems like [adversarial examples][ae] and expanding
new architectures like [capsule networks][capsules]. Ultimately, I want to make
deep learning more robust, more practical, and more useful within our society.
Feel free to [email](mailto:mail@brandonlmorris.com) me if you'd like to
discuss!

## Previous Activity

In 2018 I received my undergraduate degrees in Software Engineering and Applied
Mathematics from Auburn University, where I also conducted research under [Dr.
Anh Nguyen][nguyen] studying deep learning robustness against [adversarial
attacks][vectordefense]. Before that, I researched [High Performance Computing
methods][mpignite] under Dr. Anthony Skjellum and [GPU-accelerated chemistry
simulations][mcgpu]
under Dr. Jeffrey Overbey and Dr. Orlando Acevedo.

Outside of class, I've had a number of professional experiences. For a full year
I worked as a software engineering co-op at ADTRAN, Inc. During my time, I
rotated through teams working in quality assurance, [internal IaaS testing
infrastructure], and [cloud networking cluster management technology][firefly].

More recently, I interned at Sandia National Labs where I engineered a machine
learning model to predict hard drive failure from raw S.M.A.R.T. disc attributes
(to be open-sourced soon). My full resume can be found [here][resume].


---

{% for post in site.posts limit:1 %}
<h1>
  <a style="color:black; text-decoration:none" href="{{post.url}}">Latest Post: {{post.title}}</a>
</h1>
{{ post.content }}
{% endfor %}

[ae]: https://blog.openai.com/adversarial-example-research/
[nguyen]: http://anhnguyen.me/
[marcus]: https://arxiv.org/abs/1801.00631
[capsules]: {{site.url}}/2017/11/16/dynamic-routing-between-capsules/
[yang]: https://yezhouyang.engineering.asu.edu/
[apg]: https://yezhouyang.engineering.asu.edu/research-group/
[vectordefense]: https://arxiv.org/abs/1804.08529
[mcgpu]: https://github.com/orlandoacevedo/MCGPU
[mpignite]: https://arxiv.org/abs/1707.04788
[tbaas]: https://www.adtran.com/index.php/blog/technology-blog/269-creating-integration-test-environments-at-adtran
[firefly]: https://www.adtran.com/index.php/blog/technology-blog/269-creating-integration-test-environments-at-adtran
[resume]: https://goo.gl/oiTq72

