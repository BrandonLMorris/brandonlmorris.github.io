---
layout: home
title: "Brandon Morris"
permalink: /
---

# Welcome

<figure class="image" style="float:right; width:30%; margin-left:10pt">
<img style="border-radius:20%;" src="/images/avatar.jpeg">
<figcaption style="text-align:center;"><a href="mailto:mail@brandonmorris.dev" style="color:black;">mail [AT] brandonmorris.dev</a></figcaption>
</figure>

I am a software engineer currently located in the Silicon Valley. My areas of
experience and interest lie in **automated mobile testing** (iOS and Android),
**machine learning**, and **software infrastructure**.

## Previous Activity

In 2018 I received my undergraduate degrees in Software Engineering and Applied
Mathematics from Auburn University, where I also conducted research under [Dr.
Anh Nguyen][nguyen] studying deep learning robustness against [adversarial
attacks][vectordefense]. Before that, I researched [High Performance Computing
methods][mpignite] under Dr. Anthony Skjellum and [GPU-accelerated chemistry
simulations][mcgpu] under Dr. Jeffrey Overbey and Dr. Orlando Acevedo.

Outside of class, I've had a number of professional experiences. For a full year
I worked as a software engineering co-op at ADTRAN, Inc. During my time, I
rotated through teams working in quality assurance, [internal IaaS testing
infrastructure][tbaas], and [cloud networking cluster management
technology][firefly].

After graduation, I interned at Sandia National Labs where I engineered a machine
learning model to predict hard drive failure from raw S.M.A.R.T. disk attributes.

---

{% for post in site.posts limit:1 %}
<h2>
  <a style="color:black; text-decoration:none" href="{{post.url}}">Latest Post: {{post.title}}</a>
</h2>
{{ post.content }}
{% endfor %}

[nguyen]: http://anhnguyen.me/
[vectordefense]: https://arxiv.org/abs/1804.08529
[mcgpu]: https://github.com/orlandoacevedo/MCGPU
[mpignite]: https://arxiv.org/abs/1707.04788
[tbaas]: https://www.adtran.com/index.php/blog/technology-blog/269-creating-integration-test-environments-at-adtran
[firefly]: https://www.adtran.com/index.php/blog/technology-blog/269-creating-integration-test-environments-at-adtran
