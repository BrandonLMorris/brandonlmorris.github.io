---
layout: post
title: 'Docker for Beginners'
categories:
 - docker
 - tutorial
excerpt: 'Docker and containerization are all the rage nowadays, but what are
they, and what makes them so technologically appealing?'
---

Docker and containerization are all the rage nowadays, but what are they, and
what makes them so technologically appealing?

We should first consider the problems that they are trying to solve. Let's say
that you have an application you want to deploy: some code that will have to
run on another machine, either your own server or in the mystical cloud. How
can you be sure that your app will work after you deploy it? Of course, you
have tests, and of course, they are passing before you even consider releasing,
so you know that it runs fine on your computer, but that's not where we're
deploying. On a different computer or in a different environment, there's no
real way for us to ensure that *something* won't break *somewhere* because
the configurations don't perfectly match up.

Now let's consider a slightly different problem. Perhaps your application has
become incredibly popular. Congratulations. But the immense attention its
now receiving is causing a serious degredation in the performance. How can you
scale to meet your users needs? You could buy a nicer server or upgrade your
cloud instance, but that's pricey and can only get us so far. What we need is
an effective way to scale outward, or _horizontally_.


## The Old Solution: Virtual Machines

Virtual machines can be a viable solution to these problems. They serve as
full-fledged computers implemented solely in software. Its like having a
computer inside your computer. They're entirely self-contained, which solves
the consistency issue: instead of shipping code, you can ship a VM that has
your code on it. So long as you deploy with the same machine you test with,
you can rest assured that the two environments will remain consistent.

Virtual machines can also aid in terms of scalability, though in a limited
fashion. If you design your application appropriately, you can scale by adding
new instances of your VM to a pool in production. When a user hits your
service, they will utilize only part of your total deployment, leading to a
distributed workload. Of course, developing an application for this kind of
architecture is no small feat, since distribution introduces problems of
consistency and fault-tolerance, to name a few.

So why even bother with containization when we already have virtual machines.
A large drawback to VMs is that they are bulky. A typical virtual machine will
be gigabytes in size, since they have to exhibit all the characteristics of a
legitimate computer. All of that virtualization creates additional overhead,
even with "bare-metal" VMs that interact more closely with the underlying
hardware.

In addition, VMs aren't very flexible. Each one has to be configured prior to
starting with the exact amount of physical hardware (CPUs, memory, etc.) that
it, and it alone, has access to. Virtual machines can't share resources, and
the ammount alloted to them can't be modified at runtime. So although a
physical machine can run multiple VMs, its limited in terms of its total
resources.


## Containers: The Lean, Mean, Virtual Machine

Containers leverage a lot of the benefits of virtual machines while avoiding
their detriments, and they do it in a very clever way. Recall that virtual
machines created a significant amount of overhead by recreating every aspect
of the physical computer in software. Containers skip this and leverage the
underlying kernel of the host, effectively _sharing_ the low level code and
resources with the host and other containers. In addition, they also share
common binaries and packages. In reality, containers aren't so much a
virtualized computer as they are a shim: the real work is keeping track of the
ways that a particular container differs from the actual machine its running
on. By only keeping up with the differences, containers are dramatically
smaller and faster.

Here's what I mean concretely: the ISO for Ubuntu 14.04 on I have downloaded
is 649 megabytes. If I download the same version as a Docker container image,
it's only 188 megabytes.

Similarly, starting an Ubuntu VM takes around 30 seconds on my machine
(Macbook Pro with SSD), even if it just headless. A container, however, will
take less than a second.

But with all this sharing going on, doesn't that negate one of our primary
reasons for choosing VMs: that we would have completely isolated environments?
Actually, no. Containers maintain all their "differences", what makes them
unique to the host and other containers, constantly. The effect is that they
are in fact completely isolated on a logical level. They may share a file or
a library under the hood, but its completely transparent to you, the user.

The fact that containers are still isolated while also being lightweight means
that we can use them similarly to the way we can use VMs with much less
overhead. We can be confident that our apps will operate the way we expect
them too because we can literally __ship the environment with the app__ all
bundled up in neat containers.

## Containers are not VMs

It can be tempting, especially at the outset, to view containers simply as
virtual machines, particularly since the analogy is so elegant. However the
two differ in some substantial ways; not only technologically, but
functionally.

First of all, __containers should be ephemeral__. They should be logically
small and capable of quickly being started and stopped. They should __not__
serve as a location for data that needs to be maintained over time (for that,
you need volumes).

Additionally, __containers should only run one main process__. This is
important. Since containers are lightweight, we should use them as such. Since
containers are ephemeral, we shouldn't rely on them sticking around. By
breaking up our app into multiple containers, we can acheive greater modularity
and scalability, though not without some redesign.

## Working with Docker

Enough theory, let's actually get our hands dirty.

To utilize containers yourself, you'll need a containerization engine. The
most popular one out there is Docker. Instruction for installing on your
platform can be found [here][docker-install].

If everything went smoothly, you should be able to run `docker info` and get
some reasonable output.

Once you have Docker running, the real fun begins. Let's print a message
from the Docker whale using cowsay:

```bash
$ docker run docker/whalesay cowsay "Hello, docker"
```

The command should take a few seconds to run fully. If this is your first time
running this command, Docker has to pull (download) the image first. An
image in Docker is analogous to an snapshot of a VM. It serves as a template
from which we build all of our containers off of. However, each container from
this image won't cause the image to be reconstructed, since the container will
simply keep track of how it differs from that base image. Concretely, this
means that no matter how many containers of an image we spin up, we only need
to have __one__ copy of the image on disk.

Images are almost always layered, which is why you probably saw multiple lines
downloading after issuing the command. Even these layers can be reused by
Docker. So if in the future you use a different, but similar, image, the
download time (and disk space) will be decreased.

Docker hosts tons of images for lots of different applications, which can be
found at [Docker hub][docker-hub]. Docker will pull from Dockerhub
automatically if it can't find the image locally. You can download an image
with `docker pull <<image name>>` and search images with `docker search
<<image name>>`

Once the image downloads, the container will start. In our command, we
specified that the `docker/whalesay` image should execute the command
`cowsay`. You should see the following output:

![whalesay]({{site.url}}/images/whalesay.jpeg)


Once the `cowsay` command ends, the container will promptly stop. However, it
will stick around should you want to run it again. You can view your running
containers with `docker ps` and view _all_ containers with `docker ps -a`.

Docker will automatically generate a silly name for your container, because
whimsy is an important aspect of software engineering. To delete the container,
you can run `docker rm <<container name/id>>`

## Playing in a Sandbox

Let's toy around with this some more. If you ran the `docker/whalesay`
container from the previous section, you should have the image saved on your
computer. You can verify this with `docker images`.

Earlier, we ran the `cowsay` command on this image, but there's nothing
stopping us from running other commands. Try

`$ docker run docker/whalesay date`

It should print out the current time. That's nice, but what if we want to
keep the container up and run multiple commands? This is called attaching to
a container. To do this, we will need two things. We need to tell Docker
to give our container a tty interface, so we can issue commands, as well as to
read input from `stdin` (our keyboard, in this case). Additionally, we will
need to execute a __long-running process__ that won't immediately end and kill
our container. We can do all this with the following command:

`$ docker run -it docker/whalesay bash`

Here, the `-it` flags give us an "interactive" and "tty" run on our container.
We also execute `bash`, which will serve as our long-running process. After
executing this command, you should see a different terminal prompt, coming
from inside your container.

Feel free to mess around and issue any commands you normally could on an
Ubuntu machine. But be warned: any changes you make die along with the
container. __Do not store data inside a container__. Instead you should use
[volumes][docker-volumes] to export the data to a more persistent location.
When you're done, you can exit the container by exiting the bash shell:
`<Ctrl-d>`.


## Go forth, and Dockerize

There's a load more to say about Docker and containers in general. Some
interesting points that I didn't get to in this post include (but are not
limited to):

- Volumes and persistent storage
- Building your own images with Dockerfiles
- Creating a full-stack app with containers
- Networking to and between containers

## Security Warning

I would be remiss if I did not specifically clarify a security concern
regarding containers: __containers are not a securely isolated solution__.
Although we spoke of the isolation that containers offer, they fundamentally
share the kernel with the host operating system, and are therefore not
secure in their own right. The common solution is to run Docker from within
a (single) virtual machine. Since the VM _is_ securely isolated, the host
machine is not at risk.


[docker-install]: https://docs.docker.com/engine/installation/
[docker-hub]: https://hub.docker.com
[docker-volumes]: https://docs.docker.com/engine/tutorials/dockervolumes/

