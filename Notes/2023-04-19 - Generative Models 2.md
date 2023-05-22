# GAN
As we know, in a GAN, we want to train a generator, and then, we want to generate data that  is similar to what is present in our dataset.  
We then have a discriminator network, which  will output 0 or 1 accordingly. 
![[GAN.png]]

#### GAN's loss function
From a mathematical standpoint, we could see this process as a MinMax game:
![[minmax_GAN.png]]
The loss functoin has 2 parts:
1. negative cross entropy of the discriminator w.r.t the true data distribution 
2. negative cross entropy of the “false” discriminator w.r.t the fake generator. 
	- This is also the only part in which the Generator has an influence on the loss. 

The training must be done alternately for the _discriminator_ and the _generator_. Otherwise, our generator will always be shitty. 
In particular, we alternately train the discriminator, freezing the generator, and the generator freezing the discriminator. 

We could see this process as having a student (generator) and a teacher (discriminator). 

### Example 
In this example, we have a Green curve (associated with our generator/generated data) and a black curve (associated to the real data distribution). We could see the line $x$ as the latent space, while the line $z$ is the real data distribution.
![[gan_graph.png]]
One problem: since the purpose of the generator is just to fool the discriminator, the generator could find _a single picture_ that would _fool the discriminator_ and always choose that picture, ingnoring the other points in the latent space, so there's no guarantees that its generations will be varied. This is called the __Mode Collapse problem__.
We will see later how to solve this.  

## Properties and applications of GAN

#### Staying inside the data manifold 
GAN drives the reconstruction towards the natural image manifold producing perceptually more convincing solutions wrt MSE-based solutions.
So, essentially, _GANs images arent subject to the manifold problem_.

## GANs problems 
- ==the fact that the discriminator get fooled does not mean the fake is good== (neural networks are easily fooled) 
- problems with counting, perspective, global structure, ... 
	- i.e. horse with 6 legs
- __mode collapse__: generative specialization on a good, fixed sample.

# Latent space exploration
Small movements in the latent space can result in some small modification in the visible image. 
![[latent_space_exploration.png]]
For example, what if we wanted to transform a picture of a cat into a picture of a dog, we take all the latent representation of cats that we know, and we find the hyperplane which separates the pictures of the dogs from the picture of the cats. 
After that, we move into the perpendicular direction of the hyperplane. 

The generative process is continuous: a small ==deplacement in the latent space produces a small modification in the visible space==. Real-world data depends on a relatively _small number of explanatory factors of variation_ (latent features) providing compressed internal representations. Understanding these features _we may define __trajectories___ producing desired alterations of data in the visible space.

## Entanglement and disentanglement
One of the problem of latent exploration/modification of attributes is that many times, these attributes that we're are interested in are not so __disentangled__.
- i.e. the transformation that allows a picture to have glasses makes also the person older etc.

When there is more than one attribute, editing one may affect another since some semantics can be coupled with each other (entanglement). To achieve more precise control (disentanglement), we can use _projections_ to force the _different directions of variation to be __orthogonal__ to each other_.
![[disentanglement_projection.png]]

## Comparing spaces
Another cool experiment is the comparison of latent spaces between GANs.
![[comparing_spaces.png]]
Frequently, we can map the latent space of the same GAN (trained differently) by using a simple linear mapping, and preserving most of the content. 

The organization of the latent space seems to be independent from 
- the training process 
- the network architecture 
- the learning objective: ==GAN and VAE share the same space==! 
The map can be defined by a small set of points common to the two spaces: _the support set_. Locating these points in the two spaces is enough to define the map.
The latent space mostly depends on the dataset. 

# Diffusion Models 
![[diffusion_models 1.png]]
In diffusion models, we have to do the reverse of what is called a __forward diffusion process__. In this type of processes, we essentially are distributing loss on a matrix of pixel, so by reversing that we are instead _denoising_ the image. 

After the image is denoised, we restart from this guess of what the real image would be, and then we reinject the noise at a smaller rate an we try to remove the noise again. 

The number of steps highly depends on the model. 

## The denoising network 
The denoising network implements the inverse of the operation of adding a given amount of noise to an image (direct diffusion) -> it removes a _specific amount of noise_ from the image.
The denoising network takes in input: 
1. a noisy image $x_t$ 
2. a signal _rate_ $α_t$ expressing the _amount of the original signal remaining in the noisy image_. 
and try to _predict the noise_ in it:
$$\epsilon_{\theta}(x_t, \alpha_t)$$
The predicted image would be:
![[predicted_image_formula.png]]

In other words, it receives in input a noise quantity and tries to remove that quantity of noise from the image. 

## Training step 
- take an input image $x_0$ in the training set and normalize it
- consider a signal ratio $α_t$
- generate a random noise $\epsilon \sim N(0,1)$
- generate a _noisy version_ $x_t$ of $x_0$ defined as
![[train_diff1.png]]
- let the network predict the noise $\epsilon_θ(x_t, α_t)$ from the noisy image $x_t$ and the signal ratio $α_t$
- train the network to _minimize the prediction error_, namely $||\epsilon - \epsilon_θ (x_t, \alpha_t)||$
	- meaning, the actual noise minus what was predicted.  

## Sampling procedure
With $T$ as the number of steps:
- fix a _sheduling_ $α_T > α_{T−1} > ... > α_1$ 
- start with a random noisy image $x_T ∼ N(0, 1)$ 
- for $t$ in $T...1$ do: 
	- compute the predicted error $\epsilon_θ(x_t , α_t)$ 
	- compute the current approximation of the result, that is the predicted image formula:
		![[predicted_image_formula.png]]
	- obtain $x_{t−1}$ reinjecting noise at rate $α_{t−1}$, namely
		![[noise_reinject.png]]

## A network for denoising 
Use a (conditional) Unet, since it is very good for image to image transformations. 


We could see this denoising process as sort of collapsing the space over the data points that we're interested in.  