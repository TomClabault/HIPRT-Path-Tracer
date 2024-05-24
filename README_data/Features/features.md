### TODO
- Disney BSDF (Diffuse, fake subsurface, metallic, roughness, anisotropy + anisotropy rotation, clearcoat, sheen, glass, volumetric Beer-Lambert absorption, ...) \[Burley, 2015\]
	- For experimentation purposes, the BRDF diffuse lobe can be switched for either:
		- The original "Disney diffuse" presented in [\[Burley, 2012\]](https://disneyanimation.com/publications/physically-based-shading-at-disney/)
		- A lambertian distribution
		- The Oren Nayar microfacet diffuse model.
### TODO
- Texture support for all the parameters of the BSDF
### TODO

- BSDF Direct lighting multiple importance sampling
### TODO

- HDR Environment map + importance sampling using
	- CDF-inversion binary search
### TODO
- Emissive geometry light sampling

### TODO
- Nested dielectrics support 
	- Automatic handling as presented in \[Ray Tracing Gems, 2019\]
	- Handling with priorities as proposed in \[Simple Nested Dielectrics in Ray Traced Images, Schmidt, 2002\]
### Per-pixel adaptive sampling

Adaptive sampling is a technique that allows focusing the samples on pixels that need more of them. This is useful because not all parts of a scene are equally complex to render.

Consider this modified cornell box for example:

![Cornell box PBR reflective caustic reference](./img/cornell_pbr_reference.jpg)

Half of the rays of this scene don't even intersect any geometry and directly end up in the environment where the color of the environment map is computed. The variance of the radiance of these rays is very low since a given camera ray direction basically always results in the same radiance (almost) being returned.

However, the same cannot be said for the reflective caustic (the emissive light panel reflecting off the mirror small box) at the top right of the Cornell box. A camera ray that hits this region of the ceiling then has a fairly low chance of bouncing in direction of the small box to then bounce directly in the direction of the light. This makes the variance of these rays very high which really slows down the converge of this part of the scene. As a result, we would like to shoot more rays at these pixels than at other parts of the scene.

Adaptive sampling allows us to do just that. The idea is to estimate the error of each pixel of the image, compare this estimated error with a user-defined threshold $T$ and only continue to sample the pixel if the pixel's error is still larger than the threshold.

A very simple error metric is that of the variance of the luminance $\sigma^2$ of the pixel. In practice, we want to estimate the variance of a pixel across the $N$ samples $x_k$ it has received so far. 

The variance of $N$ samples is usually computed as:
#### $$\sigma^2 = \frac{1}{N}\sum_{k=1}^N (x_k - \mu) ^2$$

However, this approach would imply keeping the average of each pixel's samples (which is the framebuffer itself so that's fine) as well as the values of all samples (that's not fine). Every time we want to estimate the error of a single pixel, we would then have to loop over all the previous samples to compute their difference with the average and get our variance $\sigma^2$. Keeping track of all the samples is infeasible in terms of memory consumption (that would be 2GB of RAM/VRAM for a mere 256 samples' floating-point luminance at 1080p) and looping over all the samples seen so far is computationally way too demanding.

The practical solution is to evaluate the running-variance of the $N$ pixel samples $x_k$:
#### $$\sigma^2 = \frac{1}{N - 1} \left(\sum_{k=1}^N x_k^2 - \left( \sum_{k=1}^N x_k \right)^2\right)$$
  *Note that due to the nature of floating point numbers, this formula can have some precision issues. [This](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm) Wikipedia article presents good alternatives.*

With the variance, we can compute a 95% confidence interval $I$:
#### $$I = 1.96 \frac{\sigma}{\sqrt{N}}$$
This 95% confidence interval gives us a range around our samples mean $\mu$ and we can be 95% sure that, for the current number of samples $N$ and and their variance $\sigma$ that we used to compute this interval, the converged mean (true mean) of an infinite amount of samples is in that interval.

![Confidence interval visualization](./img/confidenceInterval.png)

*Visualization of the confidence interval **I** (green arrows) around **µ**.*

Judging by how $I$ is computed, it is easy to see that as the number of samples $N$ increases or the variance $\sigma^2$ decreases (and thus $\sigma$ decreases too), $I$ decreases. 

That should make sense since as we increase the number of samples, our mean $\mu$ should get closer and closer to the "true" mean value of the pixel (which is the value of the fully converged pixel when an infinite amount of samples are averaged together). 

If $I$ gets smaller, this means for our $\mu$ that it also gets closer to the "true" mean and that is the sign that our pixel has converged a little more.

![Confidence interval visualization](./img/confidenceInterval2.png)

*As the number of samples increases (or as the computed variance decreases), **I** gets smaller, meaning that the true mean is closer to our current mean which in turn means that our pixel has converged a little more.*

Knowing that we can interpret $I$ as a measure of the convergence of our pixel, the question now becomes: 

**When do we assume that our pixel has sufficiently converged and stop sampling?**

We use that user-given threshold $T$ we talked about earlier! Specifically, we can assume that if:
#### $$I \leq T\mu$$
Then that pixel as is converged enough for that threshold $T$. As a practical example, consider $T=0$. We then have:
#### $$I \leq T\mu \ \ \Leftrightarrow \ \ I \leq 0$$
If $I =0$, then the interval completely collapses on $\mu$ (as visualized in the above graphs). Said otherwise, $\mu$ **is** the true mean and our pixel has completely converged. Thus, for $T=0$, we will only stop sampling the pixel when it has fully converged.

In practice, having $I=0$ is infeasible. After some experimentations a $T$ threshold of $0.1$ seem to target a very reasonable amount of noise. Any $T$ lower than that represents a significant overhead in terms of rendering time for a visually incremental improvement on the perceived level of noise:

![cornellThreshold](./img/cornellThreshold.jpg)
**

### TODO
- Normal mapping

### TODO
- Interactive ImGui interface + interactive first-person camera

### TODO
- Different frame-buffer visualisation (visualize the adaptive sampling map, the denoiser normals / albedo, ...)

### TODO
- Use of the ASSIMP library to support [many](https://github.com/assimp/assimp/blob/master/doc/Fileformats.md) scene file formats.

### TODO
 - Optimized application startup time with:
	- Multithreaded texture loading
	- Asynchronous path tracing kernel compilation
### TODO
- Intel Open Image Denoise + Normals & Albedo AOV support

### TODO