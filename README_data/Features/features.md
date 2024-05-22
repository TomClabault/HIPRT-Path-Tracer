[\[Pres\]]() Disney BSDF (Diffuse, fake subsurface, metallic, roughness, anisotropy + anisotropy rotation, clearcoat, sheen, glass, volumetric Beer-Lambert absorption, ...) \[Burley, 2015\]
	- For experimentation purposes, the BRDF diffuse lobe can be switched for either:
		- The original "Disney diffuse" presented in [\[Burley, 2012\]](https://disneyanimation.com/publications/physically-based-shading-at-disney/)
		- A lambertian distribution
		- The Oren Nayar microfacet diffuse model.
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
### TODO
- Per-pixel adaptive sampling

### TODO
- Texture support for all the parameters of the BSDF

### TODO
- Normal mapping

### TODO
- Interactive ImGui interface + interactive first-person camera

### TODO
- Different frame-buffer visualisation (visualize the adaptive sampling map, the denoiser normals / albedo, ...)

### TODO
- Use of the ASSIMP library to support [many](https://github.com/assimp/assimp/blob/master/doc/Fileformats.md) scene file formats.

### TODO
- Intel Open Image Denoise + Normals & Albedo AOV support

### TODO