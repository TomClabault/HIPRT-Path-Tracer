/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Compiler/GPUKernelCompiler.h"
#include "Scene/SceneParser.h"
#include "Threads/ThreadFunctions.h"
#include "Threads/ThreadManager.h"
#include "UI/Interaction/LinuxRenderWindowMouseInteractor.h"
#include "UI/Interaction/WindowsRenderWindowMouseInteractor.h"
#include "UI/RenderWindow.h"
#include "Utils/Utils.h"

#include <functional>
#include <iostream>

#include "stb_image_write.h"

// GPUKernelCompiler for waiting on threads currently reading files on disk
extern GPUKernelCompiler g_gpu_kernel_compiler;
extern ImGuiLogger g_imgui_logger;

// ******* TODO ReSTIR PT & refactor **********
// - Remove ReSTIR GI impl
// - NEE++ visibility queries for direct-light reuse
// - Remove LTC shading
// - Remove ReGIR
// - Remove RIS LTC estimator
// - Rename tree cut to light cut everywhere
//
// Ideas for neural importance sampling many lights:
//	- Splitting in the subtree + RIS
//		Or maybe juste do splitting in the subtree if visibility variance is high
//		The better theoretical question is when is splitting needed? This is probably not necessarily when vis variance is high
//	- Wider subtree + sample one root cluster brute force WRS
//		Or if caching spatially works well, do caching of the wider subtree as well to avoid
//	- Double mlp in the subtree as well ? i.e. multiple level MLP. One MLP inference for probabilities on the root node wide clusters and another MLP inference
//	for the subtree wide root clusters.
//		Or maybe a single MLP that does all that
//	- How to to NISML but on 1024 root node clusters?
//  - Why is it that 50% of training samples budget doesn't learn much faster than 15%? What is the theory behind that?
//	- Specular isn't amazing so do we even need the view direction in the input of the net?
//
// Ideas for learning to cluster:
//	- Spatial filtering to reduce grid artifacts, Practical Path Guiding 2019
//	- Splitting in the subtree + RIS
//		Or maybe juste do splitting in the subtree if visibility variance is high
//		The better theoretical question is when is splitting needed? This is probably not necessarily when vis variance is high
//	- Wider subtree + sample one root cluster brute force WRS
//		Or if caching spatially works well, do caching of the wider subtree as well to avoid
//	- Why doesn't learning to cluster reach 0 variance? Where is the bottleneck? Compare against optimal brute force and find where the bottleneck is
//	- Online Bayesian regression for the optimal cluster sampling probabilities assuming that the subtree sampler isn't optimal
//	- Use a hash grid for normal aware stuff instead of dense 6-face, same as NISML
//	- When to stop learning automatically?
//	- Rename clustering to lightcut: IlluminationAwareKDTreeLightClusteringData ---> IlluminationAwareKDTreeLightcutData
//	- How to refine more aggressively? Because now that we are learning from all samples, waiting fir lightcut refinement is actually the bottleneck, we need to
//		refine more aggressively to learn better & faster
//	- Is it worth it to only use a few SG lobes for descending the subtree? It's going to be much faster and maybe we don't need the precision of many lobes at
//		that point? Maybe variance will be fine while being much faster?
//		------> Yes it's much better
//	- Can we do something that accumulates NEE samples in the clusters of the light cuts, update statistics, refine, potentially split the lightcut and then
//		accumulate the samples again? The goal is to be able to learn (update statistics)/split the lightcut multiple times per SPP to greatly accelerate
//		learning. For that we should probably drop the reservoir logic per cluster and just keep every NEE samples, same as for the KD tree update and then just
//		replay those samples into learning to cluster, finding what cluster of the lightcut that light sample went through and accumulate that light sample into
//		that cluster of the cut
//
//
// TODO SG Light tree
//	- Maybe still do the hard coded distributions, cache points style, may still be good
//	- How to use more SG spatial lobes per precomputed nodes of the tree cut (which is basically free quality) but no more of these lobes when traversing the
// subtrees to not tank perf
//	- Can we somehow have a root node that is very large (1024?) and build a conservative distribution on it, cache points like. Basically what was done for
//		ReGIR: one CDF per spatial cell in the scene and one distribution over the 1024 root nodes per spatial cell.
//		We can theory test this by having a 1024 wide root node and just brute force WRS 1024x over it just to see if quality would at least be good. And if
//		yes, then use the CDF + spatial structure that will be an approximation but useable in speed at least
//
//		We could go a bit potato with the number of lobes per node because this is all precomputation so
//
//		We'll have to cover the bias or not use hard rejections in the tree importance, see which one is better for efficiency
//
//		Adaptive number of nodes per distribution to cover ~95% or something of the incoming energy (that incoming energy can be estimated by the importance of
//		the root node of the tree? se we don't have to loop through all 1024 nodes, sum the importances and then only keep the 95% best, that would be looping
//		twice)?
//
//		To produce the 1024 root nodes, what if we use splitting until we have 1024 samples? instead of the same 1024 nodes for every cell? For the PDF: Store a
//		macro-root ID on every light primitive.
//
//
//		We're going to need a fallback for the cached light distributions so we need a wide tree for that, because we're not going to use splitting for the
//		fallback so we can use a wide tree for lower variance
//
//	- Can the 1024-wide root node approach even be combined with a tree wider than binary below the 1024 nodes?
//	- For visibility, maybe drop splitting and use some NEE++ style thing between shading points and all those 1024 nodes
//	- Introduce cost in the splitting factor because splitting as long as we reduce variance forgets about cost and we could be splitting for little gains
//	- How to reduce splitting when specular dominates? LTC Sampling + BSDF MIS (RIS incident lighting even?) should be enough for that
//	- How to approximate the PDF to gain in speed for MIS? PDF replay is super expensive with splitting
//	- Implement new splitting factors and everything in ATS as well and see
//	- Can we vary the MaxLightSamples of splitting based on NEE variance at the shading point?
//	- How to reduce splitting intensity where variance is very low in the scene (back of the couch in the white room)
//  - Max optimize SG tree
//	- Can we do even better than SG-ATS hybrid while maybe more expensive
//	- Compare current tree against best theoretical (importance of node = iterate over all triangles and sum contributions): where is the gap?
//	- How to make splitting work with higher tree arity?
//
//
//
//
// - Multithread app startup scene loading with kernel compilation
// - Brightening bias at > 32 neighbors pairwise MIS?
// - Ray volume state reconstruction @ sample point
// - Reduce number of NEE candidates (light tree splitting) based on bounce depth
// - Remove all BSDF incident light info optimizations, so annoying to maintain and probably not that much perf to gain? Test perf loss
// - ReSTIR Spatial reuse doesn't have to store the sample for the output reservoir, we can just store the pixel index of the sample and then fetch it when
// needed, massively reduce registers needed since we don't keep the selected sample in the output reservoir anymore
// - Issue with alpha testing windows in scandinavian studio, unusual noise
// - Can we use a target function without the second BSDF for performance?
// - fp16 wherever possible and compression and everything for SPMIS
// - Could it be possible to precompute spatial reuse neighbors ahead of time to be able to share computations with pairwise MIS, basically doing paired spatial
// reuse of restir pt enhanced but while keeping SPMIS capability
//		- Or maybe we can produce a map of random seeds and those random seeds choose the spatial neighbors, this could also be used to share duplicated
// computations
// - We don't have to store the ReSTIR **samples** in the spatial pass. We can just store a pixel index and then on the next pass, when we need the sample, we
// can use that pixel index to go fetch the sample at the right pixel
//
// PSS or solid angle? Read papers to see what they need
//	- Check Area ReSTIR
//	- PT enchanced
//	- MCMC Decoupled shading
//	- Mutations

// TODO known bugs / incorrectness:
// - Minecraft harbor + principled BSDF + envmap --> iron bars look dark with ReSTIR PT at 0 bounces but should reflect the envmap at least
// - ImGui Material editor crash with 0 materials in the scene (only default material when blender export for example)
// - Adaptive sampling broken in furance test?
// - IOR 1 glass at roughness 1 without energy conservation loses a lot of energy but it shouldn't even be here because it's IOR 1
// - Updating the emissive property of a material in ImGui should update the emissive triangle primitive indices buffer: a material that goes from emission 1 to
//		emission 0 should be removed from that buffer but it's not at the moment
// - There is some weird color corruption issue with NEE++ linear probing max steps = 16 + ReGIR
// - take transmission color into account when direct sampling a light source that is inside a volume: leave that for when implementing proper volumes?
// - denoiser AOVs not accounting for transmission correctly since Disney  BSDF
//	  - same with perfect reflection
// - threadmanager: what if we start a thread with a dependency A on a thread that itself has a dependency B? we're going to try join dependency A even if
// thread with dependency on B hasn't even started yet --> joining nothing --> immediate return --> should have waited for the dependency but hasn't
// - Thin-film interference energy conservation/preservation is broken with "strong BSDF energy conservation" --> too bright (with transmission at 1.0f), even
// with film thickness == 0.0f
// - When overriding the base color for example in the global material overrider, if we then uncheck the base color override to stop overriding the base color,
// it returns the material to its very default base color  (the one  read from the scene file) instead of  returning it to what the user may have modified up to
// that point
// - Probably some weirdness with how light sampling is handled while inside a dielectric: inside_surface_multiplier? cosine term < 0 check? there shouldn't be
// any of that basically, it should just be evaluating the BSDF
// - For any material that is perfectly specular / perfectly transparent (the issue is most appearant with mirrors or IOR 1 glass), seeing the envmap through
// this object takes the envmap intensity scaling into account and so the envmap through the object is much brighter than the main background (when camera rays
// miss the scene and hit the envmap directly) without background envmap intensity scaling:
// https://mega.nz/file/x8I12Q6b#DJ2ZobBav9rwFdtvTX-CmgA1eFEgKprjXSvOg0My38o
// - White furnace mode not turning emissives off in the cornell_pbr with ReSTIR GI?

// - Test ReSTIR GI with diffuse transmission

// TODO ReSTIR
// - How to have better confidence weights that are proportional to variance rather than just counting how many samples it has seen?
// - Is multiple temporal buffers a good idea for reducing correlations? Such that temporal reuse has more potential candidates to choose from. We need
// something to avoid duplicated in the temporal buffer though.
//		Said otherwise, it's about having multiple temporal reservoirs per pixel. RIS without duplicates? What's research on that?
// - Sample space filtering paper: really good for diffuse. Advances in rendering IV in mega
// - For the hash grid restir spatial neighbord trick, we can probably add trivial antithetic sampling in there by sorting all neighbors within a hash grid
//		Even build a CDF on the GPU instead of antithetic sampling
//		We're going to need the theory of uhhhh though for unbiased sampling based on sample value
//		Can we use the super pixel algorithm instead of hashed screen space grid for restir spatial reuse
// - For hash grid screen space spatial reuse,
//		- We can sort the samples by intensity before building the CDF and then sample the CDF with a blue noise texture to get blue noise output from ReSTIR,
// amazing.
//		- Maybe we're going to need the paper on stratified RIS to keep the blue noise properties here? Otherwise is going to destroy the blue noise properties?
//		- SLIC superpixel to group pixels together and reuse in these groups instead of with a hash grid? We would run SLIC on a denoised image and this would
// give us lighting discontinuities as well, to not reuse accross lighting discontinuities
// - For adaptive sampling + restir we can use that idea of keeping relevant neighbors in a screen space hash grid such that we reuse good neighbors directly
// and never reuse stale neighbors
// - Can we use visibility variance to guide reservoir visibility reuse?
// - Reduce spatial reuse radius the lower the roughness
// - We shouldn't shoot a shadow ray in the light evaluation if the BSDF sample was chosen because this already has visibility
// - What about replacing visibility reuse with NEE++?
// - Can we do something for restir that has a hash grid for the first hits of the rays and then for spatial reuse, each pixel looks up its cell and reuse paths
// from the same cell (and thus same geometry if we include the normals in the hash grid). This would basically be a more accurate version of the directional
// spatial reuse
//		- One issue that we're going to have is: for a given pixel, we can compute it hash cell but then how do we know which other reservoirs (neighbors) are
// in the same hash cell?
//			- Fix that by: counting how different hash cell the primary hits create
//			- Create one counter per hash cell
//			- Count how many pixels fall in a given hash cell
//			4. Index the hash cell from 0 to N-1 where N is the number of hash cells
//			- Then we can have a pass that assigns each pixel index to a hash cell:
//				- For each pixel, find its hash cell. From the hash cell index of step 4., we know where in the fullscreen-wide buffer we need to write the
// pixel index by using the prefix sum of the hash cell counters up til the current hash cell index
//			- Once that's done, we know for a given pixel how many valid neighbors there are and what's their pixel indices
//
// - For the spatial reuse buffer, we don't have to store a whole grid at all, we can just store the index of the cell the reservoir reused from --> massive
// VRAM saves
// - Using the indirect index for the spatial output buffer, can we double buffer the initial candidates grid and run the spatial reuse of ReGIR async of the
// path tracing too?
// - There is bias in ReSTIR
// - Greedy spatial reuse to retry neighbors if we didn't get a good one
//			For the greedy neighbor search of restir spatial reuse, maybe reduce progressively the radius ?
// - memory coalescing aware spatial reuse pattern --> per warp / per half warp to reduce correlation artifacts?
// - can we maybe stop ReSTIR GI from resampling specular lobe samples? Since it's bound to fail anwyays. And do not resample on glass
// - See how many pixels of ReSTIR GI end up with the initial candidate as the final sample --> we can reuse NEE at the first hit for those samples in the
// shading pass instead of recomputing NEE
// - BSDF MIS reuse for ReSTIR
// - Force albedo to white for spatial reuse? Because what's interesting to reuse is the shape of the BRDF and the incident radiance. Resampling from a black
// diffuse is still interesting. The albedo doesn't matter
// - Have a look at compute usage with the profiler with only a camera ray kernel and more and more of the code to see what's dropping the compute usage
// - If it is the canonical sample that was resampled in ReSTIR GI, recomputing direct lighting at the sample point isn't needed and could be stored in the
// reservoir?

//
// ------------------- DO AFTER WAVEFRONT -------------------
// - maybe have shaders without energy compensation? because this do be eating quite a lot of registers
// - let's do some ray reordering because in complex scenes and complex materials and without hardware RT; this may actually  be quite worth it
// - dispatch mega kernel when only a few rays are left alive after compaction?
// - investigate where the big register usage comes from (by commenting lines) --> split shaders there?
// - split shaders for material specifics and dispatch in parallel?
// - use wavefront path tracing to evaluate direct  lighting, envmap and BSDF sample in parallel
// - start shooting camera rays for frame N+1 during frame N?
// - compaction - https://github.com/microsoft/directxshadercompiler/wiki/wave-intrinsics#example
// - launch bounds optimization?
// - thread group size optimization?
// - double buffering of frames in general to better keep the GPU occupied?
// - can we gain in performance by having the trace rays functions in completely separate passes so that we can have the maximum amount of L1 cache in the
// passes that now don't trace rays? (and use max amount of shared mem in the rays only passes)
// ------------------- DO AFTER WAVEFRONT -------------------

// TODO Features:
// - Stochastic vertex NEE: do NEE at only one vertex of the path but choose that vertex probalistically to avoid bias. The question is then how to choose the
// vertex probability correctly? NEE-only radiance cache?
// - Radiance cache somehow, have a look at what HouseOfCards is doing, looks pretty good
// - Neural incident radiance cache
// - Use the neural incident radiance cache to do specular reflections resampling with ReSTIR: when resampling a specular neighbor, estimate target function at
// center with BSDF_center * incident_radiance_cache_direction_of_neighbor_same_random_seed
// - Neural path guiding anisotropic gaussians
// - Neural visibility cache for envmap sampling?
// - Another separate render graph for interactivity
// - Use only packed material throughout the shaders to save registers?
// - We can use incoming radiance radiance cache to sample BSDF directions for MIS: we would cache the incoming radiance only from emissives and use that with
//		RIS when sampling a BSDF direction for MIS maybe, somethiung lioke that
// - With our radiance cache, we can fully use the paper that helps with cache placement
// - For caching incoming radiance in a radcache, we can use a SH representation or this more precise representation here: https://suikasibyl.github.io/gilo/#/
// - Better adaptive sampling + denoiser: https://www.cg.tuwien.ac.at/research/publications/2025/sakai-2025-stater/sakai-2025-stater-paper.pdf
// - Constant memory for Render data such that it is available everywhere and we don't have it to pass it around all the time.
//		- Same for more variables?
// - Can we use LTCs for sampling the BSDFs with energy compensation? Since energy compensation has high variance, LTCs may help a lot there?
// - Variance aware MIS weights? https://cgg.mff.cuni.cz/~jaroslav/papers/2019-variance-aware-mis/2019-grittmann-variance-aware-mis-paper.pdf
// - RISLTC: https://data.ishaanshah.xyz/research/pdfs/risltc.pdf. Some explanations in there for projected solid angle and LTC sampling
// - Inciteful graph to explore (started with Practical product sampling warping NVIDIA):
// https://inciteful.xyz/p?ids%5B%5D=W4220995884&ids%5B%5D=W3179788358&ids%5B%5D=W4403641440&ids%5B%5D=W4390345185&ids%5B%5D=W4388994411&ids%5B%5D=W4200187284&ids%5B%5D=W2885975589&ids%5B%5D=W3183450244&ids%5B%5D=W1893031899&ids%5B%5D=W3036883119&ids%5B%5D=W3044759327&ids%5B%5D=W4240396283&ids%5B%5D=W3110265079&ids%5B%5D=W2073976119&ids%5B%5D=W2988541899&ids%5B%5D=W2885239691&ids%5B%5D=W2964425571&ids%5B%5D=W2030242873&ids%5B%5D=W3044185278
// - VisibilityCluster: Average Directional Visibility for Many-Light Rendering: https://ieeexplore.ieee.org/document/6464264
// - Practical product sampling warping NVIDIA, there's a shadertoy for that
// - Sample specular/diffuse lobe with the luminance of the diffuse lobe
// - Sample specular/diffuse by taking the throughput of the path into account? To avoid sampling a green diffuse lobe when the throughpuit is all red for
// example
// - Sample by evaluating the contribution of both samples and choosing proportional to the contribution:
//		- a next "clever way" would be to generate L with both diffuse and specular but using the sample random number, then compare their total "contributions"
//(whole specular+diffuse BRDF value divided by pdf of generator and multiplied by path prefix throughput), then depending on the luma of that you choose either
// the first or second sample.
//		- so you're making decisions about what branch you take posteriori not a-priori.
//
//		- This has a few drawbacks :
//		- you're computing the contribution twice (but you're not really doing double the work for generation because Low Discrepancy Sequences / random
// numbers->maximum divergence)
//		- if you need the PDF for MIS or RIS for a given L you need to do far more work, your sampling routines must be invertible
//
//		- The latter part is annoying because many BRDF sampling routines don't require you to find the xi which produce a given L when you want to query
// pdf(L).
//		- However the issue is that for every 2D value of xi you have two values of L between which you've chosen based on the value of the whole BRDF, so to
// know the PDF of any of the L in the pair, you need to know exactly what the other L is.
//
//		This is why this method is not tractable / fun for more than 2 BRDFs.
//
// - Vector valued monte carlo: https://suikasibyl.github.io/files/vvmc/paper.pdf
// - Reweighting path guiding: https://zhiminfan.work/paper/mi_reweight_preprint.pdf
// - Fixed balance heuristic: https://qingqin-hua.com/publication/2025-correct-balance/2025-correct-balance.pdf
// - Envmap with visibility sampling: https://static.chaos.com/documents/assets/000/000/377/original/adaptive_dome_abstract.pdf?1676455588
// - Faster PNG loading: https://github.com/richgel999/fpng
// - Need something blocking inn "start thread with dependency" so that the main thread is blocked until the other thread actually started. This should solve
// the issue where sometilmes the main threds just joins everyone but everyone hasn't even started yet
// - Can we have something like sharc but for light sampling? We store reservoirs in the hash table and resample everytime we read into the hash grid with some
// initial candidates?
//		- And maybe we can spatial reuse on that
//		- Issue with MIS weights though because the MIS weights here are going to be an integral over the scene surface of the grid cell
//			- Maybe SMIS and MMIS have something to say about that
// - Stochastic light culling: https://jcgt.org/published/0005/01/02/paper-lowres.pdf
// - Disney adaptive sampling: https://la.disneyresearch.com/wp-content/uploads/Adaptive-Rendering-with-Linear-Predictions-Paper.pdf?utm_source=chatgpt.com
// - flush to zero denormal float numbers compiler option?
//		// -fcuda-flush-denormals-to-zero
//		// -fgpu-flush-denormals-to-zero
// - Use a CPP preprocessor lib to preprocess shaders and see if some macro is used or not
//		- Also uses a dead code removal library such that we only have relevant code in the shader and we can know for sure which macros are used or not
// - Eta scaling for russian roulette refractions
// - Better adaptive sampling error metrics: https://theses.hal.science/tel-03675200v1/document, section 10.1.1, Heitz et al 2018 + Rigau et al 2003
// - Projected solid angle light sampling https://momentsingraphics.de/ToyRenderer4RayTracing.html
// - Disable back facing lights for performance because most of those lights, for correct meshes, are going to be occluded
//		- Add an option to re-enable manually back facing lights in the material
// - Efficient Image-Space Shape Splatting for Monte Carlo Rendering
// - DRMLT: https://joeylitalien.github.io/assets/drmlt/drmlt.pdf
// - What's NEE-AT of RTXPT?
// - Area ReSTIR just for the antialiasing part
// - Directional albedo sampling weights for the principled BSDF importance sampling. Also, can we do "perfect importance" sampling where we sample each
// relevant lobe, evaluate them (because we have to evaluate them anyways in eval()) and choose which one is sampled proportionally to its contribution or is it
// exactly the idea of sampling based on directional albedo?
// - Russian roulette improvements: http://wscg.zcu.cz/wscg2003/Papers_2003/C29.pdf
// - Some MIS weights ideas in: https://momentsingraphics.de/ToyRenderer4RayTracing.html in "Combining diffuse and specular"
// - Radiance caching for feeding russian roulette
// - Tokuyoshi (2023), Efficient Spatial Resampling Using the PDF Similarity
//		- Not for offline?
// - A Dynamically-Updating Hierarchical Stopping Condition for Monte Carlo Illumination
// - software opacity micromaps
// - Add parameters to increase the strength of specular / coat darkening
// - sample BSDF diffuse lobe proba based on its luminance?
// - how to help with shaders combination compilation times?
//		RocFFT has some ideas for parallel compilation
// https://github.com/ROCm/rocFFT/blob/e9303acfb993de98b78358f3bf6fdd93f810f5fd/docs/design/runtime_compilation.rst#parallel-compilation
//		- wavefront path tracing should help
//		- Maybe have two sets of shaders:
//			- One that uses the #if for performance
//			- One that uses if() everywhere instead of #if for fast preview
//				- to accelerate compilation times: we can use if() everywhere in the code so that switching an option doesn't require a compilation but if we
// want, we can then apply the options currently selected and compiler everything for maximum performance. This can probably be done with a massive shader that
// has all the options using if() instead of #if ? Maybe some better alternative though?
//				----------- That's a good one too ^
// - next event estimation++? --> 2023 paper improvement with the octree
// - ideas of https://pbr-book.org/4ed/Light_Sources/Further_Reading for performance
// - envmap visibility cache?
// - If GMoN is enabled, it would be cool to be able to denoise the GMoN blend between GMoN and the default framebuffer but currently the denoiser only denoises
// the full GMoN and nothing else
// - Exploiting Visibility Correlation in Direct Illumination
// - smarter shader cache (hints to avoid using all kernel options when compiling a kernel? We know that Camera ray doesn't care about direct lighting strategy
// for example)
// - for LTC sheen lobe, have the option to use either SGGX volumetric sheen or approximation precomputed LTC data
// - for volumes, we don't have to use the same phase function at each bounce, for artistic control of the "blur shape"
// - --help on the commandline
// - Normal mapping seems broken again, light rays going under the surface... p1 env light
// - performance/bias tradeoff by ignoring alpha tests (either for global rays or only shadow rays) after N bounce?
// - performance/bias tradeoff by ignoring direct lighting occlusion after N bounce? --> strong bias but maybe something to do by reducing the length of shadow
// rays instead of just hard-disabling occlusion
// - energy conserving Oren Nayar: https://mimosa-pudica.net/improved-oren-nayar.html#images
// - experiment with a feature that ignores really dark pixel in the variance estimation of the adaptive
//		sampling because it seems that very dark areas in the image are always flagged as very
//		noisy / very high variance and they take a very long time to converge (always red on the heatmap)
//		even though they are very dark regions and we don't even noise in them. If our eyes can't see
//		the noise, why bother? Same with very bright regions
// - Reuse miss BSDF ray on the last bounce to sample envmap with MIS
// - We're using an approximation of the clearcoated BSDF directional albedo for energy compensation right now. The approximation breaks down when what's below
// the coat is 0.0f roughness. We could potentially bake the directional albedo for a mirror-coated BSDF and interpolate between that mirror-coated LUT and the
// typical rough-coated BSDF LUT based on the roughness of what's below the coat. This mirror-coated LUT doesn't work very well if there's a
// smooth-dielectric-coated lambert below the coat so maybe we would need a third LUT for that case
// - For/switch paradigm for instruction cache misses? https://youtu.be/lxRgmZTEBHM?si=FcaEYqAMVO_QyfwX&t=3061
//		- kind of need a good way to profile that to see the difference though
// - have a light BVH for intersecting light triangles only: useful when we want to know whether or not a direction could have be sampled by the light sampler:
// we don't need to intersect the whole scene BVH, just the light geometry, less expensive ------> we're going to need another shadow ray though because if
// we're intersecting solely against the light BVH we don't have the rest of the geometry of the scene to occluded the lights. So we're going to need a shadow
// ray in case we do hit a light in the light BVH to make sure that light isn't occluded ----> Maybe collect statistics on how many BSDF rays light sample miss
// lights: this can help see what's going to be the benefit of a light BVH because the drawback of a light BVH is going to be only if we hit a light because
// then we need another BVH traversal to check for occlusion
// - shadow terminator issue on sphere low smooth scene: [Taming the Shadow Terminator], Matt Jen-Yuan Chiang,
// https://github.com/aconty/aconty/blob/main/pdf/bump-terminator-nvidia2019.pdf
// - use HIP/CUDA graphs to reduce launch overhead
// - linear interpolation (spatial, object space, world space) function for the parameters of the BSDF
// - compensated importance sampling of envmap
// - Product importance sampling envmap: https://github.com/aconty/aconty/blob/main/pdf/fast-product-importance-abstract.pdf
// - multiple GLTF, one GLB for different point of views per model
// - CTRL + mouse wheel for zoom in viewport, CTRL click reset zoom
// - clay render
// - build BVHs one by one to avoid big memory spike? but what about BLAS performance cost?
// - play with SBVH building parameters alpha/beta for memory/performance tradeoff + ImGui for that
// - ability to change the color of the heatmap shader in ImGui
// - do not store alpha from envmap
// - fixed point 18b RGB for envmap? 70% size reduction compared to full size. Can't use texture sampler though. Is not using a sampler ok performance-wise? -->
// it probably is since we're probably memory lantency bound, not memory bandwidth
// - look at blender cycles "medium contrast", "medium low constract", "medium high", ... --> filmic tonemapper does it?
// - normal mapping strength
// - blackbody light emitters
// - ACES mapping --> filmic tonemapper may be more comprehensive
// - better post processing: contrast, low, medium, high exposure curve --> filmic tonemapper
// - bloom post processing
// - BRDF swapper ImGui : Disney, Lambertian, Oren Nayar, Cook Torrance, Perfect fresnel dielectric reflect/transmit
// - choose principled BSDF diffuse model (disney, lambertian, oren nayar)
// - portal envmap sampling --> choose portals with ImGui
// - find a way to not fill the texcoords buffer for meshes that don't have textures
// - pack CPUMaterial informations such as texture indices (we can probably use 16 bit for a texture index --> 2 texture indices in one 32 bit register)
// - use 8 bit textures for material properties instead of float
// - use fixed point 8 bit for materials parameters in [0, 1], should be good enough
// - log size of buffers used: vertices, indices, normals, ...
// - log memory size of buffers used: vertices, indices, normals, ...
// - able / disable normal mapping
// - use only one channel for material property texture to save VRAM
// - Remove vertex normals for meshes that have normal maps and save VRAM
// - texture compression
// - WUFFS for image loading?
// - float compression for render buffers?
// - Exporter (just serialize the scene to binary file and have a look at how to do backward compatibility)
// - Allow material parameters textures manipulation with ImGui
// - Disable material parameters in ImGui that have a texture associated (since the ImGui slider in this case has no effect)
// - Upload grayscale texture (roughness, specular and other BSDF parameters basically) as one channel to the GPU instead of memory costly RGBA
// - Emissive textures sampling: how to sample an object that has an emissive texture? How to know which triangles of the mesh are covered by the emissive parts
// of the texture?
// - stream compaction / active thread compaction (ingo wald 2011)
// - sample regeneration
// - Spectral rendering / look at gemstone rendering because they quite a lot of interesting lighting effect to take into account (pleochroism, birefringent,
// dispersion, ...)
// - structure of arrays instead of arrays of struct relevant for global buffers in terms of performance?
// - data packing in buffer --> use one 32 bit buffer to store multiple information if not using all 32 bits
//		- pack active pixel in same buffer as pixel sample count
// - pack two texture indices in one int for register saving, 65536 (16 bit per index when packed) textures is enough
// - hint shadow rays for better traversal perf on RDNA3?
// - benchmarker to measure frame times precisely (avg, std dev, ...) + fixed random seed for reproducible results
// - alias table for sampling env map instead of log(n) binary search
// - image comparator slider (to have adaptive sampling view + default view on the same viewport for example)
// - thin materials
// - Have the UI run at its own framerate to avoid having the UI come to a crawl when the path tracing is expensive
// - When modifying the emission of a material with the material editor, it should be reflected in the scene and allow the direct sampling of the geometry so
// the emissive triangles buffer should be updated
// - Ray differentials for texture mipampping (better bandwidth utilization since sampling potentially smaller texture --> fit better in cache)
// - Visualizing ray depth (only 1 frame otherwise it would flicker a lot [or choose the option to have it flicker] )
// - Visualizing pixel time with the clock() instruction. Pixel heatmap:
//		- https://developer.nvidia.com/blog/profiling-dxr-shaders-with-timer-instrumentation/
//		- https://github.com/libigl/libigl/issues/1388
//		- https://github.com/libigl/libigl/issues/1534
// - Visualizing russian roulette depth termination
// - Statistics on russian roulette efficiency
// - feature to disable ReSTIR after a certain percentage of convergence --> we don't want to pay the full price of resampling and everything only for a few
// difficult isolated pixels (especially true with adaptive sampling where neighbors don't get sampled --> no new samples added to their reservoir --> no need
// to resample)
// - Realistic Camera Model
// - Focus blur
// - Flakes BRDF (maybe look at OSPRay implementation for a reference ?)
// - ImGuizmo for moving objects in the scene
// - choose denoiser quality in imgui
// - try async buffer copy for the denoiser (maybe run a kernel to generate normals and another to generate albedo buffer before the path tracing kernel to be
// able to async copy while the path tracing kernel is running?)
// - write scene details to imgui (nb vertices, triangles, ...)
// - choose env map at runtime imgui
// - choose scene file at runtime imgui
// - lock camera checkbox to avoid messing up when big render in progress
// - PBRT v3 scene parser
// - implement ideas of https://blog.selfshadow.com/publications/s2017-shading-course/imageworks/s2017_pbs_imageworks_slides_v2.pdf
// - Efficiency Aware Russian roulette and splitting
// - ReSTIR PT

void glfw_window_resized_callback(GLFWwindow* window, int width, int height)
{
	int new_width_pixels, new_height_pixels;
	glfwGetFramebufferSize(window, &new_width_pixels, &new_height_pixels);

	if (new_width_pixels == 0 || new_height_pixels == 0)
		// This probably means that the application has been minimized, we're not doing anything then
		return;
	else
	{
		// We've stored a pointer to the RenderWindow in the "WindowUserPointer" of glfw
		RenderWindow* render_window = reinterpret_cast<RenderWindow*>(glfwGetWindowUserPointer(window));
		render_window->resize(width, height);
	}
}

// Implementation from https://learnopengl.com/In-Practice/Debugging
void APIENTRY
RenderWindow::gl_debug_output_callback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam)
{
	// ignore non-significant error/warning codes
	if (id == 131169 || id == 131185 || id == 131218 || id == 131204)
		return;

	if (id == 131154)
		// NVIDIA specific warning
		// Pixel-path performance warning: Pixel transfer is synchronized with 3D rendering.
		//
		// Mainly happens when we take a screenshot
		return;

	if (id == 131154)
		// NVIDIA specific warning
		// Pixel-path performance warning: Pixel transfer is synchronized with 3D rendering.
		//
		// Mainly happens when we take a screenshot
		return;

	std::string source_str;
	std::string type_str;
	std::string severity_str;

	switch (source)
	{
	case GL_DEBUG_SOURCE_API:
		source_str = "Source: API";
		break;
	case GL_DEBUG_SOURCE_WINDOW_SYSTEM:
		source_str = "Source: Window System";
		break;
	case GL_DEBUG_SOURCE_SHADER_COMPILER:
		source_str = "Source: Shader Compiler";
		break;
	case GL_DEBUG_SOURCE_THIRD_PARTY:
		source_str = "Source: Third Party";
		break;
	case GL_DEBUG_SOURCE_APPLICATION:
		source_str = "Source: Application";
		break;
	case GL_DEBUG_SOURCE_OTHER:
		source_str = "Source: Other";
		break;
	}

	switch (type)
	{
	case GL_DEBUG_TYPE_ERROR:
		type_str = "Type: Error";
		break;
	case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
		type_str = "Type: Deprecated Behaviour";
		break;
	case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
		type_str = "Type: Undefined Behaviour";
		break;
	case GL_DEBUG_TYPE_PORTABILITY:
		type_str = "Type: Portability";
		break;
	case GL_DEBUG_TYPE_PERFORMANCE:
		type_str = "Type: Performance";
		break;
	case GL_DEBUG_TYPE_MARKER:
		type_str = "Type: Marker";
		break;
	case GL_DEBUG_TYPE_PUSH_GROUP:
		type_str = "Type: Push Group";
		break;
	case GL_DEBUG_TYPE_POP_GROUP:
		type_str = "Type: Pop Group";
		break;
	case GL_DEBUG_TYPE_OTHER:
		type_str = "Type: Other";
		break;
	}

	switch (severity)
	{
	case GL_DEBUG_SEVERITY_HIGH:
		severity_str = "Severity: high";
		break;
	case GL_DEBUG_SEVERITY_MEDIUM:
		severity_str = "Severity: medium";
		break;
	case GL_DEBUG_SEVERITY_LOW:
		severity_str = "Severity: low";
		break;
	case GL_DEBUG_SEVERITY_NOTIFICATION:
		severity_str = "Severity: notification";
		break;
	}

	g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR,
							"---------------\n"
							"Debug message (%d): %s\n"
							"%s\n%s\n%s\n\n",
							id, message, source_str.c_str(), type_str.c_str(), severity_str.c_str());

	// The following breaks into the debugger to help pinpoint what OpenGL
	// call errored
	Debug::debugbreak();
}

const std::string RenderWindow::PERF_METRICS_CPU_OVERHEAD_TIME_KEY = "CPUDisplayTime";

RenderWindow::RenderWindow(int renderer_width, int renderer_height, std::shared_ptr<HIPRTOrochiCtx> hiprt_oro_ctx)
	: m_viewport_width(renderer_width), m_viewport_height(renderer_height)
{
	// Adding the size of the windows around the viewport such that these windows
	// have their base size and the viewport has the size the the user has asked for
	// (through the commandline)
	int window_width  = renderer_width + ImGuiSettingsWindow::BASE_SIZE;
	int window_height = renderer_height + ImGuiLogWindow::BASE_SIZE;

	init_glfw(window_width, window_height);
	init_gl(renderer_width, renderer_height);
	ImGuiRenderer::init_imgui(m_glfw_window);

	m_application_state	   = std::make_shared<ApplicationState>();
	m_application_settings = std::make_shared<ApplicationSettings>();
	m_renderer			   = std::make_shared<GPURenderer>(this, hiprt_oro_ctx, m_application_settings);
	m_gpu_baker			   = std::make_shared<GPUBaker>(m_renderer);

	// Disabling auto samples per frame is accumulation is OFF
	m_application_settings->auto_sample_per_frame = m_renderer->get_render_settings().accumulate ? m_application_settings->auto_sample_per_frame : false;

	m_renderer->resize(renderer_width, renderer_height);

	ThreadManager::start_thread(ThreadManager::RENDER_WINDOW_CONSTRUCTOR,
								[this, renderer_width, renderer_height]()
								{
									m_denoiser = std::make_shared<OpenImageDenoiser>();
									m_denoiser->initialize();
									m_denoiser->resize(renderer_width, renderer_height);
									m_denoiser->set_use_albedo(m_application_settings->denoiser_use_albedo);
									m_denoiser->set_use_normals(m_application_settings->denoiser_use_normals);
									m_denoiser->finalize();

									m_perf_metrics = std::make_shared<PerformanceMetricsComputer>();

									m_imgui_renderer = std::make_shared<ImGuiRenderer>();
									m_imgui_renderer->set_render_window(this);

									// Making the render dirty to force a cleanup at startup
									set_render_dirty(true);
								});

	// Cannot create that on a thread since it compiles OpenGL shaders
	// which the OpenGL context which is only available to the thread it was created on (the main thread)
	m_display_view_system = std::make_shared<DisplayViewSystem>(m_renderer, this);

	// Same for the screenshoter
	m_screenshoter = std::make_shared<Screenshoter>();
	m_screenshoter->set_renderer(m_renderer);
	m_screenshoter->set_render_window(this);
}

RenderWindow::~RenderWindow()
{
	g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_INFO, "Exiting...");

	// Hiding the window to show the user that the app has exited. This is basically only useful if the
	// wait function call below hangs for a while: we don't want the user to see the application
	// frozen in this case. Note that we're *hiding* the window and not *destroying* it because
	// destroying the window also destroys the GL context which may cause crashes is some
	// other part of the app is still using buffers or whatnot
	glfwHideWindow(m_glfw_window);

	// Waiting for all threads that are currently reading from the disk (for compiling kernels in the background)
	// to finish the reading to avoid SEGFAULTING
	g_gpu_kernel_compiler.wait_compiler_file_operations();

	// Waiting for the renderer to finish its frame otherwise
	// we're probably going to close the window / destroy the
	// GL context / etc... while the renderer might still be
	// using so OpenGL Interop buffers --> segfault
	m_renderer->synchronize_all_kernels();
	// Manually destroying the renderer now before we destroy the GL context
	// glfwDestroyWindow()
	m_renderer = nullptr;
	// Same for the screenshoter
	m_screenshoter = nullptr;
	// Same for the baker
	m_gpu_baker = nullptr;
	// Same for the display view system
	m_display_view_system = nullptr;
	// Same for the imgui renderer
	m_imgui_renderer = nullptr;

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	glfwDestroyWindow(m_glfw_window);
}

void RenderWindow::init_glfw(int window_width, int window_height)
{
	g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_INFO, "Initializing GLFW...");
	if (!glfwInit())
	{
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "Could not initialize GLFW...");

		int trash = std::getchar();

		std::exit(1);
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, true);

#ifdef __unix__
	m_mouse_interactor = std::make_shared<LinuxRenderWindowMouseInteractor>();
#elif defined(_WIN32) || defined(WIN32)
	m_mouse_interactor = std::make_shared<WindowsRenderWindowMouseInteractor>();
#endif
	m_keyboard_interactor.set_render_window(this);

	const GLFWvidmode* mode = glfwGetVideoMode(glfwGetPrimaryMonitor());

	m_glfw_window = glfwCreateWindow(window_width, window_height, "HIPRT-Path-Tracer", NULL, NULL);
	if (!m_glfw_window)
	{
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "Could not initialize the GLFW window...");

		int trash = std::getchar();

		std::exit(1);
	}

	glfwMakeContextCurrent(m_glfw_window);
	// Setting a pointer to this instance of RenderWindow inside the m_window GLFWwindow so that
	// we can retrieve a pointer to this instance of RenderWindow in the callback functions
	// such as the window_resized_callback function for example
	glfwSetWindowUserPointer(m_glfw_window, this);
	glfwSwapInterval(1);
	glfwSetWindowSizeCallback(m_glfw_window, glfw_window_resized_callback);
	m_mouse_interactor->set_callbacks(m_glfw_window);
	m_keyboard_interactor.set_callbacks(m_glfw_window);

	glewInit();

	g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_INFO, "GLFW Initialized!");
}

void RenderWindow::init_gl(int width, int height)
{
	glViewport(0, 0, width, height);

	// Initializing the debug output of OpenGL to catch errors
	// when calling OpenGL function with an incorrect OpenGL state
	int flags;
	glGetIntegerv(GL_CONTEXT_FLAGS, &flags);
	if (flags & GL_CONTEXT_FLAG_DEBUG_BIT)
	{
		glEnable(GL_DEBUG_OUTPUT);
		glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
		glDebugMessageCallback(RenderWindow::gl_debug_output_callback, nullptr);
		glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_TRUE);
	}
}

void RenderWindow::resize(int pixels_width, int pixels_height)
{
	if (pixels_width == m_viewport_width && pixels_height == m_viewport_height)
	{
		// Already the right size, nothing to do. This can happen
		// when the window comes out of the minized state. Getting
		// in the minimized state triggers a queue_resize event with a new size
		// of (0, 0) and getting out of the minimized state triggers a queue_resize
		// event with a size equal to the one before the minimization, which means
		// that the window wasn't actually resized and there is nothing to do

		return;
	}

	glViewport(0, 0, pixels_width, pixels_height);

	m_viewport_width  = pixels_width;
	m_viewport_height = pixels_height;

	// Taking resolution scaling into account
	float& resolution_scale = m_application_settings->render_resolution_scale;
	if (m_application_settings->keep_same_resolution)
		// TODO what about the height changing ?
		resolution_scale = m_application_settings->target_width / static_cast<float>(pixels_width);

	int new_render_width  = std::floor(pixels_width * resolution_scale);
	int new_render_height = std::floor(pixels_height * resolution_scale);

	if (new_render_height == 0 || new_render_width == 0)
		// Can happen if resizing the window to a 1 pixel width/height while having a resolution scaling < 1.
		// Integer maths will round it down to 0
		return;

	m_renderer->resize(new_render_width, new_render_height);
	m_denoiser->resize(new_render_width, new_render_height);
	m_denoiser->finalize();

	m_display_view_system->resize(new_render_width, new_render_height);

	set_render_dirty(true);
}

void RenderWindow::change_resolution_scaling(float new_scaling)
{
	float new_render_width	= std::floor(m_viewport_width * new_scaling);
	float new_render_height = std::floor(m_viewport_height * new_scaling);

	m_renderer->resize(new_render_width, new_render_height);
	m_denoiser->resize(new_render_width, new_render_height);
	m_denoiser->finalize();
	m_display_view_system->resize(new_render_width, new_render_height);
}

int RenderWindow::get_width()
{
	return m_viewport_width;
}

int RenderWindow::get_height()
{
	return m_viewport_height;
}

bool RenderWindow::is_interacting()
{
	return m_mouse_interactor->is_interacting() || m_keyboard_interactor.is_interacting();
}

bool RenderWindow::render_resetted_with_imgui_item_held()
{
	// TODO this is a bit scuffed, this is just to avoid a bug that resets the render too often when holding ImGui widgets but that bug should be fixed in the
	// first place, this is just a band aid
	if (!m_renderer->reset_when_holding_imgui_items())
		return false;

	return m_application_state->m_render_resetted_with_imgui_item_held;
}

RenderWindowKeyboardInteractor& RenderWindow::get_keyboard_interactor()
{
	return m_keyboard_interactor;
}

std::shared_ptr<RenderWindowMouseInteractor> RenderWindow::get_mouse_interactor()
{
	return m_mouse_interactor;
}

std::shared_ptr<ApplicationSettings> RenderWindow::get_application_settings()
{
	return m_application_settings;
}

std::shared_ptr<ApplicationState> RenderWindow::get_application_state()
{
	return m_application_state;
}

std::shared_ptr<DisplayViewSystem> RenderWindow::get_display_view_system()
{
	return m_display_view_system;
}

DisplaySettings& RenderWindow::get_display_settings()
{
	return m_display_settings;
}

void RenderWindow::update_renderer_view_translation(float translation_x, float translation_y, bool scale_translation)
{
	if (scale_translation)
	{
		translation_x *= m_application_state->last_CPU_frame_delta_time_ms / 1000.0f;
		translation_y *= m_application_state->last_CPU_frame_delta_time_ms / 1000.0f;

		translation_x *= m_renderer->get_camera().camera_movement_speed * m_renderer->get_camera().user_movement_speed_multiplier;
		translation_y *= m_renderer->get_camera().camera_movement_speed * m_renderer->get_camera().user_movement_speed_multiplier;
	}

	if (translation_x == 0.0f && translation_y == 0.0f)
		return;

	set_render_dirty(true);

	glm::vec3 translation = glm::vec3(translation_x, translation_y, 0.0f);
	m_renderer->translate_camera_view(translation);
}

void RenderWindow::update_renderer_view_rotation(float offset_x, float offset_y)
{
	set_render_dirty(true);

	float rotation_x, rotation_y;

	rotation_x = offset_x / m_viewport_width * hippt::M_TWO_PI / m_application_settings->view_rotation_sldwn_x;
	rotation_y = offset_y / m_viewport_height * hippt::M_TWO_PI / m_application_settings->view_rotation_sldwn_y;

	// Inverting X and Y here because moving your mouse to the right actually means
	// rotating the camera around the Y axis
	m_renderer->rotate_camera_view(glm::vec3(rotation_y, rotation_x, 0.0f));
}

void RenderWindow::update_renderer_view_zoom(float offset, bool scale_delta_time)
{
	if (scale_delta_time)
		offset *= m_application_state->last_CPU_frame_delta_time_ms / 1000.0f;
	offset *= m_renderer->get_camera().camera_movement_speed * m_renderer->get_camera().user_movement_speed_multiplier;

	if (offset == 0.0f)
		return;

	set_render_dirty(true);

	m_renderer->zoom_camera_view(offset);
}

bool RenderWindow::is_rendering_done()
{
	HIPRTRenderSettings& render_settings = m_renderer->get_render_settings();

	bool rendering_done = false;

	// No more active pixels (in the case of adaptive sampling for example)
	rendering_done |= !m_renderer->get_status_buffer_values().one_ray_active;

	// All pixels have converged to the noise threshold given
	float proportion_converged;
	proportion_converged = m_renderer->get_status_buffer_values().pixel_converged_count /
						   static_cast<float>(m_renderer->m_render_resolution.x * m_renderer->m_render_resolution.y);
	proportion_converged *= 100.0f; // To percentage as used in the ImGui interface

	// We're allowed to stop the render after the given proportion of pixel of the image converged if we're actually
	// using the pixel stop noise threshold feature (enabled + threshold > 0.0f) or if we're using the
	// stop noise threshold but only for the proportion stopping condition (we're not using the threshold of the pixel
	// stop noise threshold feature) --> (enabled & adaptive sampling enabled)
	bool use_proportion_stopping_condition = (render_settings.stop_pixel_noise_threshold > 0.0f && render_settings.use_pixel_stop_noise_threshold) ||
											 (render_settings.use_pixel_stop_noise_threshold && render_settings.adaptive_sampling_enabled());
	bool minimum_sample_count_reached =
		render_settings.sample_number >= m_application_settings->pixel_stop_noise_threshold_min_sample_count || render_settings.adaptive_sampling_enabled();
	rendering_done |=
		proportion_converged > render_settings.stop_pixel_percentage_converged && use_proportion_stopping_condition && minimum_sample_count_reached;

	// Max sample count
	rendering_done |= (m_application_settings->max_sample_count != 0 && render_settings.sample_number + 1 > m_application_settings->max_sample_count);

	// Max render time
	float render_time_ms = m_application_state->current_render_time_ms / 1000.0f;
	rendering_done |= (m_application_settings->max_render_time != 0.0f && render_time_ms >= m_application_settings->max_render_time);

	// If we are at 0 samples, this means that the render got resetted and so
	// the render is not done
	rendering_done &= render_settings.sample_number > 0;

	if (rendering_done)
		set_ImGui_status_text("Finished!");
	else
	{
		if (m_imgui_renderer->get_status_text() == "Finished!" || m_imgui_renderer->get_status_text() == "")
			clear_ImGui_status_text();
	}

	return rendering_done;
}

bool RenderWindow::needs_viewport_refresh()
{
	// Update every X seconds
	bool enough_time_has_passed = get_time_ms_before_viewport_refresh() <= 0.0f;
	// The render was reset and one frame has been rendered
	bool render_was_reset = m_application_state->frame_number == 1;
	// We always need to update the viewport if real-time rendering
	bool realtime_rendering = !m_renderer->get_render_settings().accumulate;
	bool force_refresh		= m_application_state->force_viewport_refresh;
	bool denoiser_enabled	= m_application_settings->enable_denoising;
	bool denoise_each_frame = !m_application_settings->denoise_when_rendering_done && !m_application_settings->denoise_only_on_viewport_refresh;

	bool needs_refresh = enough_time_has_passed || realtime_rendering || render_was_reset || force_refresh || (denoiser_enabled && denoise_each_frame);
	if (!needs_refresh)
		return false;

	if (m_renderer->gmon_used())
	{
		// With GMoN however, we want to recompute the GMoN framebuffer with the new samples accumulated so far
		// before refreshing the viewport

		if (!needs_refresh)
			// No need to run GMoN
			return false;

		if (m_renderer->get_gmon_render_pass()->recomputation_completed())
			// We requested a GMoN recomputation before and it is actually complete, we're ready to display
			return true;
		else
		{
			// So if we need a refresh, we're going to request a GMoN computation first
			m_renderer->get_gmon_render_pass()->request_recomputation();

			return false;
		}
	}
	else
		// Not using GMoN
		return needs_refresh;
}

float RenderWindow::get_viewport_refresh_delay_ms()
{
	if (m_application_state->current_render_time_ms < 1000.0f)
		// Always update if less than a second of render time
		return 0.0f;
	else if (m_application_state->current_render_time_ms > 1000.0f && m_application_state->current_render_time_ms < 5000.0f)
		// 1s update in between 1s and 5s of total render time
		return 1000.0f;
	else
		// Update every 5s otherwise
		return 5000.0f;
}

float RenderWindow::get_time_ms_before_viewport_refresh()
{
	float time_since_last_refresh =
		(glfwGetTimerValue() - m_application_state->last_viewport_refresh_timestamp) / static_cast<float>(glfwGetTimerFrequency()) * 1000.0f;
	return get_viewport_refresh_delay_ms() - time_since_last_refresh;
}

void RenderWindow::reset_render()
{
	m_application_state->m_render_resetted_with_imgui_item_held = ImGui::IsAnyItemActive();

	m_application_settings->last_denoised_sample_count = -1;

	m_application_state->current_render_time_ms = 0.0f;
	m_application_state->render_dirty			= false;
	m_application_state->frame_number			= 0;

	m_renderer->reset(is_interacting() || m_application_state->interacting_last_frame);
}

void RenderWindow::set_render_dirty(bool render_dirty)
{
	m_application_state->render_dirty = render_dirty;
}

void RenderWindow::set_force_viewport_refresh(bool force_viewport_refresh)
{
	m_application_state->force_viewport_refresh = force_viewport_refresh;
}

void RenderWindow::set_ImGui_status_text(const std::string& status_text)
{
	if (status_text == "")
		// Do not call RenderWindow::set_ImGui_status_text with an empty text.
		//
		// To clear the status text, call clear_status_text()
		Debug::debugbreak();
	m_imgui_renderer->set_status_text(status_text);
}

void RenderWindow::clear_ImGui_status_text()
{
	set_ImGui_status_text("Rendering...");
}

float& RenderWindow::get_current_render_time_ms()
{
	return m_application_state->current_render_time_ms;
}

float RenderWindow::get_samples_per_second()
{
	return m_application_state->samples_per_second;
}

float RenderWindow::compute_samples_per_second()
{
	float samples_per_frame = m_renderer->get_render_settings().do_render_low_resolution() ? 1.0f : m_renderer->get_render_settings().samples_per_frame;

	// Frame time divided by the number of samples per frame
	// 1 sample per frame assumed if rendering at low resolution
	if (m_application_state->last_GPU_submit_time > 0)
	{
		uint64_t current_time = glfwGetTimerValue();
		float difference_ms	  = (current_time - m_application_state->last_GPU_submit_time) / static_cast<float>(glfwGetTimerFrequency()) * 1000.0f;

		return 1000.0f / (difference_ms / samples_per_frame);
	}
	else
		return 0.0f;
}

float RenderWindow::compute_GPU_stall_duration()
{
	if (m_application_settings->GPU_stall_percentage > 0.0f)
	{
		float last_frame_time = m_renderer->get_last_frame_time();
		float stall_duration  = last_frame_time * (1.0f / (1.0f - m_application_settings->GPU_stall_percentage / 100.0f)) - last_frame_time;

		return stall_duration;
	}

	return 0.0f;
}

float RenderWindow::get_UI_delta_time()
{
	return m_application_state->last_CPU_frame_delta_time_ms;
}

std::shared_ptr<OpenImageDenoiser> RenderWindow::get_denoiser()
{
	return m_denoiser;
}

std::shared_ptr<GPURenderer> RenderWindow::get_renderer()
{
	return m_renderer;
}

std::shared_ptr<GPUBaker> RenderWindow::get_baker()
{
	return m_gpu_baker;
}

std::shared_ptr<PerformanceMetricsComputer> RenderWindow::get_performance_metrics()
{
	return m_perf_metrics;
}

std::shared_ptr<Screenshoter> RenderWindow::get_screenshoter()
{
	return m_screenshoter;
}

std::shared_ptr<ImGuiRenderer> RenderWindow::get_imgui_renderer()
{
	return m_imgui_renderer;
}

void RenderWindow::run(const std::string& output_filepath, int render_samples)
{
	HIPRTRenderSettings& render_settings = m_renderer->get_render_settings();
	bool batch_output					 = !output_filepath.empty();

	if (batch_output)
	{
		// Batch captures must contain exactly the requested number of noisy samples.
		m_application_settings->max_sample_count	  = render_samples;
		m_application_settings->auto_sample_per_frame = false;
		m_application_settings->enable_denoising	  = false;
		render_settings.samples_per_frame			  = 1;
	}

	uint64_t timer_frequency = glfwGetTimerFrequency();

	m_renderer->start_render_thread();

	while (!glfwWindowShouldClose(m_glfw_window))
	{
		uint64_t frame_start_time = glfwGetTimerValue();
		// Saving whether the renderer as finished its frame
		// at the beginning of this CPU frame.
		//
		// If yes, we will use this variable later to record the
		// whole CPU overhead of launching a new frame + updating the UI
		// (swapBuffers etc...)
		//
		// This is simply done by computing the delta time between
		// 'frame_start_time' and 'frame_stop_time'. And because the renderer
		// is done with its frame, a new GPU frame is going to be queued in between
		// this two timer points so our CPU overhead counter will also take into account
		// the time taken for launching a new frame so that's perfect
		bool frame_render_done = m_renderer->frame_render_done();

		glfwPollEvents();
		glClear(GL_COLOR_BUFFER_BIT);

		m_application_state->render_dirty |= is_interacting();
		m_application_state->render_dirty |= m_application_state->interacting_last_frame != is_interacting();

		bool held_this_frame = ImGui::IsAnyItemActive();
		static bool held_last_frame;

		m_application_state->render_dirty |= render_resetted_with_imgui_item_held() && held_last_frame != held_this_frame;

		held_last_frame = held_this_frame;

		render();
		m_display_view_system->display();
		m_imgui_renderer->draw_interface();

		if (batch_output && is_rendering_done())
		{
			m_screenshoter->write_to_png(output_filepath);
			glfwSetWindowShouldClose(m_glfw_window, GLFW_TRUE);
		}

		// Measuring the CPU overhead before 'glfwSwapBuffers' because we do not want
		// to count the VSync as CPU overhead
		uint64_t cpu_overhead_stop_time = glfwGetTimerValue();

		glfwSwapBuffers(m_glfw_window);

		float delta_time_ms								  = (glfwGetTimerValue() - frame_start_time) / static_cast<float>(timer_frequency) * 1000.0f;
		m_application_state->last_CPU_frame_delta_time_ms = delta_time_ms;
		m_application_state->last_viewport_refresh_timestamp += m_application_state->last_CPU_frame_delta_time_ms;

		if (!is_rendering_done())
			m_application_state->current_render_time_ms += delta_time_ms;

		if (frame_render_done)
		{
			float cpu_overhead_time = (cpu_overhead_stop_time - frame_start_time) / static_cast<float>(timer_frequency) * 1000.0f;
			m_perf_metrics->add_value(RenderWindow::PERF_METRICS_CPU_OVERHEAD_TIME_KEY, cpu_overhead_time);
			m_perf_metrics->add_value(GPURenderer::FULL_FRAME_TIME_WITH_CPU_KEY,
									  cpu_overhead_time + m_perf_metrics->get_current_value(GPURenderer::ALL_RENDER_PASSES_TIME_KEY));
		}

		m_keyboard_interactor.poll_keyboard_inputs();
	}
}

void RenderWindow::render()
{
	HIPRTRenderSettings& render_settings = m_renderer->get_render_settings();

	// Boolean local to this function to remember whether or not we need to upload
	// the frame result to OpenGL for displaying
	static bool buffer_upload_necessary = true;

	if (m_renderer->frame_render_done())
	{
		// ------------------------------------------------------------
		// Everything that is in there is synchronous with the renderer
		// ------------------------------------------------------------

		m_renderer->download_status_buffers();

		if (m_application_state->GPU_stall_duration_left > 0 && !is_rendering_done())
		{
			// If we're stalling the GPU.
			// We're whether or not the rendering is done because we don't need to
			// stall the GPU if the rendering is done

			if (m_application_state->GPU_stall_duration_left > 0.0f)
				// Updating the duration left to stall the GPU.
				m_application_state->GPU_stall_duration_left -= m_application_state->last_CPU_frame_delta_time_ms;
		}
		else if (!is_rendering_done() || m_application_state->render_dirty)
		{
			// To save resources, we're only going to update the viewport only so often because
			// it can be a bit expensive and for offline rendering, we don't need an update every
			// frame, we can afford to update only every few samples (or every few seconds) to save
			// resources
			bool needs_refresh = needs_viewport_refresh();
			if (needs_refresh)
			{
				// We can unmap the renderer's buffers so that OpenGL can use them for displaying
				m_renderer->unmap_buffers();

				// Update the display view system so that the display view is changed to the
				// one that we want to use (in the DisplayViewSystem's queue)
				m_display_view_system->update_selected_display_view();

				// Denoising to fill the buffers with denoised data (if denoising is enabled)
				denoise();

				DisplayViewType current_display_view_type = m_display_view_system->get_current_display_view_type();
				bool uses_device_display_post_process =
					current_display_view_type == DisplayViewType::DEFAULT || current_display_view_type == DisplayViewType::GMON_BLEND ||
					current_display_view_type == DisplayViewType::DENOISED_BLEND || current_display_view_type == DisplayViewType::DISPLAY_DENOISER_ALBEDO ||
					current_display_view_type == DisplayViewType::DISPLAY_DENOISER_NORMALS ||
					current_display_view_type == DisplayViewType::WHITE_FURNACE_THRESHOLD;
				if (uses_device_display_post_process)
					m_renderer->launch_display_post_process();

				// Upload the final device-side post-process result for the trivial OpenGL display program.
				m_display_view_system->upload_final_display_buffer();

				// We just displayed so let's reset the timer
				m_application_state->last_viewport_refresh_timestamp = glfwGetTimerValue();

				// We just refreshed so we're clearing the flag
				m_application_state->force_viewport_refresh = false;
			}

			// We got a frame rendered --> We can compute the samples per second
			m_application_state->samples_per_second = compute_samples_per_second();

			// Adding the time for *one* sample to the performance metrics counter
			if (!m_renderer->was_last_frame_low_resolution() && m_application_state->samples_per_second > 0.0f)
				m_renderer->update_perf_metrics(m_perf_metrics);

			render_settings.wants_render_low_resolution = is_interacting();
			bool samples_per_frame_auto_mode			= m_application_settings->auto_sample_per_frame;
			bool current_or_last_frame_low_res			= render_settings.do_render_low_resolution() || m_renderer->was_last_frame_low_resolution();
			if (samples_per_frame_auto_mode && current_or_last_frame_low_res && render_settings.accumulate)
				// Only one sample when low resolution rendering.
				//
				// Also, we only want to apply this if we're accumulating. If we're not accumulating,
				// (so we have the renderer in "interactive mode") we may want more than 1 sample per frame
				// to experiment
				render_settings.samples_per_frame = 1;
			else if (m_application_settings->auto_sample_per_frame)
				// Otherwise and if the user is using auto samples per frame, we're going to compute
				// the appropriate number of samples per frame to use such that the GPU renders a frame
				// "exactly" as fast as the 'm_application_settings->target_GPU_framerate'
				//
				// This is to keep the GPU busy and improve rendering performance
				render_settings.samples_per_frame =
					std::min(std::max(1, static_cast<int>(m_application_state->samples_per_second / m_application_settings->target_GPU_framerate)), 65536);

			if (m_application_state->render_dirty)
				reset_render();

			m_application_state->GPU_stall_duration_left = compute_GPU_stall_duration();
			m_application_state->interacting_last_frame	 = is_interacting();

			// Queuing a new frame for the GPU to render
			uint64_t current_timestamp = glfwGetTimerValue();
			float delta_time_gpu = (current_timestamp - m_application_state->last_GPU_submit_time) / static_cast<float>(glfwGetTimerFrequency()) * 1000.0f;

			m_application_state->frame_number++;
			m_application_state->last_GPU_submit_time = current_timestamp;

			m_renderer->render(delta_time_gpu, this);

			m_application_state->m_render_resetted_with_imgui_item_held_last_frame = m_application_state->m_render_resetted_with_imgui_item_held;

			buffer_upload_necessary = true;
		}
		else // The rendering is done
		{
			bool display_view_changed = m_display_view_system->update_selected_display_view();
			buffer_upload_necessary |= display_view_changed;

			if (m_application_settings->enable_denoising)
			{
				// We may still want to denoise on the final frame
				if (denoise())
					buffer_upload_necessary = true;
			}

			DisplayViewType current_display_view_type = m_display_view_system->get_current_display_view_type();
			bool uses_device_display_post_process =
				current_display_view_type == DisplayViewType::DEFAULT || current_display_view_type == DisplayViewType::GMON_BLEND ||
				current_display_view_type == DisplayViewType::DENOISED_BLEND || current_display_view_type == DisplayViewType::DISPLAY_DENOISER_ALBEDO ||
				current_display_view_type == DisplayViewType::DISPLAY_DENOISER_NORMALS || current_display_view_type == DisplayViewType::WHITE_FURNACE_THRESHOLD;
			bool display_post_process_refresh_needed = display_view_changed || m_application_state->force_viewport_refresh;
			if (uses_device_display_post_process && display_post_process_refresh_needed)
			{
				m_renderer->launch_display_post_process();
				buffer_upload_necessary = true;
			}

			if (buffer_upload_necessary)
			{
				// Re-uploading only if necessary
				m_display_view_system->upload_final_display_buffer();

				buffer_upload_necessary = false;
			}

			if (display_post_process_refresh_needed)
				m_application_state->force_viewport_refresh = false;

			RendererAnimationState& renderer_animation_state = m_renderer->get_animation_state();
			if (renderer_animation_state.is_rendering_frame_sequence &&
				renderer_animation_state.frames_rendered_so_far < renderer_animation_state.number_of_animation_frames)
			{
				// If we're rendering an animation and the frame just converged
				renderer_animation_state.ensure_output_folder_exists();
				m_screenshoter->write_to_png(renderer_animation_state.get_frame_filepath());
				// Indicating that the animations can step forward since we're done
				// with this frame
				renderer_animation_state.frames_rendered_so_far++;
				if (renderer_animation_state.frames_rendered_so_far == renderer_animation_state.number_of_animation_frames)
					// We just rendered the last frame, deactivating rendering frame sequence state
					renderer_animation_state.is_rendering_frame_sequence = false;
				else
				{
					// Not the last frame
					renderer_animation_state.can_step_animation = true;

					set_render_dirty(true);
				}
			}

			// Sleeping so that we don't burn the CPU and GPU with the UI drawing
			std::this_thread::sleep_for(std::chrono::milliseconds(3));
		}
	}
}

bool RenderWindow::denoise()
{
	HIPRTRenderSettings& render_settings = m_renderer->get_render_settings();
	DisplaySettings& display_settings	 = get_display_settings();

	display_settings.blend_override = -1.0f;

	if (m_application_settings->enable_denoising)
	{
		// Evaluating all the conditions for whether or not we want to denoise
		// the current color framebuffer and whether or not we want to display
		// the denoised framebuffer to the viewport (we may want NOT to display
		// the denoised framebuffer if we're only denoising when the render is done
		// but the render isn't done yet. That's just one example)

		// ---- Utility variables ----
		// Do we want to denoise only when reaching the rendering is done?
		bool denoise_when_done = m_application_settings->denoise_when_rendering_done;
		// Is the rendering done?
		bool rendering_done = is_rendering_done();
		// Whether or not we've already denoise the framebuffer after the rendering is done.
		// This is to avoid denoising again and again the framebuffer when the rendering is done (because that would just be using the machine for nothing)
		bool final_frame_denoised_already = !m_application_settings->denoiser_settings_changed && rendering_done &&
											m_application_settings->last_denoised_sample_count == render_settings.sample_number;

		// ---- Conditions for denoising / displaying noisy ----
		// - Is the rendering done
		// - And we only want to denoise when the rendering is done
		// - And we haven't alraedy denoised the final frame
		bool denoise_rendering_done = rendering_done && denoise_when_done;
		// Have we rendered enough samples since last time we denoised that we need to denoise again?
		bool sample_skip_threshold_reached =
			!denoise_when_done &&
			(render_settings.sample_number - std::max(0, m_application_settings->last_denoised_sample_count) >= m_application_settings->denoiser_sample_skip);
		// We're also going to denoise if we changed the denoiser settings
		// (because we need to denoise to reflect the new settings)
		bool denoiser_settings_changed = m_application_settings->denoiser_settings_changed;

		bool need_denoising = false;
		bool display_noisy	= false;

		// Denoise if:
		//	- The render is done and we're denoising when the render
		//	- We have rendered enough samples since the last denoise step that we need to denoise again
		//	- We're not denoising if we're interacting (moving the camera)
		need_denoising |= denoise_rendering_done;
		need_denoising |= sample_skip_threshold_reached;
		need_denoising |= denoiser_settings_changed;
		need_denoising &= !is_interacting();
		need_denoising &= !final_frame_denoised_already;

		// Display the noisy framebuffer if:
		//	- We only denoise when the rendering is done but it isn't done yet
		//	- We want to denoise every m_application_settings->denoiser_sample_skip samples
		//		but we haven't even reached that number yet. We're displaying the noisy framebuffer in the meantime
		//	- We're moving the camera
		display_noisy |= !rendering_done && denoise_when_done;
		display_noisy |= !sample_skip_threshold_reached && m_application_settings->last_denoised_sample_count == -1 && !rendering_done;
		display_noisy |= is_interacting();

		if (need_denoising)
		{
			float denoise_duration = 0.0f;
			if (m_application_settings->denoiser_use_interop_buffers)
				denoise_duration = denoise_interop_buffers();
			else
				denoise_duration = denoise_no_interop_buffers();

			m_application_settings->last_denoised_duration	   = denoise_duration;
			m_application_settings->last_denoised_sample_count = render_settings.sample_number;
		}

		if (display_noisy)
			// We need to display the noisy framebuffer so we're forcing the blending factor to 0.0f to only
			// choose the first view out of the two that are going to be blend (and the first view is the noisy view)
			display_settings.blend_override = 0.0f;

		m_application_settings->denoiser_settings_changed = false;

		return need_denoising && !display_noisy;
	}

	return false;
}

float RenderWindow::denoise_interop_buffers()
{
	std::shared_ptr<OpenGLInteropBuffer<float3_t>> normals_buffer	= nullptr;
	std::shared_ptr<OpenGLInteropBuffer<ColorRGB32F>> albedo_buffer = nullptr;

	if (m_application_settings->denoiser_use_normals)
		normals_buffer = m_renderer->get_denoiser_normals_AOV_interop_buffer();

	if (m_application_settings->denoiser_use_albedo)
		albedo_buffer = m_renderer->get_denoiser_albedo_AOV_interop_buffer();

	auto start = std::chrono::high_resolution_clock::now();
	m_denoiser->denoise(m_renderer->get_color_interop_framebuffer(), normals_buffer, albedo_buffer);
	auto stop = std::chrono::high_resolution_clock::now();

	m_denoiser->copy_denoised_data_to_buffer(m_renderer->get_denoised_interop_framebuffer());

	return std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
}

float RenderWindow::denoise_no_interop_buffers()
{
	std::shared_ptr<OrochiBuffer<float3_t>> normals_buffer	 = nullptr;
	std::shared_ptr<OrochiBuffer<ColorRGB32F>> albedo_buffer = nullptr;

	if (m_application_settings->denoiser_use_normals)
		normals_buffer = m_renderer->get_denoiser_normals_AOV_no_interop_buffer();

	if (m_application_settings->denoiser_use_albedo)
		albedo_buffer = m_renderer->get_denoiser_albedo_AOV_no_interop_buffer();

	auto start = std::chrono::high_resolution_clock::now();
	m_denoiser->denoise(m_renderer->get_color_interop_framebuffer(), normals_buffer, albedo_buffer);
	auto stop = std::chrono::high_resolution_clock::now();

	m_denoiser->copy_denoised_data_to_buffer(m_renderer->get_denoised_interop_framebuffer());

	return std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
}
