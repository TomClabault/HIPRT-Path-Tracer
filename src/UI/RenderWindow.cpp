/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Compiler/GPUKernelCompiler.h"
#include "Scene/SceneParser.h"
#include "Threads/ThreadFunctions.h"
#include "Threads/ThreadManager.h"
#include "tracy/TracyOpenGL.hpp"
#include "UI/RenderWindow.h"
#include "UI/Interaction/LinuxRenderWindowMouseInteractor.h"
#include "UI/Interaction/WindowsRenderWindowMouseInteractor.h"
#include "Utils/Utils.h"

#include <functional>
#include <iostream>

#include "stb_image_write.h"

// - try simplifying the material to just a diffuse component to see if that helps memory accesses --> 8/10%
// - try removing everything about nested dielectrics to see the register/spilling usage and performance --> ~1/2%


// GPUKernelCompiler for waiting on threads currently reading files on disk
extern GPUKernelCompiler g_gpu_kernel_compiler;
extern ImGuiLogger g_imgui_logger;

// TODO still some config with envmap sampling that doesn't quite match the reference when playing with all the ReGIR / ReSTIR DI settings


// TODO to mix microfacet regularization & BSDF MIS RAY reuse, we can check if we regularized hard or not. If the regularization roughness difference is large, let's not reuse the ray as this may roughen glossy objects. Otherwise, we can reuse
// - Test ReSTIR GI with diffuse transmission
// - We don't have to store the ReSTIR **samples** in the spatial pass. We can just store a pixel index and then on the next pass, when we need the sample, we can use that pixel index to go fetch the sample at the right pixel
// - distance rejection heuristic for GI reconnection
// - Alpah tests darkening ReSTIR DI
// - ReSTIR DI + the-white-room.gltf + CPU (opti on) + no debug + no envmap ---> denormalized check triggered

// TODO ReSTIR
// - We shouldn't shoot a shadow ray in the light evaluation if the BSDF sample was chosen because this already has visibility
// - Can we do something for restir that has a hash grid for the first hits of the rays and then for spatial reuse, each pixel looks up its cell and reuse paths from the same cell (and thus same geometry if we include the normals in the hash grid). This would basically be a more accurate version of the directional spatial reuse
//		- One issue that we're going to have is: for a given pixel, we can compute it hash cell but then how do we know which other reservoirs (neighbors) are in the same hash cell?
//			- Fix that by: counting how different hash cell the primary hits create
//			- Create one counter per hash cell
//			- Count how many pixels fall in a given hash cell
//			4. Index the hash cell from 0 to N-1 where N is the number of hash cells
//			- Then we can have a pass that assigns each pixel index to a hash cell:
//				- For each pixel, find its hash cell. From the hash cell index of step 4., we know where in the fullscreen-wide buffer we need to write the pixel index by using the prefix sum of the hash cell counters up til the current hash cell index
//			- Once that's done, we know for a given pixel how many valid neighbors there are and what's their pixel indices
// 
// - For the spatial reuse buffer, we don't have to store a whole grid at all, we can just store the index of the cell the reservoir reused from --> massive VRAM saves
// - Using the indirect index for the spatial output buffer, can we double buffer the initial candidates grid and run the spatial reuse of ReGIR async of the path tracing too?
// - There is bias in ReSTIR DI
// - Greedy spatial reuse to retry neighbors if we didn't get a good one
//			For the greedy neighbor search of restir spatial reuse, maybe reduce progressively the radius ?
// - memory coalescing aware spatial reuse pattern --> per warp / per half warp to reduce correlation artifacts?
// - can we maybe stop ReSTIR GI from resampling specular lobe samples? Since it's bound to fail anwyays. And do not resample on glass
// - See how many pixels of ReSTIR GI end up with the initial candidate as the final sample --> we can reuse NEE at the first hit for those samples in the shading pass instead of recomputing NEE
// - BSDF MIS Reuse for ReSTIR DI
// - Force albedo to white for spatial reuse? Because what's interesting to reuse is the shape of the BRDF and the incident radiance. Resampling from a black diffuse is still interesting. The albedo doesn't matter
// - Have a look at compute usage with the profiler with only a camera ray kernel and more and more of the code to see what's dropping the compute usage 
// - If it is the canonical sample that was resampled in ReSTIR GI, recomputing direct lighting at the sample point isn't needed and could be stored in the reservoir?

// TODO ReGIR
// - Can we group light triangles by their meshes and pre-compute a CDF per each grid cell for which meshes are the best one for that grid cell.
//		- We would then use that CDF during the grid fill to resample a good mesh and then resample a good triangle in that mesh
//		- We'er going to have a similar "CDF per cell" situation as "Cache Points" from Disney some maybe there are going to be some ideas to pick from their CDF blending / visibility integration etc...
// - Store target function in reservoir to avoid recomputing it during pairwise MIS shading resampling?
// - Stochastic light culling harada et al to improve base candidate sampling
// - Pixel deinterleaving for reducing correlations in light presampling ? Segovia 2006
// - Can we have a lightweight light rejection (russian roulette) method in the grid fill? So even if we have 32 candidates per grid fill cell, if a light is evidently too far away to contribute, we can reject it and not count that as a try from our 32 tries. The rejection test needs to be lightweight such that it is significantly less expensive than doing a full candidate
//		- To make the rejection lightweight, can we have a luminance of emission baked into our materials such that we can simply check the luminance of the light (one float fetch) instead of the full RGB color which would be 3 float fetchtes. Maybe that would be faster
// - Increased the number of shading retries?
// - NEE++ compaction: we only need uchar per value, not uint
// - Can we shade only the 4 non canonical neighbors + the final reservoir instead of shading everyone?
//		-  For the MIS weights, we can use the unnormalized target functions for everyone and it should be fine?
// - For interacting with ReGIR, we can probably just shade with non canonical candidates and that's it. It will be biased but at least it won't be disgusting because of the lack of pre integration information
// - Can we shade multiple reservoirs without shooting shadow rays by using NEE++ to make sure that the reservoir isn't shadowed? This may be biased but maybe not too bad?
// - Can we have a biased NEE++ where we clamp the normalization factor to avoid fireflies?
// - Can we evaluate the ratio between the UCW and the final contribution? If the ratio is higher than a threshold then that's an outlier / Firefly and we may want to skip it attenuate it
// - Can we do many many more samples per each reservoir during the pre integration pass (and thus have less reservoirs per cell) to improve the quality of the integral estimate with less reservoirs and less integration iterations?
// - Spatial reuse seems to introduce quite a bit of correlations so we would be better off improving the base sampling to not have to rely on spatial reuse for good samples quality
// - NEE++ maximum load factor to avoid the hash grid being totally filled and performance dying because of that
// - Can we randomize the hash of grid cells to avoid correlations? Basically subdivide each grid cell into 2/3/4/... grid cells and randomly assign the space of the main grid cell to either 1/2/3/... of the sub such that correlation aretifacts are basically randomized and do not look bad
// - Can we compute the "gradient" of cell occupancy of the grid to adjust the factor by which we resize the grid every time? To avoid overshooting too much and having a resized grid that is too large
// - Can we just use the 32 reservoirs for shading as the input to the pre integration process? Is that enough for an accurate integral estimate?
// - Maybe not having the spatial reuse in the pre integration is ok still for normalization factor
// - No need to read random reservoirs in the pre integration kernel, we can just read the reservoirs one by one of each grid cell and integrate them all. 
//		- Opens up possibilities for coalescing the reads of the reservoirs in the pre integration kernel
// - Super large resolution on surfaces that do not allow light sampling for the hash grid since we do not need ReGIR here
// - We need a special path for ReGIR, hard to use as a light sampling plug in, lots of opti to do with a special path
// - Variable jitter radius basezd on cell size
// - Include normal in hash grid for low roughness surfaces to have better BRDF sampling precision
// - Decoupled shading and reuse ReGIR: add visibility rays during the shading so that we have visiblity resampling which is very good and on top of that, we can totally shade the reservoir because the visibility has been computed so the rest of the shading isn't super expensive: maybe use NEE++ in there to reduce shadow rays? Or the visibility caching thing that is biased?
// - Can we maybe add BRDF samples in the grid fill for rough BRDFs? This will enable perfect MIS for diffuse BRDFs which should be good for the bistro many light with the light close for example. This could also be enough for rough-ish specular BRDFs
//		- We can probably trace the BRDF rays in a light-only BVH here and then if an intersection point is found, use NEE++ visibility estimation there
//		- Maybe have some form of roughness threshold when using ReGIR with MIS to use MIS only on specular surfaces where the grid fill BRDF rays didn't help
// - Only need 1 bit per cell here for 'grid cells alive': whether or not a given grid cell is alive
// - Quantize ahsh grid cell data .sum_points: we don't need the precision since this is just an average for getting an approximate center of cell
// - Light to light grid cells should be cached in the same hash cell entry
// - Reintroduce temporal reuse but maybe with a small M-cap, should be worth it on difficult scenes, the many lights bistro for example
// - Limit the grid cell life length of NEE++ if it hasn't been hit in a long time
// - Limit the grid cell life length of ReGIR if it hasn't been hit in a long time
// - Multiple spatial reuse passes
// - We can deallocate the emissive triangle index of the ReGIR reservoir if not using ReSTIR DI
// - Should we have something to limit the life length of an NEE++ grid cell? So that we can remove cells unused and keep the grid size in check
// - Trry to disable canonical and see if it converges quicker
//		- It does -----> We need to find some better MIS weights for the canonical sample
//		- Try to downweigjt canonical MIS weight instead of 1 / M
// - Interrupt target function evaluation in ReGIR if the cosine term drops to zero such that we don't fill the NEE hash grid if the light is back facing for example
// - Lambertian BRDF goes through lampshade in white room but principled BSDF doesn't
// - Can we keep the grid of reservoirs from the last frame to pick them during shading to reduce correlations? Only memory cost but it's ok
//		- Maybe only that for primary hit reservoirs because those are the only one to be correlated hard?
// - Have a variable radius when picking reservoirs for shading 
// - Issue with microfacet regularization x ReGIR?
// - Scalarization of hash grid fill because we know that consecutive threads are in the same cell
// - Scalarization of the hash grid fetches for the camera rays?
// - We can optimize the grid cell aliv ecounter atomic increment by incrementing by the number of threads in the wavefront instead of 1 per thread
// - Deduplicate hash grid cell idnex calculations in fetch reservoirs functions mainly for performance reasons
// - To profile the hash grid, may be useful to, for example, store everything from the camera rays pass into some buffers and then run a separate hash grid cell data fill kernel just to be able to profile that kernel in isolation
// - For the spatial reuse output grid buffer, we don't have to store the rservoirs, we can just store the indices of the cell which we resample from so let's save some VRAM there
// - Can we store just the light index per each regir sample? And reconstruct, the normal and everything from that? Maybe that's not going to be much more expensive that having to read everything from the Regir sample but this would save a lot of memory
// - Directional spatial reuse to directly hit the right neighbors instead of having to retry multiple times (one memory access for each retry)
// - Do we have bad divergence when ReGIR falls back to power sampling? Maybe we could retry more and more ReGIR until we find a reservoir to avoid the divergence
// - If we want initial visibility in ReGIR, we're going to have to check whether the center of the cell is in an object or not because otherwise, all the samples for that cell are going to be occluded and that's going to be biased if a surface goes through that cell
// - Use some shjortcut in the BSDF in the target function during shading: rough material only use a constant BSDF, nothing more
// - When computing the MIS weights by counting the neighbors, we actually don't need the full target function with the emission and everything, we just need the cosine term and shadow ray probably
// - De-duplicate BSDF computations during shading: we evaluate the BRDF during the reservoir resampling and again during the light sampling
//		May be exclusive with the BSDF simplifications that can be done in the target function because then we wouldn't be evaluating the proper full BSDF in the target function
// - Can we have some kind of visibility percentage grid that we can use during the resampling to help with visibility noise? 
//		- We would have a voxel grid on top of the ReGIR grid. 
//		- That grid would contain as many floats per cell as there are reservoirs per cell in ReGIR
//		- Each one of these floats would contain a percentage of visibility for the corresponding reservoir index of the cell
//		- The visibility percentage would be computed by averaging the successful visibility rays traced during shading
//			- The issue is that the reservoirs aren't persistent so any data accumulated will be discarded at the next frame when
//			- the grid is rebuilt
//		
//			- We would need a prepass at lower resolution, same as for radiance caching?
//			- Maybe we can keep the grids of past frames to help with that?
// - For the visibility reuse of ReGIR, maybe we can just trace from the center of the cell and if at shading time, the reservoir is 0, we know that this must be because the reservoir is occluded for that sample so we can just draw a canonical candidate instead there
//		- Always tracing from the center of the cell may be always broken depending on the geometry of the scene so maybe we want to trace from the center of the cell as a default but as path tracing progresses, we want to save one point on the surface of geometry in that cell and use that point to trace shadow rays from onwards, that way we're always tracing from a valid surface in the grid cell
//		- And with that new "representative point" for each cell, we can also have the normal to evaluate the cosine term
// - For performance, at shading time when resampling the reservoirs, there may be only a few materials that benefit from the BSDF in the resampling target function because lambertian doesn't care, mirrors don't care, specular don't care, really it's only materials at like 0.3 roughness ish
// - Looking at the average contribution of cells seems to be giving some good metric on the performance of the sampling per cell no? What can we do with that info? Adaptive sampling somehow?
//		Maybe we can adaptively adapt the number of samples per grid cell during grid fill with that
// - Cull lights that have too low a contribution during grid fill. Maybe some power function or something to keep things unbiased, not just plain reject
// - NEE++ mix up to help with visibility sampling?
// - The spatial reuse seems giga compute bound, try to optimize the cell compute functions in Settings.h
// - Is the grid fill bottleneck by random light sampling? Try on the class white room to see if perf improves
//		A little bit yeah. Maybe we can do something with light presampling per cell
// - Shared mem ray tracing helps a ton for ReGIR grid fill & spatial reuse ----> maybe have them in a separate kernel to be able to use max shared mem without destroying the L1 for the rest of the kernels?
// - Can we add the canonical sample at the end of the spatial pass instead of in the shading pass?
// - The idea to fix the bad ReGIR target function that may prioritze occluded samples is to use NEE with a visibility weight
// - Maybe we can just swap the buffers for ReGIR staging buffers instead of copying
// - Can we use ReSTIR DI and fill the ReGIR grid with the ReSTIR DI samples? ---> Doesn't work at later bounces though
// - Can we start another grid fill in parallel of the mega kernel after the spatial reuse such that we overlap some work and don't have to do the grid fill at the next frame
//		- We can even decouple the spatial reuse with the visibility pass of it and launch the grid fill during the visibility pass of teh spatial reuse
// - Introduce envmap sampling into ReGIR to avoid having to integrate the envmap in a separate domain: big perf boost
// - When shading, maybe pick random reservoirs from a single neighboring cell to reduce shadow rays count but do that on a per warp basis to reduce the size of artifacts (which would be grid cell size otherwise)
// - Is there something to do with a wavefront architecture when tracing shadow rays at the end of the spatial reuse or something? Do we want maybe to dispatch kernels together for tracing from a given cell?
// - Maybe we can do some double buffering on the grid to be able to spatially reuse WHILE generating the gri fill: we would run the grid fill and fill grid 1 while spatially reusing on grid 2 which was filled last frame
//		The hope being that the computations can overlap a bit with the ray traversal
//		We can just test that tehroretically and see if that helps performance at all
// - Can we do something with the time per grid cell ray? To try and reduce this "long tails" effect
//		- Maybe what we can do here is compact the hard threads together so that we are able to launch all the light rays together and avoid divergence between light and heavy rays
// - Gather some information of how many light samples are rejected because of visibility to get a feel for how much can be gained with NEE++
//		- Also incorporate back facing lights info

// TODO restir gi render pass inheriting from megakernel render pass seems to compile mega kernel even though we don't need it
// - ReSTIR redundant render_data.g_buffer.primary_hit_position[pixel_index] load for both shading_point and view_direction
// - ReSTIR only load the rest of the reservoir if its UCW isn't 0


// TODOs  performance improvements branch:
// - Remove HIPRT INLINE everywhere
// - Vertex cache optimization buffer arrangement for better triangle pairing and better tracing performance?
// - Thread is swizzling (reorder ray invocations) https://github.com/BoyBaykiller/IDKEngine/blob/95a15c1db02f11bd2f47bb81bcfccf0943d3e703/IDKEngine/Resource/Shaders/PathTracing/FirstHit/compute.glsl#L206
// - Option for terminating rays on emissive hits? --> this is going to be biased but may help performance
// - Have a look at reweghing fireflies for Monte Carlo instead of Gmon so we can remove fireflies unbiasedly without the darkening
// - There seems be some scratch store on the RNG state? Try to offload that to shared mem?
//		- Do that after wavefront because wavefront may solve the issue
// - also reuse BSDF mis ray of envmap MIS
// - We do not need the nested dielecttrics stack management in the camera rays kernel
// - In the material packing, pack major material properties together: coat, metallic, specular_transmission, diffuse_transmission, ... so that we can, in a single memory access, determine whether or not we need to read the rest of the coat, specular transmission ,...
// - If hitting the same material as before, don't load the material from VRAM as it's exactly the same? (only works for non-textured materials)
// - When doing MIS, if we sampled a BSDF sample on a delta distribution, we shouldn't bother sampling lights because we know that the BSDF sample is going to overweight everything else and the light sample is going to have a MIS weight of 0 anyways
// - MIS disabled after some number of bounces? not on glass though? MIS disabled after the ray throughput gets below some threshold?
// - texture compression
// - store full pointers to textures in materails instead of indirect indices? probably cheaper to have ibigger materials than to havbe to do that indirect fetch?
// - limit  number of bounces based on material type
// - use material SoA in GBuffer and only load what's necessary (i.e. not the thin film and all of that if the material isn't using thin-film, ...)
// - use the fact that some values are already computed in bsdf_sample to pass them to bsdf_eval in a big BSDFStateStructure or something to avoid recomputing
// - schlick fresnel in many places? instead of correct fresnel. switch in "performance settings"
// 
// ------------------- STILL RELEVANT WITH WAVEFRONT ? -------------------
// - if we don't have the ray volume state in the GBuffer anymore, we can remove the stack handlign in the trace ray function of the camera rays
// - merge camera rays and path tracer?
// - store Material in GBuffer only if using ReSTIR, otherwise, just reconstruct it in the path tracign kernel
// ------------------- STILL RELEVANT WITH WAVEFRONT ? -------------------
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
// - can we gain in performance by having the trace rays functions in completely separate passes so that we can have the maximum amount of L1 cache in the passes that now don't trace rays? (and use max amount of shared mem in the rays only passes)
// ------------------- DO AFTER WAVEFRONT -------------------


// TODO known bugs / incorrectness:
// - take transmission color into account when direct sampling a light source that is inside a volume: leave that for when implementing proper volumes?
// - denoiser AOVs not accounting for transmission correctly since Disney  BSDF
//	  - same with perfect reflection
// - threadmanager: what if we start a thread with a dependency A on a thread that itself has a dependency B? we're going to try join dependency A even if thread with dependency on B hasn't even started yet --> joining nothing --> immediate return --> should have waited for the dependency but hasn't
// - Thin-film interference energy conservation/preservation is broken with "strong BSDF energy conservation" --> too bright (with transmission at 1.0f), even with film thickness == 0.0f
// - When overriding the base color for example in the global material overrider, if we then uncheck the base color override to stop overriding the base color, it returns the material to its very default base color  (the one  read from the scene file) instead of  returning it to what the user may have modified up to that point
// - Probably some weirdness with how light sampling is handled while inside a dielectric: inside_surface_multiplier? cosine term < 0 check? there shouldn't be any of that basically, it should just be evaluating the BSDF
// - Emissive chminey texture broken in scandinavian-studio
// - For any material that is perfectly specular / perfectly transparent (the issue is most appearant with mirrors or IOR 1 glass), seeing the envmap through this object takes the envmap intensity scaling into account and so the envmap through the object is much brighter than the main background (when camera rays miss the scene and hit the envmap directly) without background envmap intensity scaling: https://mega.nz/file/x8I12Q6b#DJ2ZobBav9rwFdtvTX-CmgA1eFEgKprjXSvOg0My38o
// - White furnace mode not turning emissives off in the cornell_pbr with ReSTIR GI?

// TODO Features:
// - Variance aware MIS weights? https://cgg.mff.cuni.cz/~jaroslav/papers/2019-variance-aware-mis/2019-grittmann-variance-aware-mis-paper.pdf
// - RISLTC: https://data.ishaanshah.xyz/research/pdfs/risltc.pdf. Some explanations in there for projected solid angle and LTC sampling
// - Inciteful graph to explore (started with Practical product sampling warping NVIDIA): https://inciteful.xyz/p?ids%5B%5D=W4220995884&ids%5B%5D=W3179788358&ids%5B%5D=W4403641440&ids%5B%5D=W4390345185&ids%5B%5D=W4388994411&ids%5B%5D=W4200187284&ids%5B%5D=W2885975589&ids%5B%5D=W3183450244&ids%5B%5D=W1893031899&ids%5B%5D=W3036883119&ids%5B%5D=W3044759327&ids%5B%5D=W4240396283&ids%5B%5D=W3110265079&ids%5B%5D=W2073976119&ids%5B%5D=W2988541899&ids%5B%5D=W2885239691&ids%5B%5D=W2964425571&ids%5B%5D=W2030242873&ids%5B%5D=W3044185278
// - VisibilityCluster: Average Directional Visibility for Many-Light Rendering: https://ieeexplore.ieee.org/document/6464264
// - Practical product sampling warping NVIDIA, there's a shadertoy for that
// - Sample specular/diffuse lobe with the luminance of the diffuse lobe
// - Sample specular/diffuse by taking the thrioughput of the path into account?
// - Sample by evaluating the contribution of both samples and choosing proportional to the contribution:
//		- a next "clever way" would be to generate L with both diffuse and specular but using the sample random number, then compare their total "contributions" (whole specular+diffuse BRDF value divided by pdf of generator and multiplied by path prefix throughput), then depending on the luma of that you choose either the first or second sample.
//		- so you're making decisions about what branch you take posteriori not a-priori.
//
//		- This has a few drawbacks :
//		- you're computing the contribution twice (but you're not really doing double the work for generation because Low Discrepancy Sequences / random numbers->maximum divergence)
//		- if you need the PDF for MIS or RIS for a given L you need to do far more work, your sampling routines must be invertible
// 
//		- The latter part is annoying because many BRDF sampling routines don't require you to find the xi which produce a given L when you want to query pdf(L).
//		- However the issue is that for every 2D value of xi you have two values of L between which you've chosen based on the value of the whole BRDF, so to know the PDF of any of the L in the pair, you need to know exactly what the other L is.
//
//		This is why this method is not tractable / fun for more than 2 BRDFs.
// 
// - Vector valued monte carlo: https://suikasibyl.github.io/files/vvmc/paper.pdf
// - Reweighting path guiding: https://zhiminfan.work/paper/mi_reweight_preprint.pdf
// - Fixed balance heuristic: https://qingqin-hua.com/publication/2025-correct-balance/2025-correct-balance.pdf
// - Envmap with visibility sampling: https://static.chaos.com/documents/assets/000/000/377/original/adaptive_dome_abstract.pdf?1676455588
// - Faster PNG loading: https://github.com/richgel999/fpng
// - Need something blocking inn "start thread with dependency" so that the main thread is blocked until the other thread actually started. This should solve the issue where sometilmes the main threds just joins everyone but everyone hasn't even started yet
// - Can we have something like sharc but for light sampling? We store reservoirs in the hash table and resample everytime we read into the hash grid with some initial candidates?
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
// - Directional albedo sampling weights for the principled BSDF importance sampling. Also, can we do "perfect importance" sampling where we sample each relevant lobe, evaluate them (because we have to evaluate them anyways in eval()) and choose which one is sampled proportionally to its contribution or is it exactly the idea of sampling based on directional albedo?
// - Russian roulette improvements: http://wscg.zcu.cz/wscg2003/Papers_2003/C29.pdf
// - Some MIS weights ideas in: https://momentsingraphics.de/ToyRenderer4RayTracing.html in "Combining diffuse and specular"
// - Radiance caching for feeding russian roulette
// - Tokuyoshi (2023), Efficient Spatial Resampling Using the PDF Similarity
//		- Not for offline?
// - Some automatic metric to determine automatically what GMoN blend factor to use
// - software opacity micromaps
// - Add parameters to increase the strength of specular / coat darkening
// - sample BSDF diffuse lobe proba based on its luminance?
// - how to help with shaders combination compilation times?
//		RocFFT has some ideas for parallel compilation https://github.com/ROCm/rocFFT/blob/e9303acfb993de98b78358f3bf6fdd93f810f5fd/docs/design/runtime_compilation.rst#parallel-compilation
//		- wavefront path tracing should help
//		- Maybe have two sets of shaders:
//			- One that uses the #if for performance
//			- One that uses if() everywhere instead of #if for fast preview
//				- to accelerate compilation times: we can use if() everywhere in the code so that switching an option doesn't require a compilation but if we want, we can then apply the options currently selected and compiler everything for maximum performance. This can probably be done with a massive shader that has all the options using if() instead of #if ? Maybe some better alternative though?
//				----------- That's a good one too ^
// - next event estimation++? --> 2023 paper improvement with the octree
// - ideas of https://pbr-book.org/4ed/Light_Sources/Further_Reading for performance
// - envmap visibility cache? 
// - If GMoN is enabled, it would be cool to be able to denoise the GMoN blend between GMoN and the default framebuffer but currently the denoiser only denoises the full GMoN and nothing else
// - Exploiting Visibility Correlation in Direct Illumination
// - smarter shader cache (hints to avoid using all kernel options when compiling a kernel? We know that Camera ray doesn't care about direct lighting strategy for example)
// - for LTC sheen lobe, have the option to use either SGGX volumetric sheen or approximation precomputed LTC data
// - for volumes, we don't have to use the same phase function at each bounce, for artistic control of the "blur shape"
// - --help on the commandline
// - Normal mapping seems broken again, light rays going under the surface... p1 env light
// - performance/bias tradeoff by ignoring alpha tests (either for global rays or only shadow rays) after N bounce?
// - performance/bias tradeoff by ignoring direct lighting occlusion after N bounce? --> strong bias but maybe something to do by reducing the length of shadow rays instead of just hard-disabling occlusion
// - energy conserving Oren Nayar: https://mimosa-pudica.net/improved-oren-nayar.html#images
// - experiment with a feature that ignores really dark pixel in the variance estimation of the adaptive 
//		sampling because it seems that very dark areas in the image are always flagged as very 
//		noisy / very high variance and they take a very long time to converge (always red on the heatmap) 
//		even though they are very dark regions and we don't even noise in them. If our eyes can't see 
//		the noise, why bother? Same with very bright regions
// - Reuse miss BSDF ray on the last bounce to sample envmap with MIS
// - We're using an approximation of the clearcoated BSDF directional albedo for energy compensation right now. The approximation breaks down when what's below the coat is 0.0f roughness. We could potentially bake the directional albedo for a mirror-coated BSDF and interpolate between that mirror-coated LUT and the typical rough-coated BSDF LUT based on the roughness of what's below the coat. This mirror-coated LUT doesn't work very well if there's a smooth-dielectric-coated lambert below the coat so maybe we would need a third LUT for that case
// - For/switch paradigm for instruction cache misses? https://youtu.be/lxRgmZTEBHM?si=FcaEYqAMVO_QyfwX&t=3061 
//		- kind of need a good way to profile that to see the difference though
// - have a light BVH for intersecting light triangles only: useful when we want to know whether or not a direction could have be sampled by the light sampler: we don't need to intersect the whole scene BVH, just the light geometry, less expensive ------> we're going to need another shadow ray though because if we're intersecting solely against the light BVH we don't have the rest of the geometry of the scene to occluded the lights. So we're going to need a shadow ray in case we do hit a light in the light BVH to make sure that light isn't occluded ----> Maybe collect statistics on how many BSDF rays light sample miss lights: this can help see what's going to be the benefit of a light BVH because the drawback of a light BVH is going to be only if we hit a light because then we need another BVH traversal to check for occlusion
// - shadow terminator issue on sphere low smooth scene: [Taming the Shadow Terminator], Matt Jen-Yuan Chiang, https://github.com/aconty/aconty/blob/main/pdf/bump-terminator-nvidia2019.pdf
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
// - fixed point 18b RGB for envmap? 70% size reduction compared to full size. Can't use texture sampler though. Is not using a sampler ok performance-wise? --> it probably is since we're probably memory lantency bound, not memory bandwidth
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
// - Emissive textures sampling: how to sample an object that has an emissive texture? How to know which triangles of the mesh are covered by the emissive parts of the texture?
// - stream compaction / active thread compaction (ingo wald 2011)
// - sample regeneration
// - Spectral rendering / look at gemstone rendering because they quite a lot of interesting lighting effect to take into account (pleochroism, birefringent, dispersion, ...)
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
// - When modifying the emission of a material with the material editor, it should be reflected in the scene and allow the direct sampling of the geometry so the emissive triangles buffer should be updated
// - Ray differentials for texture mipampping (better bandwidth utilization since sampling potentially smaller texture --> fit better in cache)
// - Visualizing ray depth (only 1 frame otherwise it would flicker a lot [or choose the option to have it flicker] )
// - Visualizing pixel time with the clock() instruction. Pixel heatmap:
//		- https://developer.nvidia.com/blog/profiling-dxr-shaders-with-timer-instrumentation/
//		- https://github.com/libigl/libigl/issues/1388
//		- https://github.com/libigl/libigl/issues/1534
// - Visualizing russian roulette depth termination
// - Statistics on russian roulette efficiency
// - feature to disable ReSTIR after a certain percentage of convergence --> we don't want to pay the full price of resampling and everything only for a few difficult isolated pixels (especially true with adaptive sampling where neighbors don't get sampled --> no new samples added to their reservoir --> no need to resample)
// - Realistic Camera Model
// - Focus blur
// - Flakes BRDF (maybe look at OSPRay implementation for a reference ?)
// - ImGuizmo for moving objects in the scene
// - choose denoiser quality in imgui
// - try async buffer copy for the denoiser (maybe run a kernel to generate normals and another to generate albedo buffer before the path tracing kernel to be able to async copy while the path tracing kernel is running?)
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
void APIENTRY RenderWindow::gl_debug_output_callback(GLenum source,
	GLenum type,
	GLuint id,
	GLenum severity,
	GLsizei length,
	const GLchar* message,
	const void* userParam)
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
	case GL_DEBUG_SOURCE_API:             source_str = "Source: API"; break;
	case GL_DEBUG_SOURCE_WINDOW_SYSTEM:   source_str = "Source: Window System"; break;
	case GL_DEBUG_SOURCE_SHADER_COMPILER: source_str = "Source: Shader Compiler"; break;
	case GL_DEBUG_SOURCE_THIRD_PARTY:     source_str = "Source: Third Party"; break;
	case GL_DEBUG_SOURCE_APPLICATION:     source_str = "Source: Application"; break;
	case GL_DEBUG_SOURCE_OTHER:           source_str = "Source: Other"; break;
	}

	switch (type)
	{
	case GL_DEBUG_TYPE_ERROR:               type_str = "Type: Error"; break;
	case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: type_str = "Type: Deprecated Behaviour"; break;
	case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:  type_str = "Type: Undefined Behaviour"; break;
	case GL_DEBUG_TYPE_PORTABILITY:         type_str = "Type: Portability"; break;
	case GL_DEBUG_TYPE_PERFORMANCE:         type_str = "Type: Performance"; break;
	case GL_DEBUG_TYPE_MARKER:              type_str = "Type: Marker"; break;
	case GL_DEBUG_TYPE_PUSH_GROUP:          type_str = "Type: Push Group"; break;
	case GL_DEBUG_TYPE_POP_GROUP:           type_str = "Type: Pop Group"; break;
	case GL_DEBUG_TYPE_OTHER:               type_str = "Type: Other"; break;
	}

	switch (severity)
	{
	case GL_DEBUG_SEVERITY_HIGH:         severity_str = "Severity: high"; break;
	case GL_DEBUG_SEVERITY_MEDIUM:       severity_str = "Severity: medium"; break;
	case GL_DEBUG_SEVERITY_LOW:          severity_str = "Severity: low"; break;
	case GL_DEBUG_SEVERITY_NOTIFICATION: severity_str = "Severity: notification"; break;
	}

	g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, 
		"---------------\n"
		"Debug message (%d): %s\n"
		"%s\n%s\n%s\n\n", id, message, source_str.c_str(), type_str.c_str(), severity_str.c_str());

	// The following breaks into the debugger to help pinpoint what OpenGL
	// call errored
	Utils::debugbreak();
}

const std::string RenderWindow::PERF_METRICS_CPU_OVERHEAD_TIME_KEY = "CPUDisplayTime";

RenderWindow::RenderWindow(int renderer_width, int renderer_height, std::shared_ptr<HIPRTOrochiCtx> hiprt_oro_ctx) : m_viewport_width(renderer_width), m_viewport_height(renderer_height)
{
	// Adding the size of the windows around the viewport such that these windows
	// have their base size and the viewport has the size the the user has asked for
	// (through the commandline)
	int window_width = renderer_width + ImGuiSettingsWindow::BASE_SIZE;
	int window_height = renderer_height + ImGuiLogWindow::BASE_SIZE;

	init_glfw(window_width, window_height);
	init_gl(renderer_width, renderer_height);
	ImGuiRenderer::init_imgui(m_glfw_window);

	m_application_state = std::make_shared<ApplicationState>();
	m_application_settings = std::make_shared<ApplicationSettings>();
	m_renderer = std::make_shared<GPURenderer>(this, hiprt_oro_ctx, m_application_settings);
	m_gpu_baker = std::make_shared<GPUBaker>(m_renderer);

	// Disabling auto samples per frame is accumulation is OFF
	m_application_settings->auto_sample_per_frame = m_renderer->get_render_settings().accumulate ? m_application_settings->auto_sample_per_frame : false;

	m_renderer->resize(renderer_width, renderer_height);

	ThreadManager::start_thread(ThreadManager::RENDER_WINDOW_CONSTRUCTOR, [this, renderer_width, renderer_height]() {
		// m_denoiser->initialize();
		// m_denoiser = std::make_shared<OpenImageDenoiser>();
		// m_denoiser->resize(renderer_width, renderer_height);
		// m_denoiser->set_use_albedo(m_application_settings->denoiser_use_albedo);
		// m_denoiser->set_use_normals(m_application_settings->denoiser_use_normals);
		// m_denoiser->finalize();

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

	TracyGpuContext;

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

	m_viewport_width = pixels_width;
	m_viewport_height = pixels_height;

	// Taking resolution scaling into account
	float& resolution_scale = m_application_settings->render_resolution_scale;
	if (m_application_settings->keep_same_resolution)
		// TODO what about the height changing ?
		resolution_scale = m_application_settings->target_width / static_cast<float>(pixels_width);

	int new_render_width = std::floor(pixels_width * resolution_scale);
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
	float new_render_width = std::floor(m_viewport_width * new_scaling);
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

std::shared_ptr<DisplayViewSystem> RenderWindow::get_display_view_system()
{
	return m_display_view_system;
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

	rotation_x = offset_x / m_viewport_width * M_TWO_PI / m_application_settings->view_rotation_sldwn_x;
	rotation_y = offset_y / m_viewport_height * M_TWO_PI / m_application_settings->view_rotation_sldwn_y;

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
	proportion_converged = m_renderer->get_status_buffer_values().pixel_converged_count / static_cast<float>(m_renderer->m_render_resolution.x * m_renderer->m_render_resolution.y);
	proportion_converged *= 100.0f; // To percentage as used in the ImGui interface

	// We're allowed to stop the render after the given proportion of pixel of the image converged if we're actually
	// using the pixel stop noise threshold feature (enabled + threshold > 0.0f) or if we're using the
	// stop noise threshold but only for the proportion stopping condition (we're not using the threshold of the pixel
	// stop noise threshold feature) --> (enabled & adaptive sampling enabled)
	bool use_proportion_stopping_condition = (render_settings.stop_pixel_noise_threshold > 0.0f && render_settings.use_pixel_stop_noise_threshold)
		|| (render_settings.use_pixel_stop_noise_threshold && render_settings.enable_adaptive_sampling);
	bool minimum_sample_count_reached = render_settings.sample_number >= m_application_settings->pixel_stop_noise_threshold_min_sample_count || render_settings.enable_adaptive_sampling;
	rendering_done |= proportion_converged > render_settings.stop_pixel_percentage_converged && use_proportion_stopping_condition && minimum_sample_count_reached;

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
	bool force_refresh = m_application_state->force_viewport_refresh;

	bool needs_refresh = enough_time_has_passed || realtime_rendering || render_was_reset || force_refresh;
	if (!needs_refresh)
		return false;

	if (m_renderer->get_gmon_render_pass()->is_render_pass_used())
	{
		// With GMoN however, we want to recompute the GMoN framebuffer with the new samples accumulated so far
		// before refreshing the viewport

		if (!needs_refresh)
			// No need of 
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
	float time_since_last_refresh = (glfwGetTimerValue() - m_application_state->last_viewport_refresh_timestamp) / static_cast<float>(glfwGetTimerFrequency()) * 1000.0f;
	return get_viewport_refresh_delay_ms() - time_since_last_refresh;
}

void RenderWindow::reset_render()
{
	m_application_settings->last_denoised_sample_count = -1;

	m_application_state->current_render_time_ms = 0.0f;
	m_application_state->render_dirty = false;
	m_application_state->frame_number = 0;

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
		Utils::debugbreak();
	m_imgui_renderer->set_status_text(status_text);
}

void RenderWindow::clear_ImGui_status_text()
{
	set_ImGui_status_text("Rendering...");
}

float RenderWindow::get_current_render_time()
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
		float difference_ms = (current_time - m_application_state->last_GPU_submit_time) / static_cast<float>(glfwGetTimerFrequency()) * 1000.0f;

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
		float stall_duration = last_frame_time * (1.0f / (1.0f - m_application_settings->GPU_stall_percentage / 100.0f)) - last_frame_time;

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

void RenderWindow::run()
{
	HIPRTRenderSettings& render_settings = m_renderer->get_render_settings();

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

		render();
		m_display_view_system->display();
		m_imgui_renderer->draw_interface();

		// Measuring the CPU overhead before 'glfwSwapBuffers' because we do not want
		// to count the VSync as CPU overhead
		uint64_t cpu_overhead_stop_time = glfwGetTimerValue();

		glfwSwapBuffers(m_glfw_window);
		TracyGpuCollect;

		float delta_time_ms = (glfwGetTimerValue() - frame_start_time) / static_cast<float>(timer_frequency) * 1000.0f;
		m_application_state->last_CPU_frame_delta_time_ms = delta_time_ms;
		m_application_state->last_viewport_refresh_timestamp += m_application_state->last_CPU_frame_delta_time_ms;

		if (!is_rendering_done())
			m_application_state->current_render_time_ms += delta_time_ms;

		if (frame_render_done)
		{
			float cpu_overhead_time = (cpu_overhead_stop_time - frame_start_time) / static_cast<float>(timer_frequency) * 1000.0f;
			m_perf_metrics->add_value(RenderWindow::PERF_METRICS_CPU_OVERHEAD_TIME_KEY, cpu_overhead_time);
			m_perf_metrics->add_value(GPURenderer::FULL_FRAME_TIME_WITH_CPU_KEY, cpu_overhead_time + m_perf_metrics->get_current_value(GPURenderer::ALL_RENDER_PASSES_TIME_KEY));
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

				// We upload the data to the OpenGL textures for displaying
				m_display_view_system->upload_relevant_buffers_to_texture();

				// We want the next frame to be displayed with the same 'wants_render_low_resolution' setting
				// as it was queued with. This is only useful for first frames when getting in low resolution
				// (when we start moving the camera for example) or first frames when getting out of low resolution
				// (when we stop moving the camera). In such situations, the last kernel launch in the GPU queue is
				// a "first frame" that was queued with the corresponding wants_render_low_resolution (getting in or out of low resolution).
				// and so we want to display it the same way.
				m_display_view_system->set_render_low_resolution(m_renderer->was_last_frame_low_resolution());
				// Updating the uniforms so that next time we display, we display correctly
				m_display_view_system->update_current_display_program_uniforms();

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
			bool samples_per_frame_auto_mode = m_application_settings->auto_sample_per_frame;
			bool current_or_last_frame_low_res = render_settings.do_render_low_resolution() || m_renderer->was_last_frame_low_resolution();
			bool using_debug_kernel = m_renderer->is_using_debug_kernel();
			if ((samples_per_frame_auto_mode && current_or_last_frame_low_res && render_settings.accumulate)
				|| using_debug_kernel)
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
				render_settings.samples_per_frame = std::min(std::max(1, static_cast<int>(m_application_state->samples_per_second / m_application_settings->target_GPU_framerate)), 65536);

			if (m_application_state->render_dirty)
				reset_render();

			m_application_state->GPU_stall_duration_left = compute_GPU_stall_duration();
			m_application_state->interacting_last_frame = is_interacting();

			// Queuing a new frame for the GPU to render
			uint64_t current_timestamp = glfwGetTimerValue();
			float delta_time_gpu = (current_timestamp - m_application_state->last_GPU_submit_time) / static_cast<float>(glfwGetTimerFrequency()) * 1000.0f;

			m_application_state->frame_number++;
			m_application_state->last_GPU_submit_time = current_timestamp;

			m_renderer->render(delta_time_gpu, this);

			buffer_upload_necessary = true;
		}
		else // The rendering is done
		{
			buffer_upload_necessary |= m_display_view_system->update_selected_display_view();

			if (m_application_settings->enable_denoising)
			{
				// We may still want to denoise on the final frame
				if (denoise())
					buffer_upload_necessary = true;
			}

			if (buffer_upload_necessary)
			{
				// Re-uploading only if necessary
				m_display_view_system->upload_relevant_buffers_to_texture();

				buffer_upload_necessary = false;
			}

			m_display_view_system->set_render_low_resolution(m_renderer->was_last_frame_low_resolution());
			// Updating the uniforms if the user touches the post processing parameters
			// or something else (denoiser blend, ...)
			m_display_view_system->update_current_display_program_uniforms();

			RendererAnimationState& renderer_animation_state = m_renderer->get_animation_state();
			if (renderer_animation_state.is_rendering_frame_sequence && renderer_animation_state.frames_rendered_so_far < renderer_animation_state.number_of_animation_frames)
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
	DisplaySettings& display_settings = m_display_view_system->get_display_settings();

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
		bool final_frame_denoised_already = !m_application_settings->denoiser_settings_changed && rendering_done && m_application_settings->last_denoised_sample_count == render_settings.sample_number;



		// ---- Conditions for denoising / displaying noisy ----
		// - Is the rendering done 
		// - And we only want to denoise when the rendering is done
		// - And we haven't alraedy denoised the final frame
		bool denoise_rendering_done = rendering_done && denoise_when_done && !final_frame_denoised_already;
		// Have we rendered enough samples since last time we denoised that we need to denoise again?
		bool sample_skip_threshold_reached = !denoise_when_done && (render_settings.sample_number - std::max(0, m_application_settings->last_denoised_sample_count) >= m_application_settings->denoiser_sample_skip);
		// We're also going to denoise if we changed the denoiser settings
		// (because we need to denoise to reflect the new settings)
		bool denoiser_settings_changed = m_application_settings->denoiser_settings_changed;




		bool need_denoising = false;
		bool display_noisy = false;

		// Denoise if:
		//	- The render is done and we're denoising when the render 
		//	- We have rendered enough samples since the last denoise step that we need to denoise again
		//	- We're not denoising if we're interacting (moving the camera)
		need_denoising |= denoise_rendering_done;
		need_denoising |= sample_skip_threshold_reached;
		need_denoising |= denoiser_settings_changed;
		need_denoising &= !is_interacting();

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

			m_application_settings->last_denoised_duration = denoise_duration;
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
	std::shared_ptr<OpenGLInteropBuffer<float3>> normals_buffer = nullptr;
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
	std::shared_ptr<OrochiBuffer<float3>> normals_buffer = nullptr;
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
