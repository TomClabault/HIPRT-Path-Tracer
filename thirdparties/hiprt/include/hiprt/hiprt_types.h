#pragma once

#if ( defined( __CUDACC__ ) || defined( __HIPCC__ ) )
#define __KERNELCC__
#endif

#include <hiprt/hiprt_vec.h>

struct _hiprtGeometry;
struct _hiprtScene;
struct _hiprtContext;
struct _hiprtFuncTable;

typedef void*			 hiprtDevicePtr;
typedef _hiprtGeometry*	 hiprtGeometry;
typedef _hiprtScene*	 hiprtScene;
typedef _hiprtContext*	 hiprtContext;
typedef _hiprtFuncTable* hiprtFuncTable;
typedef uint32_t		 hiprtLogLevel;
typedef uint32_t		 hiprtBuildFlags;
typedef uint32_t		 hiprtRayMask;

typedef int	  hiprtApiDevice;	// hipDevice, cuDevice
typedef void* hiprtApiCtx;		// hipCtx, cuCtx
typedef void* hiprtApiStream;	// hipStream, cuStream
typedef void* hiprtApiFunction; // hipFunction, cuFunction
typedef void* hiprtApiModule;	// hipModule, cuModule

/** \brief Ray traversal type.
 *
 */
enum hiprtTraversalType
{
	/*!< 0 or 1 element iterator with any hit along the ray */
	hiprtTraversalTerminateAtAnyHit = 1,
	/*!< 0 or 1 element iterator with a closest hit along the ray */
	hiprtTraversalTerminateAtClosestHit = 2,
};

/** \brief Traversal state.
 *
 * On-device traversal can be in either hit state (and can be continued using
 * hiprtNextHit) or finished state.
 */
enum hiprtTraversalState
{
	hiprtTraversalStateInit,
	hiprtTraversalStateFinished,
	hiprtTraversalStateHit,
	hiprtTraversalStateStackOverflow
};

/** \brief Traversal hint.
 *
 * An additional information about the rays for the traversal object.
 * It is taken into account only on AMD Navi3x (RDNA3) and above.
 */
enum hiprtTraversalHint
{
	hiprtTraversalHintDefault		 = 0,
	hiprtTraversalHintShadowRays	 = 1,
	hiprtTraversalHintReflectionRays = 2
};

/** \brief Various constants.
 *
 */
enum : uint32_t
{
	hiprtInvalidValue			   = hiprt::InvalidValue,
	hiprtFullRayMask			   = hiprt::FullRayMask,
	hiprtMaxBatchBuildMaxPrimCount = hiprt::MaxBatchBuildMaxPrimCount,
	hiprtMaxInstanceLevels		   = hiprt::MaxInstanceLevels,
	hiprtBranchingFactor		   = hiprt::BranchingFactor
};

/** \brief Error codes.
 *
 */
enum hiprtError
{
	hiprtSuccess				= 0,
	hiprtErrorNotImplemented	= 1,
	hiprtErrorInternal			= 2,
	hiprtErrorOutOfHostMemory	= 3,
	hiprtErrorOutOfDeviceMemory = 4,
	hiprtErrorInvalidApiVersion = 5,
	hiprtErrorInvalidParameter	= 6
};

/** \brief Log levels.
 *
 */
enum hiprtLogLevelBits
{
	hiprtLogLevelNone  = 0,
	hiprtLogLevelInfo  = 1 << 0,
	hiprtLogLevelWarn  = 1 << 1,
	hiprtLogLevelError = 1 << 2
};

/** \brief Type of geometry/scene build operation.
 *
 * hiprtBuildGeometry/hiprtBuildScene can either build or update
 * an underlying acceleration structure.
 */
enum hiprtBuildOperation
{
	hiprtBuildOperationBuild  = 1,
	hiprtBuildOperationUpdate = 2
};

/** \brief Hint flags for geometry/scene build functions.
 *
 * hiprtBuildGeometry/hiprtBuildScene use these flags to choose
 * an appropriate build format/algorithm.
 */
enum hiprtBuildFlagBits
{
	hiprtBuildFlagBitPreferFastBuild		= 0,
	hiprtBuildFlagBitPreferBalancedBuild	= 1,
	hiprtBuildFlagBitPreferHighQualityBuild = 2,
	hiprtBuildFlagBitCustomBvhImport		= 3,
	hiprtBuildFlagBitDisableSpatialSplits	= 1 << 2,
	hiprtBuildFlagBitDisableTrianglePairing = 1 << 3
};

/** \brief Geometric primitive type.
 *
 * hiprtGeometry can be built from multiple primitive types,
 * such as triangle meshes, AABB lists, line lists, etc. This enum
 * defines primitive type for hiprtBuildGeometry function.
 */
enum hiprtPrimitiveType
{
	hiprtPrimitiveTypeTriangleMesh,
	hiprtPrimitiveTypeAABBList
};

/** \brief Instance type.
 *
 * hiprtScene can be bult from instances either of hiprtGeometry or hiprtScene.
 * This enum defines instance type for hiprtBuildScene function.
 */
enum hiprtInstanceType
{
	hiprtInstanceTypeGeometry,
	hiprtInstanceTypeScene
};

/** \brief Primitve types
 *
 */
enum hiprtPrimitiveNodeType
{
	hiprtTriangleNode = 0,
	hiprtCustomNode	  = 1
};

/** \brief Transformation frame type.
 *
 */
enum hiprtFrameType
{
	hiprtFrameTypeSRT,
	hiprtFrameTypeMatrix
};

/** \brief Stack type.
 *
 */
enum hiprtStackType
{
	hiprtStackTypeGlobal,
	hiprtStackTypeDynamic
};

/** \brief Stack entry type.
 *
 */
enum hiprtStackEntryType
{
	hiprtStackEntryTypeInteger,
	hiprtStackEntryTypeInstance
};

/** \brief Bvh node type.
 *
 */
enum hiprtBvhNodeType
{
	/*!< Leaf node */
	hiprtBvhNodeTypeInternal = 0,
	/*!< Internal node */
	hiprtBvhNodeTypeLeaf = 1,
};

/** \brief Ray data structure.
 *
 */
struct alignas( 16 ) hiprtRay
{
	/*!< Ray origin */
	hiprtFloat3 origin;
	/*!< Ray maximum distance */
	float minT = 0.0f;
	/*!< Ray direction */
	hiprtFloat3 direction;
	/*!< Ray maximum distance */
	float maxT = hiprt::FltMax;
};
HIPRT_STATIC_ASSERT( sizeof( hiprtRay ) == 32 );

/** \brief Ray hit data structure.
 *
 */
struct alignas( 16 ) hiprtHit
{
	/*!< Instance IDs */
	union
	{
		/*!< Instance ID (for a single level instancing) */
		uint32_t instanceID = hiprtInvalidValue;
		/*!< Instance IDs */
		uint32_t instanceIDs[hiprtMaxInstanceLevels];
	};
	/*!< Primitive ID */
	uint32_t primID = hiprtInvalidValue;
	/*!< Texture coordinates */
	hiprtFloat2 uv;
	/*!< Geometric normal (not normalized and in the object space) */
	hiprtFloat3 normal;
	/*!< Distance */
	float t = -1.0f;

	HIPRT_DEVICE bool hasHit() const { return primID != hiprtInvalidValue; }
};
HIPRT_STATIC_ASSERT( sizeof( hiprtHit ) == 48 );

/** \brief Set of device data pointers for custom functions.
 *
 */
struct hiprtFuncDataSet
{
	const void* intersectFuncData = nullptr;
	const void* filterFuncData	  = nullptr;
};

/** \brief A header of the function table.
 *
 */
struct hiprtFuncTableHeader
{
	uint32_t		  numGeomTypes;
	uint32_t		  numRayTypes;
	hiprtFuncDataSet* funcDataSets;
};

/** \brief A header of the global stack buffer.
 * Use API functions to create this buffer.
 * - hiprtCreateStackBuffer
 * - hiprtDestroyStackBuffer
 */
struct hiprtGlobalStackBuffer
{
	uint32_t stackSize;
	uint32_t stackCount;
	void*	 stackData;
};

/** \brief A header of the shared stack buffer.
 *
 */
struct hiprtSharedStackBuffer
{
	uint32_t stackSize;
	void*	 stackData;
};

/** \brief Set of function names.
 *
 */
struct hiprtFuncNameSet
{
	const char* intersectFuncName = nullptr;
	const char* filterFuncName	  = nullptr;
};

/** \brief Device type.
 *
 */
enum hiprtDeviceType
{
	/*!< AMD device */
	hiprtDeviceAMD,
	/*!< Nvidia device */
	hiprtDeviceNVIDIA,
};

/** \brief Context creation input.
 *
 */
struct hiprtContextCreationInput
{
	/*!< HIPRT API context */
	hiprtApiCtx ctxt;
	/*!< HIPRT API device */
	hiprtApiDevice device;
	/*!< HIPRT API device type */
	hiprtDeviceType deviceType;
};

/** \brief Various flags controlling scene/geometry build process.
 *
 */
struct hiprtBuildOptions
{
	/*!< Build flags */
	hiprtBuildFlags buildFlags;
	/*!< Batch build max prim count (if 0 then batch build is not used) */
	uint32_t batchBuildMaxPrimCount = 0u;
};

/** \brief Triangle mesh primitive.
 *
 * Triangle mesh primitive is represented as an indexed vertex array.
 * Vertex and index arrays are defined using device pointers and strides.
 * Each vertex has to have 3 components: (x, y, z) coordinates.
 * Indices are organized into triples (i0, i1, i2) - one for each triangle.
 * If the indices are not provided, it assumes (3*t+0, 3*t+1, 3*t+2).
 */
struct hiprtTriangleMeshPrimitive
{
	/*!< Device pointer to vertex data */
	hiprtDevicePtr vertices;
	/*!< Number of vertices in vertex array */
	uint32_t vertexCount;
	/*!< Stride in bytes between two vertices */
	uint32_t vertexStride;

	/*!< Device pointer to triangle index data (optional) */
	hiprtDevicePtr triangleIndices = nullptr;
	/*!< Number of triangles in index array */
	uint32_t triangleCount = 0u;
	/*!< Stride in bytes between two triangles */
	uint32_t triangleStride = 0u;

	/*!< Device pointer to triangle pair index data (optional) */
	hiprtDevicePtr trianglePairIndices = nullptr;
	/*!< Number of triangle pairs */
	uint32_t trianglePairCount = 0u;
};

/** \brief AABB list primitive.
 *
 * AABB list is an array of axis aligned bounding boxes, represented
 * by device memory pointer and stride between two consecutive boxes.
 * Each AABB is a pair of float3 or float4 values.
 */
struct hiprtAABBListPrimitive
{
	/*!< Device pointer to AABB data */
	hiprtDevicePtr aabbs;
	/*!< Number of AABBs in the array */
	uint32_t aabbCount;
	/*!< Stride in bytes between two AABBs (2 * sizeof(float3) or 2 * sizeof(float4)) */
	uint32_t aabbStride;
};

/** \brief Bvh node for custom import Bvh.
 *
 */
struct alignas( 64 ) hiprtBvhNode
{
	/*!< Child indices (empty slot needs to be marked by hiprtInvalidValue) */
	uint32_t childIndices[hiprtBranchingFactor];
	/*!< Child node types */
	hiprtBvhNodeType childNodeTypes[hiprtBranchingFactor];
	/*!< Child node bounding box min's */
	hiprtFloat3 childAabbsMin[hiprtBranchingFactor];
	/*!< Child ode bounding box max's */
	hiprtFloat3 childAabbsMax[hiprtBranchingFactor];
};
HIPRT_STATIC_ASSERT( sizeof( hiprtBvhNode ) == 128 );

/** \brief Bvh node list.
 *
 */
struct hiprtBvhNodeList
{
	/*!< Array of hiprtBvhNode's */
	hiprtDevicePtr nodes;
	/*!< Number of nodes */
	uint32_t nodeCount;
};

/** \brief Build input for geometry build/update operation.
 *
 * Build input defines concrete primitive type and a pointer to an actual
 * primitive description.
 */
struct alignas( 64 ) hiprtGeometryBuildInput
{
	/*!< Primitive type */
	hiprtPrimitiveType type;
	/*!< Geometry type used for custom function table */
	uint32_t geomType = hiprtInvalidValue;
	/*!< Defines the following union */
	union
	{
		/*!< Triangle mesh */
		hiprtTriangleMeshPrimitive triangleMesh;
		/*!< Bounding boxes of custom primitives */
		hiprtAABBListPrimitive aabbList;
	} primitive{};
	/*!< Custom Bvh nodes (optional) */
	hiprtBvhNodeList nodeList;
};
HIPRT_STATIC_ASSERT( sizeof( hiprtGeometryBuildInput ) == 128 );

/** \brief Instance containing a pointer to the actual geometry/scene.
 *
 */
struct alignas( 16 ) hiprtInstance
{
	/*!< Instance type */
	hiprtInstanceType type;
	/*!< Defines the following union */
	union
	{
		/*!< Geometry */
		hiprtGeometry geometry;
		/*!< Scene */
		hiprtScene scene;
	};
};
HIPRT_STATIC_ASSERT( sizeof( hiprtInstance ) == 16 );

/** \brief Build input for the scene.
 *
 * Scene consists of a set of instances. Each of the instances is defined by:
 *  - Root pointer of the corresponding geometry
 *  - Transformation header
 *  - Mask
 *
 * Instances can refer to the same geometry but with different transformations
 * (essentially implementing instancing). Mask is used to implement ray
 * masking: ray mask is bitwise &ded with an instance mask, and no intersections
 * are evaluated with the primitive of corresponding instance if the result is
 * 0. The transformation header defines the offset and the number of consecutive
 * transformation frames in the frame array for each instance. More than one frame
 * is interpreted as motion blur. If the transformation headers is NULL, it
 * assumes one frame per instance. Optionally, it is possible to import a custom
 * BVH by setting nodes and the corresponding build flag.
 */
struct alignas( 16 ) hiprtSceneBuildInput
{
	/*!< Array of instanceCount pointers to instances */
	hiprtDevicePtr instances;
	/*!< Array of instanceCount transform headers (optional: per object frame assumed if NULL) */
	hiprtDevicePtr instanceTransformHeaders;
	/*!< Array of frameCount frames (supposed to be ordered according to time) */
	hiprtDevicePtr instanceFrames;
	/*!< Per object bit masks for instance masking (optional: if NULL masks treated as hiprtFullRayMask) */
	hiprtDevicePtr instanceMasks;
	/*!< Custom Bvh nodes (optional) */
	hiprtBvhNodeList nodeList;
	/*!< Number of instances */
	uint32_t instanceCount;
	/*!< Number of frames (such that instanceCount <= frameCount) */
	uint32_t frameCount;
	/*!< Frame type (SRT or matrix) */
	hiprtFrameType frameType = hiprtFrameTypeSRT;
};
HIPRT_STATIC_ASSERT( sizeof( hiprtSceneBuildInput ) == 64 );

/** \brief Input for the global stack buffer allocation
 *
 */
struct hiprtGlobalStackBufferInput
{
	/*!< Stack type */
	hiprtStackType type = hiprtStackTypeGlobal;
	/*!< Stack entry type */
	hiprtStackEntryType entryType = hiprtStackEntryTypeInteger;
	/*!< Global stack size (e.g. 64) */
	uint32_t stackSize;
	/*!< Total number of threads (for hiprtGlobalStack only) */
	uint32_t threadCount;
};

/** \brief Stack entry for instace stacks
 *
 */
struct hiprtInstanceStackEntry
{
	/*!< Ray */
	hiprtRay ray;
	/*!< Scene */
	hiprtScene scene;
};

/** \brief SRT transformation frame.
 *
 * Represented by scale (S), rotation (R), translation (T), and frame time.
 * Object to world transformation is composed as (T * R * S) * x = y
 */
struct alignas( 16 ) hiprtFrameSRT
{
	/*!< Rotation (axis and angle) */
	hiprtFloat4 rotation;
	/*!< Scale */
	hiprtFloat3 scale;
	/*!< Translation */
	hiprtFloat3 translation;
	/*!< Frame time */
	float time;
};
HIPRT_STATIC_ASSERT( sizeof( hiprtFrameSRT ) == 48 );

/** \brief Transformation matrix frame representation.
 *
 * Represented by a 3x4 matrix and frame time.
 */
struct alignas( 64 ) hiprtFrameMatrix
{
	/*!< Matrix */
	float matrix[3][4];
	/*!< Frame time */
	float time;
};
HIPRT_STATIC_ASSERT( sizeof( hiprtFrameMatrix ) == 64 );

/** \brief Transformation header.
 *
 * Defines defines the index to the array of frames and the number of frames.
 */
struct alignas( 8 ) hiprtTransformHeader
{
	/*!< Frame index */
	uint32_t frameIndex;
	/*!< Number of frames */
	uint32_t frameCount;
};
HIPRT_STATIC_ASSERT( sizeof( hiprtTransformHeader ) == 8 );
