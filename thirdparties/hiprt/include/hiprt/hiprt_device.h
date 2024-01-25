#pragma once

#define HIPRT_PUBLIC_DEVICE_H
#define HIPRT_DEVICE __device__

#include <hiprt/hiprt_common.h>
#include <hiprt/hiprt_types.h>
#include <hiprt/hiprt_vec.h>

/** \brief An empty dummy instance stack.
 *
 * Use this stack if you use one level of instancing.
 */
class hiprtEmptyInstanceStack
{
};

/** \brief A stack using (slow) local memory internally.
 *
 */
template <typename StackEntry, uint32_t StackSize>
class hiprtPrivateStack_impl;

class hiprtPrivateStack
{
  public:
	static constexpr uint32_t StackSize = 64u;
	HIPRT_DEVICE			  hiprtPrivateStack();
	HIPRT_DEVICE ~hiprtPrivateStack();
	HIPRT_DEVICE uint32_t pop();
	HIPRT_DEVICE void	  push( uint32_t val );
	HIPRT_DEVICE bool	  empty() const;
	HIPRT_DEVICE uint32_t vacancy() const;
	HIPRT_DEVICE void	  reset();

  private:
	hiprtPimpl<hiprtPrivateStack_impl<uint32_t, StackSize>, SizePrivateStack, AlignmentPrivateStack> m_impl;
};

/** \brief A instance stack using (slow) local memory internally.
 *
 */
class hiprtPrivateInstanceStack
{
  public:
	static constexpr uint32_t StackSize = hiprtMaxInstanceLevels - 1;
	HIPRT_DEVICE			  hiprtPrivateInstanceStack();
	HIPRT_DEVICE ~hiprtPrivateInstanceStack();
	HIPRT_DEVICE hiprtInstanceStackEntry pop();
	HIPRT_DEVICE void					 push( hiprtInstanceStackEntry val );
	HIPRT_DEVICE bool					 empty() const;
	HIPRT_DEVICE uint32_t				 vacancy() const;
	HIPRT_DEVICE void					 reset();

  private:
	hiprtPimpl<
		hiprtPrivateStack_impl<hiprtInstanceStackEntry, StackSize>,
		SizePrivateInstanceStack,
		AlignmentPrivateInstanceStack>
		m_impl;
};

/** \brief A stack using both (fast) shared memory and (slow) global memory.
 *
 * The stack uses shared memory if there is enough space.
 * Otherwise, it uses global memory as a backup.
 */
template <typename StackEntry, bool DynamicAssignment>
class hiprtGlobalStack_impl;

class hiprtGlobalStack
{
  public:
	HIPRT_DEVICE
	hiprtGlobalStack( hiprtGlobalStackBuffer globalStackBuffer, hiprtSharedStackBuffer sharedStackBuffer );
	HIPRT_DEVICE ~hiprtGlobalStack();
	HIPRT_DEVICE uint32_t pop();
	HIPRT_DEVICE void	  push( uint32_t val );
	HIPRT_DEVICE uint32_t vacancy() const;
	HIPRT_DEVICE bool	  empty() const;
	HIPRT_DEVICE void	  reset();

  private:
	hiprtPimpl<hiprtGlobalStack_impl<uint32_t, false>, SizeGlobalStack, AlignmentGlobalStack> m_impl;
};

/** \brief An instance stack using both (fast) shared memory and (slow) global memory.
 *
 * The stack uses shared memory if there is enough space.
 * Otherwise, it uses global memory as a backup.
 */
class hiprtGlobalInstanceStack
{
  public:
	HIPRT_DEVICE
	hiprtGlobalInstanceStack( hiprtGlobalStackBuffer globalStackBuffer, hiprtSharedStackBuffer sharedStackBuffer );
	HIPRT_DEVICE ~hiprtGlobalInstanceStack();
	HIPRT_DEVICE hiprtInstanceStackEntry pop();
	HIPRT_DEVICE void					 push( hiprtInstanceStackEntry val );
	HIPRT_DEVICE uint32_t				 vacancy() const;
	HIPRT_DEVICE bool					 empty() const;
	HIPRT_DEVICE void					 reset();

  private:
	hiprtPimpl<hiprtGlobalStack_impl<hiprtInstanceStackEntry, false>, SizeGlobalStack, AlignmentGlobalStack> m_impl;
};

/** \brief A stack using both (fast) shared memory and (slow) global memory with dynamic assignment.
 *
 * The stack uses shared memory if there is enough space.
 * Otherwise, it uses global memory as a backup.
 */
class hiprtDynamicStack
{
  public:
	HIPRT_DEVICE
	hiprtDynamicStack( hiprtGlobalStackBuffer globalStackBuffer, hiprtSharedStackBuffer sharedStackBuffer );
	HIPRT_DEVICE ~hiprtDynamicStack();
	HIPRT_DEVICE uint32_t pop();
	HIPRT_DEVICE void	  push( uint32_t val );
	HIPRT_DEVICE uint32_t vacancy() const;
	HIPRT_DEVICE bool	  empty() const;
	HIPRT_DEVICE void	  reset();

  private:
	hiprtPimpl<hiprtGlobalStack_impl<uint32_t, true>, SizeGlobalInstanceStack, AlignmentGlobalInstanceStack> m_impl;
};

/** \brief An instance stack using both (fast) shared memory and (slow) global memory with dynamic assignment.
 *
 * The stack uses shared memory if there is enough space.
 * Otherwise, it uses global memory as a backup.
 */
class hiprtDynamicInstanceStack
{
  public:
	HIPRT_DEVICE
	hiprtDynamicInstanceStack( hiprtGlobalStackBuffer globalStackBuffer, hiprtSharedStackBuffer sharedStackBuffer );
	HIPRT_DEVICE ~hiprtDynamicInstanceStack();
	HIPRT_DEVICE hiprtInstanceStackEntry pop();
	HIPRT_DEVICE void					 push( hiprtInstanceStackEntry val );
	HIPRT_DEVICE uint32_t				 vacancy() const;
	HIPRT_DEVICE bool					 empty() const;
	HIPRT_DEVICE void					 reset();

  private:
	hiprtPimpl<hiprtGlobalStack_impl<hiprtInstanceStackEntry, true>, SizeGlobalInstanceStack, AlignmentGlobalInstanceStack>
		m_impl;
};

/** \brief A traversal object for finding the closest hit with hiprtGeometry containing triangles.
 *
 * It uses a private stack with size 64 internally.
 */
template <hiprtPrimitiveNodeType PrimitiveNodeType, hiprtTraversalType TraversalType>
class hiprtGeomTraversal_impl;

class hiprtGeomTraversalClosest
{
  public:
	HIPRT_DEVICE hiprtGeomTraversalClosest(
		hiprtGeometry	   geom,
		const hiprtRay&	   ray,
		hiprtTraversalHint hint		 = hiprtTraversalHintDefault,
		void*			   payload	 = nullptr,
		hiprtFuncTable	   funcTable = nullptr,
		uint32_t		   rayType	 = 0 );
	HIPRT_DEVICE hiprtHit			 getNextHit();
	HIPRT_DEVICE hiprtTraversalState getCurrentState();

  private:
	hiprtPimpl<
		hiprtGeomTraversal_impl<hiprtTriangleNode, hiprtTraversalTerminateAtClosestHit>,
		SizeGeomTraversalPrivateStack,
		AlignmentGeomTraversalPrivateStack>
		m_impl;
};

/** \brief A traversal object for finding the any hit with hiprtGeometry containing triangles.
 *
 * It uses a private stack with size 64 internally.
 */
class hiprtGeomTraversalAnyHit
{
  public:
	HIPRT_DEVICE hiprtGeomTraversalAnyHit(
		hiprtGeometry	   geom,
		const hiprtRay&	   ray,
		hiprtTraversalHint hint		 = hiprtTraversalHintDefault,
		void*			   payload	 = nullptr,
		hiprtFuncTable	   funcTable = nullptr,
		uint32_t		   rayType	 = 0 );
	HIPRT_DEVICE hiprtHit			 getNextHit();
	HIPRT_DEVICE hiprtTraversalState getCurrentState();

  private:
	hiprtPimpl<
		hiprtGeomTraversal_impl<hiprtTriangleNode, hiprtTraversalTerminateAtAnyHit>,
		SizeGeomTraversalPrivateStack,
		AlignmentGeomTraversalPrivateStack>
		m_impl;
};

/** \brief A traversal object for finding the closest hit with hiprtGeometry containing custom primitives.
 *
 * It uses a private stack with size 64 internally.
 */
class hiprtGeomCustomTraversalClosest
{
  public:
	HIPRT_DEVICE hiprtGeomCustomTraversalClosest(
		hiprtGeometry	   geom,
		const hiprtRay&	   ray,
		hiprtTraversalHint hint		 = hiprtTraversalHintDefault,
		void*			   payload	 = nullptr,
		hiprtFuncTable	   funcTable = nullptr,
		uint32_t		   rayType	 = 0 );
	HIPRT_DEVICE hiprtHit			 getNextHit();
	HIPRT_DEVICE hiprtTraversalState getCurrentState();

  private:
	hiprtPimpl<
		hiprtGeomTraversal_impl<hiprtCustomNode, hiprtTraversalTerminateAtClosestHit>,
		SizeGeomTraversalPrivateStack,
		AlignmentGeomTraversalPrivateStack>
		m_impl;
};

/** \brief A traversal object for finding the any hit with hiprtGeometry containing custom primitives.
 *
 * It uses a private stack with size 64 internally.
 */
class hiprtGeomCustomTraversalAnyHit
{
  public:
	HIPRT_DEVICE hiprtGeomCustomTraversalAnyHit(
		hiprtGeometry	   geom,
		const hiprtRay&	   ray,
		hiprtTraversalHint hint		 = hiprtTraversalHintDefault,
		void*			   payload	 = nullptr,
		hiprtFuncTable	   funcTable = nullptr,
		uint32_t		   rayType	 = 0 );
	HIPRT_DEVICE hiprtHit			 getNextHit();
	HIPRT_DEVICE hiprtTraversalState getCurrentState();

  private:
	hiprtPimpl<
		hiprtGeomTraversal_impl<hiprtCustomNode, hiprtTraversalTerminateAtAnyHit>,
		SizeGeomTraversalPrivateStack,
		AlignmentGeomTraversalPrivateStack>
		m_impl;
};

/** \brief A traversal object for finding the closest hit with hiprtScene.
 *
 * It uses a private stack with size 64 internally.
 */
template <hiprtTraversalType TraversalType>
class hiprtSceneTraversal_impl;

class hiprtSceneTraversalClosest
{
  public:
	HIPRT_DEVICE hiprtSceneTraversalClosest(
		hiprtScene		   scene,
		const hiprtRay&	   ray,
		hiprtRayMask	   mask		 = hiprtFullRayMask,
		hiprtTraversalHint hint		 = hiprtTraversalHintDefault,
		void*			   payload	 = nullptr,
		hiprtFuncTable	   funcTable = nullptr,
		uint32_t		   rayType	 = 0,
		float			   time		 = 0.0f );
	HIPRT_DEVICE hiprtHit			 getNextHit();
	HIPRT_DEVICE hiprtTraversalState getCurrentState();

  private:
	hiprtPimpl<
		hiprtSceneTraversal_impl<hiprtTraversalTerminateAtClosestHit>,
		SizeSceneTraversalPrivateStack,
		AlignmentSceneTraversalPrivateStack>
		m_impl;
};

/** \brief A traversal object for finding the any hit with hiprtScene.
 *
 * It uses a private stack with size 64 internally.
 */
class hiprtSceneTraversalAnyHit
{
  public:
	HIPRT_DEVICE hiprtSceneTraversalAnyHit(
		hiprtScene		   scene,
		const hiprtRay&	   ray,
		hiprtRayMask	   mask		 = hiprtFullRayMask,
		hiprtTraversalHint hint		 = hiprtTraversalHintDefault,
		void*			   payload	 = nullptr,
		hiprtFuncTable	   funcTable = nullptr,
		uint32_t		   rayType	 = 0,
		float			   time		 = 0.0f );
	HIPRT_DEVICE hiprtHit			 getNextHit();
	HIPRT_DEVICE hiprtTraversalState getCurrentState();

  private:
	hiprtPimpl<
		hiprtSceneTraversal_impl<hiprtTraversalTerminateAtAnyHit>,
		SizeSceneTraversalPrivateStack,
		AlignmentSceneTraversalPrivateStack>
		m_impl;
};

/** \brief A traversal object for finding the closest hit with hiprtGeometry containing triangles.
 *
 * \tparam hiprtStack A custom stack.
 */
template <typename hiprtStack, hiprtPrimitiveNodeType PrimitiveNodeType, hiprtTraversalType TraversalType>
class hiprtGeomTraversalCustomStack_impl;

template <typename hiprtStack>
class hiprtGeomTraversalClosestCustomStack
{
  public:
	HIPRT_DEVICE hiprtGeomTraversalClosestCustomStack(
		hiprtGeometry	   geom,
		const hiprtRay&	   ray,
		hiprtStack&		   stack,
		hiprtTraversalHint hint		 = hiprtTraversalHintDefault,
		void*			   payload	 = nullptr,
		hiprtFuncTable	   funcTable = nullptr,
		uint32_t		   rayType	 = 0 );
	HIPRT_DEVICE hiprtHit			 getNextHit();
	HIPRT_DEVICE hiprtTraversalState getCurrentState();

  private:
	hiprtPimpl<
		hiprtGeomTraversalCustomStack_impl<hiprtStack, hiprtTriangleNode, hiprtTraversalTerminateAtClosestHit>,
		SizeGeomTraversalCustomStack,
		AlignmentGeomTraversalCustomStack>
		m_impl;
};

/** \brief A traversal object for finding the any hit with hiprtGeometry containing triangles.
 *
 * \tparam hiprtStack A custom stack.
 */
template <typename hiprtStack>
class hiprtGeomTraversalAnyHitCustomStack
{
  public:
	HIPRT_DEVICE hiprtGeomTraversalAnyHitCustomStack(
		hiprtGeometry	   geom,
		const hiprtRay&	   ray,
		hiprtStack&		   stack,
		hiprtTraversalHint hint		 = hiprtTraversalHintDefault,
		void*			   payload	 = nullptr,
		hiprtFuncTable	   funcTable = nullptr,
		uint32_t		   rayType	 = 0 );
	HIPRT_DEVICE hiprtHit			 getNextHit();
	HIPRT_DEVICE hiprtTraversalState getCurrentState();

  private:
	hiprtPimpl<
		hiprtGeomTraversalCustomStack_impl<hiprtStack, hiprtTriangleNode, hiprtTraversalTerminateAtAnyHit>,
		SizeGeomTraversalCustomStack,
		AlignmentGeomTraversalCustomStack>
		m_impl;
};

/** \brief A traversal object for finding the closest hit with hiprtGeometry containing custom primitives.
 *
 * \tparam hiprtStack A custom stack.
 */
template <typename hiprtStack>
class hiprtGeomCustomTraversalClosestCustomStack
{
  public:
	HIPRT_DEVICE hiprtGeomCustomTraversalClosestCustomStack(
		hiprtGeometry	   geom,
		const hiprtRay&	   ray,
		hiprtStack&		   stack,
		hiprtTraversalHint hint		 = hiprtTraversalHintDefault,
		void*			   payload	 = nullptr,
		hiprtFuncTable	   funcTable = nullptr,
		uint32_t		   rayType	 = 0 );
	HIPRT_DEVICE hiprtHit			 getNextHit();
	HIPRT_DEVICE hiprtTraversalState getCurrentState();

  private:
	hiprtPimpl<
		hiprtGeomTraversalCustomStack_impl<hiprtStack, hiprtCustomNode, hiprtTraversalTerminateAtClosestHit>,
		SizeGeomTraversalCustomStack,
		AlignmentGeomTraversalCustomStack>
		m_impl;
};

/** \brief A traversal object for finding the any hit with hiprtGeometry containing custom primitives.
 *
 * \tparam hiprtStack A custom stack.
 */
template <typename hiprtStack>
class hiprtGeomCustomTraversalAnyHitCustomStack
{
  public:
	HIPRT_DEVICE hiprtGeomCustomTraversalAnyHitCustomStack(
		hiprtGeometry	   geom,
		const hiprtRay&	   ray,
		hiprtStack&		   stack,
		hiprtTraversalHint hint		 = hiprtTraversalHintDefault,
		void*			   payload	 = nullptr,
		hiprtFuncTable	   funcTable = nullptr,
		uint32_t		   rayType	 = 0 );
	HIPRT_DEVICE hiprtHit			 getNextHit();
	HIPRT_DEVICE hiprtTraversalState getCurrentState();

  private:
	hiprtPimpl<
		hiprtGeomTraversalCustomStack_impl<hiprtStack, hiprtCustomNode, hiprtTraversalTerminateAtAnyHit>,
		SizeGeomTraversalCustomStack,
		AlignmentGeomTraversalCustomStack>
		m_impl;
};

/** \brief A traversal object for finding the closest hit with hiprtScene.
 *
 * \tparam hiprtStack A custom stack.
 */
template <typename hiprtStack, typename hiprtInstanceStack, hiprtTraversalType TraversalType>
class hiprtSceneTraversalCustomStack_impl;

template <typename hiprtStack, typename hiprtInstanceStack>
class hiprtSceneTraversalClosestCustomStack
{
  public:
	HIPRT_DEVICE hiprtSceneTraversalClosestCustomStack(
		hiprtScene			scene,
		const hiprtRay&		ray,
		hiprtStack&			stack,
		hiprtInstanceStack& instanceStack,
		hiprtRayMask		mask	  = hiprtFullRayMask,
		hiprtTraversalHint	hint	  = hiprtTraversalHintDefault,
		void*				payload	  = nullptr,
		hiprtFuncTable		funcTable = nullptr,
		uint32_t			rayType	  = 0,
		float				time	  = 0.0f );
	HIPRT_DEVICE hiprtHit			 getNextHit();
	HIPRT_DEVICE hiprtTraversalState getCurrentState();

  private:
	hiprtPimpl<
		hiprtSceneTraversalCustomStack_impl<hiprtStack, hiprtInstanceStack, hiprtTraversalTerminateAtClosestHit>,
		SizeSceneTraversalCustomStack,
		AlignmentSceneTraversalCustomStack>
		m_impl;
};

/** \brief A traversal object for finding the any hit with hiprtScene.
 *
 * \tparam hiprtStack A custom stack.
 */
template <typename hiprtStack, typename hiprtInstanceStack>
class hiprtSceneTraversalAnyHitCustomStack
{
  public:
	HIPRT_DEVICE hiprtSceneTraversalAnyHitCustomStack(
		hiprtScene			scene,
		const hiprtRay&		ray,
		hiprtStack&			stack,
		hiprtInstanceStack& instanceStack,
		hiprtRayMask		mask	  = hiprtFullRayMask,
		hiprtTraversalHint	hint	  = hiprtTraversalHintDefault,
		void*				payload	  = nullptr,
		hiprtFuncTable		funcTable = nullptr,
		uint32_t			rayType	  = 0,
		float				time	  = 0.0f );
	HIPRT_DEVICE hiprtHit			 getNextHit();
	HIPRT_DEVICE hiprtTraversalState getCurrentState();

  private:
	hiprtPimpl<
		hiprtSceneTraversalCustomStack_impl<hiprtStack, hiprtInstanceStack, hiprtTraversalTerminateAtAnyHit>,
		SizeSceneTraversalCustomStack,
		AlignmentSceneTraversalCustomStack>
		m_impl;
};

/** \brief Returns the object to world transformation for a given instance and time in the form of the SRT frame.
 *
 * \param scene A scene.
 * \param instanceID Instance ID.
 * \param time The time.
 */
HIPRT_DEVICE hiprtFrameSRT hiprtGetObjectToWorldFrameSRT( hiprtScene scene, uint32_t instanceID, float time = 0.0f );

/** \brief Returns the world to object transformation for a given instance and time in the form of the SRT frame.
 *
 * \param scene A scene.
 * \param instanceID Instance ID.
 * \param time The time.
 */
HIPRT_DEVICE hiprtFrameSRT hiprtGetWorldToObjectFrameSRT( hiprtScene scene, uint32_t instanceID, float time = 0.0f );

/** \brief Returns the object to world transformation for a given instance and time in the form of the matrix.
 *
 * \param scene A scene.
 * \param instanceID Instance ID.
 * \param time The time.
 */
HIPRT_DEVICE hiprtFrameMatrix hiprtGetObjectToWorldFrameMatrix( hiprtScene scene, uint32_t instanceID, float time = 0.0f );

/** \brief Returns the world to object transformation for a given instance and time in the form of the matrix.
 *
 * \param scene A scene.
 * \param instanceID Instance ID.
 * \param time The time.
 */
HIPRT_DEVICE hiprtFrameMatrix hiprtGetWorldToObjectFrameMatrix( hiprtScene scene, uint32_t instanceID, float time = 0.0f );

/** \brief Transforms normal from object space to world space.
 *
 * \param normal A normal in object space.
 * \param scene A scene.
 * \param instanceID Instance ID.
 * \param time The time.
 */
HIPRT_DEVICE hiprtFloat3
hiprtNormalObjectToWorld( hiprtFloat3 normal, hiprtScene scene, uint32_t instanceID, float time = 0.0f );

/** \brief Returns the object to world transformation for a given instance and time in the form of the SRT frame.
 *
 * \param scene A scene.
 * \param instanceIDs Instance IDs (multi-level instancing).
 * \param time The time.
 */
HIPRT_DEVICE hiprtFrameSRT
hiprtGetObjectToWorldFrameSRT( hiprtScene scene, const uint32_t ( &instanceIDs )[hiprtMaxInstanceLevels], float time = 0.0f );

/** \brief Returns the world to object transformation for a given instance and time in the form of the SRT frame.
 *
 * \param scene A scene.
 * \param instanceIDs Instance IDs (multi-level instancing).
 * \param time The time.
 */
HIPRT_DEVICE hiprtFrameSRT
hiprtGetWorldToObjectFrameSRT( hiprtScene scene, const uint32_t ( &instanceIDs )[hiprtMaxInstanceLevels], float time = 0.0f );

/** \brief Returns the object to world transformation for a given instance and time in the form of the matrix.
 *
 * \param scene A scene.
 * \param instanceIDs Instance IDs (multi-level instancing).
 * \param time The time.
 */
HIPRT_DEVICE hiprtFrameMatrix hiprtGetObjectToWorldFrameMatrix(
	hiprtScene scene, const uint32_t ( &instanceIDs )[hiprtMaxInstanceLevels], float time = 0.0f );

/** \brief Returns the world to object transformation for a given instance and time in the form of the matrix.
 *
 * \param scene A scene.
 * \param instanceIDs Instance IDs (multi-level instancing).
 * \param time The time.
 */
HIPRT_DEVICE hiprtFrameMatrix hiprtGetWorldToObjectFrameMatrix(
	hiprtScene scene, const uint32_t ( &instanceIDs )[hiprtMaxInstanceLevels], float time = 0.0f );

/** \brief Transforms normal from object space to world space.
 *
 * \param normal A normal in object space.
 * \param scene A scene.
 * \param instanceIDs Instance IDs (multi-level instancing).
 * \param time The time.
 */
HIPRT_DEVICE hiprtFloat3 hiprtNormalObjectToWorld(
	hiprtFloat3 normal, hiprtScene scene, const uint32_t ( &instanceIDs )[hiprtMaxInstanceLevels], float time = 0.0f );
