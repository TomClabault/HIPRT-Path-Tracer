/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_LIGHT_TREE_SG_OPTIONS_H
#define HOST_DEVICE_COMMON_LIGHT_TREE_SG_OPTIONS_H

#include "HostDeviceCommon/KernelOptions/Common.h"

 /**
  * Options are defined in a #ifndef __KERNELCC__ block because:
  *     - If they were not, the would be defined on the GPU side. However, the -D <macro>=<value> compiler option
  *             cannot override a #define statement. This means that if the #define statement are encountered by the compiler,
  *             we cannot modify the value of the macros anymore with the -D option which means no run-time switching / experimenting :(
  * - The CPU still needs the options to be able to compile the code so here they are, in a CPU-only block
  */
#ifndef __KERNELCC__

  /**
   * If true, the specular/metallic/coat part of the material will be taken
   * into account when computing light tree node importances during sampling.
   *
   * This improves light sampling quality on specular surfaces
   */
#define LightTreeSGDoSpecularImportance KERNEL_OPTION_TRUE


#endif // #ifndef __KERNELCC__

#endif
