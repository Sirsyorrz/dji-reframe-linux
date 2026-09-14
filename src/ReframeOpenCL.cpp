#define CL_TARGET_OPENCL_VERSION 120

#include "ReframeOpenCL.h"

#include <CL/cl.h>

#include <map>
#include <mutex>

namespace {

const char *kKernelSource =
#include "reframe_kernel.inc"
    ;

struct QueuePrograms {
  cl_program program = nullptr;
  cl_kernel sphere = nullptr;
  cl_kernel cylinder = nullptr;
};

std::mutex gMutex;
std::map<cl_command_queue, QueuePrograms> gCache;

// Resolve hands out one queue per GPU, so programs are cached per queue rather
// than rebuilt every frame.
QueuePrograms *programsFor(cl_command_queue queue) {
  auto it = gCache.find(queue);
  if (it != gCache.end()) return &it->second;

  cl_context context = nullptr;
  if (clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(context), &context,
                            nullptr) != CL_SUCCESS)
    return nullptr;
  cl_device_id device = nullptr;
  if (clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(device), &device,
                            nullptr) != CL_SUCCESS)
    return nullptr;

  cl_int err = CL_SUCCESS;
  cl_program program =
      clCreateProgramWithSource(context, 1, &kKernelSource, nullptr, &err);
  if (err != CL_SUCCESS || program == nullptr) return nullptr;

  if (clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math", nullptr,
                     nullptr) != CL_SUCCESS) {
    clReleaseProgram(program);
    return nullptr;
  }

  QueuePrograms qp;
  qp.program = program;
  qp.sphere = clCreateKernel(program, "PanoSphereKernel", &err);
  if (err != CL_SUCCESS) {
    clReleaseProgram(program);
    return nullptr;
  }
  qp.cylinder = clCreateKernel(program, "PanoCylinderKernel", &err);
  if (err != CL_SUCCESS) {
    clReleaseKernel(qp.sphere);
    clReleaseProgram(program);
    return nullptr;
  }

  return &(gCache[queue] = qp);
}

struct ArgSetter {
  cl_kernel k;
  cl_uint i = 0;
  bool ok = true;

  template <typename T>
  void operator()(const T &v) {
    if (ok) ok = clSetKernelArg(k, i++, sizeof(T), &v) == CL_SUCCESS;
  }
};

}  // namespace

bool reframeRunOpenCL(void *cmdQueue, const void *srcBuf, void *dstBuf, int srcW,
                      int srcH, int dstW, int dstH, const ReframeCLParams &p) {
  auto queue = static_cast<cl_command_queue>(cmdQueue);
  if (queue == nullptr || srcBuf == nullptr || dstBuf == nullptr) return false;

  std::lock_guard<std::mutex> lock(gMutex);
  QueuePrograms *qp = programsFor(queue);
  if (qp == nullptr) return false;

  const bool cylinder = p.useCylinder != 0;
  cl_kernel k = cylinder ? qp->cylinder : qp->sphere;

  auto src = static_cast<cl_mem>(const_cast<void *>(srcBuf));
  auto dst = static_cast<cl_mem>(dstBuf);

  ArgSetter arg{k};
  arg(src);
  arg(dst);
  arg(srcW);
  arg(srcH);
  arg(dstW);
  arg(dstH);
  arg(p.viewport[0]);
  arg(p.viewport[1]);
  arg(p.viewport[2]);
  arg(p.viewport[3]);

  if (cylinder) {
    for (int i = 0; i < 9; ++i) arg(p.rot3[i]);
  } else {
    for (int i = 0; i < 16; ++i) arg(p.invMVP[i]);
    arg(p.borderMode);
    arg(p.borderRGB[0]);
    arg(p.borderRGB[1]);
    arg(p.borderRGB[2]);
  }
  arg(p.vScale);
  arg(p.vOffset);
  if (!arg.ok) return false;

  const size_t local[2] = {16, 16};
  const size_t global[2] = {((dstW + local[0] - 1) / local[0]) * local[0],
                            ((dstH + local[1] - 1) / local[1]) * local[1]};

  return clEnqueueNDRangeKernel(queue, k, 2, nullptr, global, local, 0, nullptr,
                                nullptr) == CL_SUCCESS;
}
