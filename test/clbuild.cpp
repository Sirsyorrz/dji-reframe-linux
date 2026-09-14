#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <cstdio>
#include <vector>
static const char* src =
#include "reframe_kernel.inc"
;
int main(){ cl_platform_id pf; cl_uint n; clGetPlatformIDs(1,&pf,&n); cl_device_id d; if(clGetDeviceIDs(pf,CL_DEVICE_TYPE_GPU,1,&d,&n)!=CL_SUCCESS){puts("no gpu");return 1;}
char nm[256]; clGetDeviceInfo(d,CL_DEVICE_NAME,256,nm,0); printf("device: %s\\n",nm);
cl_int e; cl_context c=clCreateContext(0,1,&d,0,0,&e); cl_program p=clCreateProgramWithSource(c,1,&src,0,&e);
e=clBuildProgram(p,1,&d,"-cl-fast-relaxed-math",0,0);
size_t ls=0; clGetProgramBuildInfo(p,d,CL_PROGRAM_BUILD_LOG,0,0,&ls); std::vector<char> log(ls+1); clGetProgramBuildInfo(p,d,CL_PROGRAM_BUILD_LOG,ls,log.data(),0);
if(e!=CL_SUCCESS){printf("BUILD FAILED %d\\n%s\\n",e,log.data());return 1;}
puts("build OK");
for(const char*k:{"PanoSphereKernel","PanoCylinderKernel"}){cl_kernel kk=clCreateKernel(p,k,&e); cl_uint na=0; clGetKernelInfo(kk,CL_KERNEL_NUM_ARGS,sizeof(na),&na,0); printf("  %s: %s, args=%u\\n",k,e==CL_SUCCESS?"ok":"FAIL",na);} return 0;}
