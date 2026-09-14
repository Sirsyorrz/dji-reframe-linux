#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <cstdio>
#include <vector>
#include <cmath>
static const char* ksrc =
#include "reframe_kernel.inc"
;
int main(){
  cl_platform_id pf; cl_uint n; clGetPlatformIDs(1,&pf,&n);
  cl_device_id d; if(clGetDeviceIDs(pf,CL_DEVICE_TYPE_GPU,1,&d,&n)!=CL_SUCCESS){puts("no gpu");return 1;}
  cl_int e; cl_context c=clCreateContext(0,1,&d,0,0,&e);
  cl_command_queue q=clCreateCommandQueue(c,d,0,&e);
  cl_program p=clCreateProgramWithSource(c,1,&ksrc,0,&e);
  if(clBuildProgram(p,1,&d,"",0,0)!=CL_SUCCESS){puts("build fail");return 1;}
  cl_kernel k=clCreateKernel(p,"PanoSphereKernel",&e);

  int sw=64,sh=32,dw=200,dh=100;
  std::vector<float> src(sw*sh*4,1.0f), dst(dw*dh*4,-1.0f);
  cl_mem sb=clCreateBuffer(c,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,src.size()*4,src.data(),&e);
  cl_mem db=clCreateBuffer(c,CL_MEM_WRITE_ONLY,dst.size()*4,0,&e);

  // Letterbox: 100-wide viewport centred in a 200-wide output.
  float vpX=50.f,vpY=0.f,vpW=100.f,vpH=100.f;
  float M[16]={1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
  int borderMode=1; float br=0.f,bg=0.f,bb=0.f, vs=1.f,vo=0.f;
  cl_uint a=0;
  clSetKernelArg(k,a++,sizeof(sb),&sb); clSetKernelArg(k,a++,sizeof(db),&db);
  clSetKernelArg(k,a++,4,&sw); clSetKernelArg(k,a++,4,&sh);
  clSetKernelArg(k,a++,4,&dw); clSetKernelArg(k,a++,4,&dh);
  clSetKernelArg(k,a++,4,&vpX); clSetKernelArg(k,a++,4,&vpY);
  clSetKernelArg(k,a++,4,&vpW); clSetKernelArg(k,a++,4,&vpH);
  for(int i=0;i<16;i++) clSetKernelArg(k,a++,4,&M[i]);
  clSetKernelArg(k,a++,4,&borderMode);
  clSetKernelArg(k,a++,4,&br); clSetKernelArg(k,a++,4,&bg); clSetKernelArg(k,a++,4,&bb);
  clSetKernelArg(k,a++,4,&vs); clSetKernelArg(k,a++,4,&vo);
  printf("args set: %u (expect 32)\n",a);
  size_t g[2]={(size_t)dw,(size_t)dh};
  if(clEnqueueNDRangeKernel(q,k,2,0,g,0,0,0,0)!=CL_SUCCESS){puts("enqueue fail");return 1;}
  clFinish(q);
  clEnqueueReadBuffer(q,db,CL_TRUE,0,dst.size()*4,dst.data(),0,0,0);

  auto px=[&](int x,int y){ return &dst[(y*dw+x)*4]; };
  printf("outside-left  (x=10):  rgb=%.2f %.2f %.2f a=%.2f\n", px(10,50)[0],px(10,50)[1],px(10,50)[2],px(10,50)[3]);
  printf("inside-centre (x=100): rgb=%.2f %.2f %.2f a=%.2f\n", px(100,50)[0],px(100,50)[1],px(100,50)[2],px(100,50)[3]);
  printf("outside-right (x=190): rgb=%.2f %.2f %.2f a=%.2f\n", px(190,50)[0],px(190,50)[1],px(190,50)[2],px(190,50)[3]);
  return 0;
}
