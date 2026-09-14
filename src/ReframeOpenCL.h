#pragma once

struct ReframeCLParams {
  float invMVP[16];
  float rot3[9];
  float viewport[4];
  float borderRGB[3];
  float vScale, vOffset;
  int borderMode;
  int useCylinder;
};

// Enqueues onto the host-supplied queue; src/dst are the host's cl_mem buffers.
// Returns false if the program or kernels could not be built.
bool reframeRunOpenCL(void *cmdQueue, const void *srcBuf, void *dstBuf, int srcW,
                      int srcH, int dstW, int dstH, const ReframeCLParams &p);
