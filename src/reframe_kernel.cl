#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
// Bilinear texture sampling (float4 RGBA, clamped)
inline float4 tex2D_bilinear(__global const float* src, int srcW, int srcH, float u, float v)
{
    u = fmax(0.0f, fmin(1.0f, u));
    v = fmax(0.0f, fmin(1.0f, v));
    float fx = u * (float)(srcW - 1);
    float fy = v * (float)(srcH - 1);
    int x0 = (int)floor(fx);
    int y0 = (int)floor(fy);
    int x1 = min(x0 + 1, srcW - 1);
    int y1 = min(y0 + 1, srcH - 1);
    float wx = fx - (float)x0;
    float wy = fy - (float)y0;
    int idx00 = (y0 * srcW + x0) * 4;
    int idx10 = (y0 * srcW + x1) * 4;
    int idx01 = (y1 * srcW + x0) * 4;
    int idx11 = (y1 * srcW + x1) * 4;
    float w00 = (1.0f - wx) * (1.0f - wy);
    float w10 = wx * (1.0f - wy);
    float w01 = (1.0f - wx) * wy;
    float w11 = wx * wy;
    float4 r;
    r.x = w00*src[idx00+0] + w10*src[idx10+0] + w01*src[idx01+0] + w11*src[idx11+0];
    r.y = w00*src[idx00+1] + w10*src[idx10+1] + w01*src[idx01+1] + w11*src[idx11+1];
    r.z = w00*src[idx00+2] + w10*src[idx10+2] + w01*src[idx01+2] + w11*src[idx11+2];
    r.w = w00*src[idx00+3] + w10*src[idx10+3] + w01*src[idx01+3] + w11*src[idx11+3];
    return r;
}
// 4x4 matrix * vec4 (column-major, passed as 16 floats for better performance)
inline float4 mat4_mul_vec4(float M0, float M1, float M2, float M3,
                            float M4, float M5, float M6, float M7,
                            float M8, float M9, float M10, float M11,
                            float M12, float M13, float M14, float M15,
                            float4 v)
{
    float4 r;
    r.x = M0*v.x + M4*v.y + M8*v.z  + M12*v.w;
    r.y = M1*v.x + M5*v.y + M9*v.z  + M13*v.w;
    r.z = M2*v.x + M6*v.y + M10*v.z + M14*v.w;
    r.w = M3*v.x + M7*v.y + M11*v.z + M15*v.w;
    return r;
}
// 3x3 matrix * vec3 (column-major, passed as 9 floats)
inline float3 mat3_mul_vec3(float M0, float M1, float M2,
                            float M3, float M4, float M5,
                            float M6, float M7, float M8,
                            float3 v)
{
    float3 r;
    r.x = M0*v.x + M3*v.y + M6*v.z;
    r.y = M1*v.x + M4*v.y + M7*v.z;
    r.z = M2*v.x + M5*v.y + M8*v.z;
    return r;
}
__kernel void PanoSphereKernel(
    __global const float* src,
    __global float* dst,
    int srcW, int srcH,
    int dstW, int dstH,
    float vpX, float vpY, float vpW, float vpH,
    float M0, float M1, float M2, float M3,
    float M4, float M5, float M6, float M7,
    float M8, float M9, float M10, float M11,
    float M12, float M13, float M14, float M15,
    int borderMode,
    float borderR, float borderG, float borderB,
    float vScale, float vOffset)
{
    int px = get_global_id(0);
    int py = get_global_id(1);
    if (px >= dstW || py >= dstH) return;
    int dstIdx = (py * dstW + px) * 4;
    float fx = (float)px;
    float fy = (float)py;
    bool insideVP = (fx >= vpX && fx < vpX + vpW &&
                     fy >= vpY && fy < vpY + vpH);
    if (!insideVP) {
        if (borderMode == 1) {
            dst[dstIdx+0] = borderR; dst[dstIdx+1] = borderG;
            dst[dstIdx+2] = borderB; dst[dstIdx+3] = 1.0f;
        } else {
            dst[dstIdx+0] = 0.0f; dst[dstIdx+1] = 0.0f;
            dst[dstIdx+2] = 0.0f; dst[dstIdx+3] = 0.0f;
        }
        return;
    }
    // NDC with Y-flip to match Metal's coordinate system
    float ndcX =  (((fx - vpX) / vpW) * 2.0f - 1.0f);
    float ndcY = -(((fy - vpY) / vpH) * 2.0f - 1.0f);
    // Unproject via precomputed invMVP (passed directly as parameters)
    float4 nearPt = mat4_mul_vec4(M0,M1,M2,M3,M4,M5,M6,M7,M8,M9,M10,M11,M12,M13,M14,M15,
                                  (float4)(ndcX, ndcY, -1.0f, 1.0f));
    float4 farPt  = mat4_mul_vec4(M0,M1,M2,M3,M4,M5,M6,M7,M8,M9,M10,M11,M12,M13,M14,M15,
                                  (float4)(ndcX, ndcY,  1.0f, 1.0f));
    float3 near3 = (float3)(nearPt.x/nearPt.w, nearPt.y/nearPt.w, nearPt.z/nearPt.w);
    float3 far3  = (float3)(farPt.x/farPt.w,   farPt.y/farPt.w,   farPt.z/farPt.w);
    float3 rayDir = (float3)(far3.x-near3.x, far3.y-near3.y, far3.z-near3.z);
    float len = sqrt(rayDir.x*rayDir.x + rayDir.y*rayDir.y + rayDir.z*rayDir.z);
    if (len < 1e-10f) {
        dst[dstIdx+0] = 0; dst[dstIdx+1] = 0; dst[dstIdx+2] = 0; dst[dstIdx+3] = 0;
        return;
    }
    float invLen = 1.0f / len;
    rayDir.x *= invLen; rayDir.y *= invLen; rayDir.z *= invLen;
    // Ray-sphere intersection (a=1 since rayDir normalized)
    float b = 2.0f * (near3.x*rayDir.x + near3.y*rayDir.y + near3.z*rayDir.z);
    float c = near3.x*near3.x + near3.y*near3.y + near3.z*near3.z - 1.0f;
    float disc = b*b - 4.0f*c;
    if (disc < 0.0f) {
        if (borderMode == 1) {
            dst[dstIdx+0] = borderR; dst[dstIdx+1] = borderG;
            dst[dstIdx+2] = borderB; dst[dstIdx+3] = 1.0f;
        } else {
            dst[dstIdx+0] = 0; dst[dstIdx+1] = 0; dst[dstIdx+2] = 0; dst[dstIdx+3] = 0;
        }
        return;
    }
    float sqrtDisc = sqrt(disc);
    float t = (-b - sqrtDisc) * 0.5f;
    if (t < 0.0f) t = (-b + sqrtDisc) * 0.5f;
    float3 hitPt = (float3)(near3.x + t*rayDir.x, near3.y + t*rayDir.y, near3.z + t*rayDir.z);
    // Convert hit point to equirectangular UV
    float theta = acos(fmax(-1.0f, fmin(1.0f, hitPt.y)));
    float phi = atan2(hitPt.x, hitPt.z);
    if (phi < 0.0f) phi += 2.0f * (float)M_PI;
    float u = phi / (2.0f * (float)M_PI);
    float v = 1.0f - theta / (float)M_PI;
    // Apply PanoSamplingParams
    v = vOffset + v * vScale;
    float4 color = tex2D_bilinear(src, srcW, srcH, u, v);
    dst[dstIdx+0] = color.x;
    dst[dstIdx+1] = color.y;
    dst[dstIdx+2] = color.z;
    dst[dstIdx+3] = color.w;
}
__kernel void PanoCylinderKernel(
    __global const float* src,
    __global float* dst,
    int srcW, int srcH,
    int dstW, int dstH,
    float vpX, float vpY, float vpW, float vpH,
    float R0, float R1, float R2,
    float R3, float R4, float R5,
    float R6, float R7, float R8,
    float vScale, float vOffset)
{
    int px = get_global_id(0);
    int py = get_global_id(1);
    if (px >= dstW || py >= dstH) return;
    int dstIdx = (py * dstW + px) * 4;
    float fx = (float)px;
    float fy = (float)py;
    bool insideVP = (fx >= vpX && fx < vpX + vpW &&
                     fy >= vpY && fy < vpY + vpH);
    if (!insideVP) {
        dst[dstIdx+0] = 0; dst[dstIdx+1] = 0; dst[dstIdx+2] = 0; dst[dstIdx+3] = 1.0f;
        return;
    }
    // Compute texCoord with Y-flip for Metal coordinate system
    float u = (fx - vpX) / vpW;
    float v = 1.0f - (fy - vpY) / vpH;
    // sphereRotate
    float theta_raw = v * (float)M_PI;
    float sinTheta_raw = sin(theta_raw);
    float phi_raw = 2.0f * (float)M_PI - u * 2.0f * (float)M_PI;
    float3 posSphere = (float3)(
        sinTheta_raw * cos(phi_raw),
        sinTheta_raw * sin(phi_raw),
        cos(theta_raw)
    );
    // Apply 3x3 rotation (passed directly as parameters, no redundant normalization)
    float3 rotated = mat3_mul_vec3(R0,R1,R2,R3,R4,R5,R6,R7,R8, posSphere);
    float theta = acos(fmax(-1.0f, fmin(1.0f, rotated.z)));
    float phi = atan2(rotated.y, rotated.x) + (float)M_PI;
    if (phi < 0.0f) phi += 2.0f * (float)M_PI;
    else if (phi > 2.0f * (float)M_PI) phi -= 2.0f * (float)M_PI;
    float su = phi / (2.0f * (float)M_PI);
    float sv = theta / (float)M_PI;
    sv = vOffset + sv * vScale;
    float4 color = tex2D_bilinear(src, srcW, srcH, su, sv);
    dst[dstIdx+0] = color.x;
    dst[dstIdx+1] = color.y;
    dst[dstIdx+2] = color.z;
    dst[dstIdx+3] = color.w;
}
