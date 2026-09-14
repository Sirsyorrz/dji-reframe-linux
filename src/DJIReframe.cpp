#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include <GL/gl.h>

#include "ofxsImageEffect.h"
#include "ofxsInteract.h"
#include "ofxsMultiThread.h"
#include "ofxsProcessing.h"

#include "ReframeOpenCL.h"

#define kPluginName "DJI Reframe"
#define kPluginGrouping "DJI"
#define kPluginIdentifier "com.pano360.reframe"
#define kPluginVersionMajor 1
#define kPluginVersionMinor 0

namespace {

constexpr float kPi = 3.14159265358979323846f;

struct Vec3 {
  float x, y, z;
};

struct Mat4 {
  float m[16];  // column-major, m[col*4 + row]
};

Mat4 identity() {
  Mat4 r{};
  r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f;
  return r;
}

Mat4 multiply(const Mat4 &a, const Mat4 &b) {
  Mat4 r{};
  for (int c = 0; c < 4; ++c)
    for (int row = 0; row < 4; ++row) {
      float s = 0.0f;
      for (int k = 0; k < 4; ++k) s += a.m[k * 4 + row] * b.m[c * 4 + k];
      r.m[c * 4 + row] = s;
    }
  return r;
}

Mat4 perspective(float fovYRad, float aspect, float zn, float zf) {
  Mat4 r{};
  const float f = 1.0f / std::tan(fovYRad * 0.5f);
  r.m[0] = f / aspect;
  r.m[5] = f;
  r.m[10] = (zf + zn) / (zn - zf);
  r.m[11] = -1.0f;
  r.m[14] = (2.0f * zf * zn) / (zn - zf);
  return r;
}

Mat4 rotationX(float a) {
  Mat4 r = identity();
  const float c = std::cos(a), s = std::sin(a);
  r.m[5] = c;
  r.m[6] = s;
  r.m[9] = -s;
  r.m[10] = c;
  return r;
}

Mat4 rotationY(float a) {
  Mat4 r = identity();
  const float c = std::cos(a), s = std::sin(a);
  r.m[0] = c;
  r.m[2] = -s;
  r.m[8] = s;
  r.m[10] = c;
  return r;
}

Mat4 rotationZ(float a) {
  Mat4 r = identity();
  const float c = std::cos(a), s = std::sin(a);
  r.m[0] = c;
  r.m[1] = s;
  r.m[4] = -s;
  r.m[5] = c;
  return r;
}

Mat4 translation(float x, float y, float z) {
  Mat4 r = identity();
  r.m[12] = x;
  r.m[13] = y;
  r.m[14] = z;
  return r;
}

bool invert(const Mat4 &src, Mat4 &out) {
  const float *m = src.m;
  float inv[16];

  inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] +
           m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
  inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] -
           m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
  inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] +
           m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
  inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] -
            m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
  inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] -
           m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
  inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] +
           m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
  inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] -
           m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
  inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] +
            m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
  inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] +
           m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
  inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] -
           m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
  inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] +
            m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
  inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] -
            m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
  inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] -
           m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
  inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] +
           m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
  inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] -
            m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
  inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] +
            m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];

  float det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
  if (std::fabs(det) < 1e-20f) return false;
  det = 1.0f / det;
  for (int i = 0; i < 16; ++i) out.m[i] = inv[i] * det;
  return true;
}

// Matches the reference kernel exactly: row 0 is v=0. OFX buffers are bottom-up,
// which already puts the pano's north pole at v=1, so no flip belongs here.
void sampleBilinear(const float *src, int srcW, int srcH, int srcRowFloats,
                    float u, float v, float *out) {
  u = std::clamp(u, 0.0f, 1.0f);
  v = std::clamp(v, 0.0f, 1.0f);
  const float fx = u * static_cast<float>(srcW - 1);
  const float fy = v * static_cast<float>(srcH - 1);
  const int x0 = static_cast<int>(std::floor(fx));
  const int y0 = static_cast<int>(std::floor(fy));
  const int x1 = std::min(x0 + 1, srcW - 1);
  const int y1 = std::min(y0 + 1, srcH - 1);
  const float wx = fx - static_cast<float>(x0);
  const float wy = fy - static_cast<float>(y0);

  const float *r00 = src + y0 * srcRowFloats + x0 * 4;
  const float *r10 = src + y0 * srcRowFloats + x1 * 4;
  const float *r01 = src + y1 * srcRowFloats + x0 * 4;
  const float *r11 = src + y1 * srcRowFloats + x1 * 4;

  const float w00 = (1.0f - wx) * (1.0f - wy);
  const float w10 = wx * (1.0f - wy);
  const float w01 = (1.0f - wx) * wy;
  const float w11 = wx * wy;

  for (int c = 0; c < 4; ++c)
    out[c] = w00 * r00[c] + w10 * r10[c] + w01 * r01[c] + w11 * r11[c];
}

enum Projection {
  kProjWide = 0,
  kProjCrystalBall,
  kProjTinyPlanet,
  kProjUltraWide,
  kProjDewarp,
};

enum BorderMode {
  kBorderTransparent = 0,
  kBorderColor,
  kBorderMirror,
};

struct ResolutionPreset {
  const char *label;
  int w, h;
};

// Labels and order taken verbatim from the original plugin's string table.
const ResolutionPreset kResolutions[] = {
    {"Source Image", 0, 0},
    {"1080p 16:9(1920*1080)", 1920, 1080},
    {"1080p 9:16(1080*1920)", 1080, 1920},
    {"1080p 1:1 (1080*1080)", 1080, 1080},
    {"1080p 2.35:1(2538*1080)", 2538, 1080},
    {"1080p 4:3 (1440*1080)", 1440, 1080},
    {"1080p 3:4(1080*1440)", 1080, 1440},
    {"4k 16:9(3840*2160)", 3840, 2160},
    {"4k 9:16(2160*3840)", 2160, 3840},
    {"4k 1:1(2160*2160)", 2160, 2160},
    {"4k 2.35:1(3840*1634)", 3840, 1634},
    {"Custom", -1, -1},
};
constexpr int kNumResolutions =
    static_cast<int>(sizeof(kResolutions) / sizeof(kResolutions[0]));

struct RenderParams {
  Mat4 invMVP;
  float rot3[9];
  Projection projection;
  BorderMode borderMode;
  float borderRGB[3];
  float vScale, vOffset;
};

}  // namespace

class ReframeProcessor : public OFX::ImageProcessor {
 public:
  explicit ReframeProcessor(OFX::ImageEffect &instance)
      : OFX::ImageProcessor(instance) {}

  void setSrcImg(OFX::Image *v) { srcImg_ = v; }
  void setParams(const RenderParams &p) { params_ = p; }

  void multiThreadProcessImages(OfxRectI window) override {
    const OfxRectI srcBounds = srcImg_->getBounds();
    const int srcW = srcBounds.x2 - srcBounds.x1;
    const int srcH = srcBounds.y2 - srcBounds.y1;
    const int srcRowFloats = srcImg_->getRowBytes() / sizeof(float);
    const float *srcBase =
        static_cast<const float *>(srcImg_->getPixelAddress(srcBounds.x1, srcBounds.y1));
    if (srcW < 2 || srcH < 2 || srcBase == nullptr) return;

    const OfxRectI dstBounds = _dstImg->getBounds();
    const float dstW = static_cast<float>(dstBounds.x2 - dstBounds.x1);
    const float dstH = static_cast<float>(dstBounds.y2 - dstBounds.y1);

    for (int y = window.y1; y < window.y2; ++y) {
      if (_effect.abort()) break;
      float *dst = static_cast<float *>(_dstImg->getPixelAddress(window.x1, y));
      if (dst == nullptr) continue;

      // NDC y is already bottom-up in OFX, matching the kernel's Y-flip.
      const float ndcY =
          ((static_cast<float>(y - dstBounds.y1) + 0.5f) / dstH) * 2.0f - 1.0f;

      for (int x = window.x1; x < window.x2; ++x, dst += 4) {
        const float ndcX =
            ((static_cast<float>(x - dstBounds.x1) + 0.5f) / dstW) * 2.0f - 1.0f;

        bool hit = false;
        float u = 0.0f, v = 0.0f;
        if (params_.projection == kProjDewarp)
          hit = cylinderUV(ndcX, ndcY, u, v);
        else
          hit = sphereUV(ndcX, ndcY, u, v);

        if (!hit) {
          writeBorder(dst);
          continue;
        }
        v = params_.vOffset + v * params_.vScale;
        sampleBilinear(srcBase, srcW, srcH, srcRowFloats, u, v, dst);
      }
    }
  }

 private:
  void writeBorder(float *dst) const {
    if (params_.borderMode == kBorderColor) {
      dst[0] = params_.borderRGB[0];
      dst[1] = params_.borderRGB[1];
      dst[2] = params_.borderRGB[2];
      dst[3] = 1.0f;
    } else {
      dst[0] = dst[1] = dst[2] = dst[3] = 0.0f;
    }
  }

  bool sphereUV(float ndcX, float ndcY, float &u, float &v) const {
    const float *M = params_.invMVP.m;
    auto unproject = [&](float z, Vec3 &out) {
      const float px = M[0] * ndcX + M[4] * ndcY + M[8] * z + M[12];
      const float py = M[1] * ndcX + M[5] * ndcY + M[9] * z + M[13];
      const float pz = M[2] * ndcX + M[6] * ndcY + M[10] * z + M[14];
      const float pw = M[3] * ndcX + M[7] * ndcY + M[11] * z + M[15];
      if (std::fabs(pw) < 1e-20f) return false;
      out = {px / pw, py / pw, pz / pw};
      return true;
    };

    Vec3 nearPt, farPt;
    if (!unproject(-1.0f, nearPt) || !unproject(1.0f, farPt)) return false;

    Vec3 dir{farPt.x - nearPt.x, farPt.y - nearPt.y, farPt.z - nearPt.z};
    const float len = std::sqrt(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
    if (len < 1e-10f) return false;
    dir.x /= len;
    dir.y /= len;
    dir.z /= len;

    const float b = 2.0f * (nearPt.x * dir.x + nearPt.y * dir.y + nearPt.z * dir.z);
    const float c =
        nearPt.x * nearPt.x + nearPt.y * nearPt.y + nearPt.z * nearPt.z - 1.0f;
    const float disc = b * b - 4.0f * c;
    if (disc < 0.0f) return false;

    const float sq = std::sqrt(disc);
    float t = (-b - sq) * 0.5f;
    if (t < 0.0f) t = (-b + sq) * 0.5f;
    if (t < 0.0f) return false;

    const Vec3 hit{nearPt.x + t * dir.x, nearPt.y + t * dir.y, nearPt.z + t * dir.z};
    const float theta = std::acos(std::clamp(hit.y, -1.0f, 1.0f));
    float phi = std::atan2(hit.x, hit.z);
    if (phi < 0.0f) phi += 2.0f * kPi;
    u = phi / (2.0f * kPi);
    v = 1.0f - theta / kPi;
    return true;
  }

  bool cylinderUV(float ndcX, float ndcY, float &u, float &v) const {
    const float tu = (ndcX + 1.0f) * 0.5f;
    const float tv = (ndcY + 1.0f) * 0.5f;

    const float thetaRaw = tv * kPi;
    const float sinTheta = std::sin(thetaRaw);
    const float phiRaw = 2.0f * kPi - tu * 2.0f * kPi;
    const Vec3 p{sinTheta * std::cos(phiRaw), sinTheta * std::sin(phiRaw),
                 std::cos(thetaRaw)};

    const float *R = params_.rot3;
    const Vec3 r{R[0] * p.x + R[3] * p.y + R[6] * p.z,
                 R[1] * p.x + R[4] * p.y + R[7] * p.z,
                 R[2] * p.x + R[5] * p.y + R[8] * p.z};

    const float theta = std::acos(std::clamp(r.z, -1.0f, 1.0f));
    float phi = std::atan2(r.y, r.x) + kPi;
    if (phi < 0.0f)
      phi += 2.0f * kPi;
    else if (phi > 2.0f * kPi)
      phi -= 2.0f * kPi;

    u = phi / (2.0f * kPi);
    v = theta / kPi;
    return true;
  }

  OFX::Image *srcImg_ = nullptr;
  RenderParams params_{};
};

class ReframePlugin : public OFX::ImageEffect {
 public:
  explicit ReframePlugin(OfxImageEffectHandle handle) : OFX::ImageEffect(handle) {
    dstClip_ = fetchClip(kOfxImageEffectOutputClipName);
    srcClip_ = fetchClip(kOfxImageEffectSimpleSourceClipName);

    resolution_ = fetchChoiceParam("resolution");
    customWidth_ = fetchIntParam("custom_width");
    customHeight_ = fetchIntParam("custom_height");
    projection_ = fetchChoiceParam("camera_projection_type");
    pan_ = fetchDoubleParam("camera_pan");
    tilt_ = fetchDoubleParam("camera_tilt");
    roll_ = fetchDoubleParam("camera_rotate");
    fov_ = fetchDoubleParam("camera_fov_control");
    distortion_ = fetchDoubleParam("camera_distortion_control");
    srcYaw_ = fetchDoubleParam("source_file_yaw");
    srcPitch_ = fetchDoubleParam("source_file_pitch");
    srcRoll_ = fetchDoubleParam("source_file_roll");
    borderMode_ = fetchChoiceParam("border_mode");
    borderColor_ = fetchRGBParam("border_color");
    overlayColor_ = fetchRGBParam("overlay_color");
    reset_ = fetchPushButtonParam("reset_button");

    syncCustomEnabled();
  }

  bool getRegionOfDefinition(const OFX::RegionOfDefinitionArguments &args,
                             OfxRectD &rod) override {
    int w = 0, h = 0;
    if (!outputSize(args.time, w, h)) return false;
    rod.x1 = 0.0;
    rod.y1 = 0.0;
    rod.x2 = w * args.renderScale.x;
    rod.y2 = h * args.renderScale.y;
    return true;
  }

  void getClipPreferences(OFX::ClipPreferencesSetter &prefs) override {
    prefs.setOutputPremultiplication(OFX::eImageUnPreMultiplied);
  }

  OfxRectD getRegionOfDefinitionForOverlay(double time) const {
    return dstClip_->getRegionOfDefinition(time);
  }

  OFX::RGBParam *overlayColorParam() const { return overlayColor_; }
  OFX::DoubleParam *panParam() const { return pan_; }
  OFX::DoubleParam *tiltParam() const { return tilt_; }
  OFX::DoubleParam *fovParam() const { return fov_; }

  void render(const OFX::RenderArguments &args) override {
    std::unique_ptr<OFX::Image> dst(dstClip_->fetchImage(args.time));
    std::unique_ptr<OFX::Image> src(srcClip_->fetchImage(args.time));
    if (!dst || !src) return;

    if (args.isEnabledOpenCLRender && renderOpenCL(args, src.get(), dst.get()))
      return;

    if (dst->getPixelDepth() != OFX::eBitDepthFloat ||
        src->getPixelDepth() != OFX::eBitDepthFloat ||
        dst->getPixelComponents() != OFX::ePixelComponentRGBA ||
        src->getPixelComponents() != OFX::ePixelComponentRGBA) {
      OFX::throwSuiteStatusException(kOfxStatErrImageFormat);
    }

    ReframeProcessor proc(*this);
    proc.setDstImg(dst.get());
    proc.setSrcImg(src.get());
    proc.setRenderWindow(args.renderWindow);
    proc.setParams(buildParams(args));
    proc.process();
  }

  void changedParam(const OFX::InstanceChangedArgs &args,
                    const std::string &name) override {
    if (name == "resolution") {
      syncCustomEnabled();
      return;
    }
    if (name != "reset_button" || args.reason != OFX::eChangeUserEdit) return;
    pan_->setValue(0.0);
    tilt_->setValue(0.0);
    roll_->setValue(0.0);
    fov_->setValue(90.0);
    distortion_->setValue(0.0);
  }

 private:
  void syncCustomEnabled() {
    int idx = 0;
    resolution_->getValue(idx);
    const bool custom = idx == kNumResolutions - 1;
    customWidth_->setEnabled(custom);
    customHeight_->setEnabled(custom);
  }

  bool outputSize(double time, int &w, int &h) const {
    int idx = 0;
    resolution_->getValueAtTime(time, idx);
    idx = std::clamp(idx, 0, kNumResolutions - 1);
    if (idx == 0) return false;  // Source Image: inherit the input RoD.
    if (idx == kNumResolutions - 1) {
      customWidth_->getValueAtTime(time, w);
      customHeight_->getValueAtTime(time, h);
      w = std::max(w, 1);
      h = std::max(h, 1);
      return true;
    }
    w = kResolutions[idx].w;
    h = kResolutions[idx].h;
    return true;
  }

  bool renderOpenCL(const OFX::RenderArguments &args, OFX::Image *src,
                    OFX::Image *dst) const {
    const OfxRectI sb = src->getBounds();
    const OfxRectI db = dst->getBounds();
    const int srcW = sb.x2 - sb.x1, srcH = sb.y2 - sb.y1;
    const int dstW = db.x2 - db.x1, dstH = db.y2 - db.y1;
    if (srcW < 2 || srcH < 2 || dstW < 1 || dstH < 1) return false;

    const RenderParams p = buildParams(args);
    ReframeCLParams cl{};
    std::memcpy(cl.invMVP, p.invMVP.m, sizeof(cl.invMVP));
    std::memcpy(cl.rot3, p.rot3, sizeof(cl.rot3));
    std::memcpy(cl.borderRGB, p.borderRGB, sizeof(cl.borderRGB));
    cl.viewport[0] = 0.0f;
    cl.viewport[1] = 0.0f;
    cl.viewport[2] = static_cast<float>(dstW);
    cl.viewport[3] = static_cast<float>(dstH);
    cl.vScale = p.vScale;
    cl.vOffset = p.vOffset;
    cl.borderMode = static_cast<int>(p.borderMode);
    cl.useCylinder = p.projection == kProjDewarp ? 1 : 0;

    return reframeRunOpenCL(args.pOpenCLCmdQ, src->getPixelData(),
                            dst->getPixelData(), srcW, srcH, dstW, dstH, cl);
  }

  RenderParams buildParams(const OFX::RenderArguments &args) const {
    RenderParams p{};
    int projIdx = 0, borderIdx = 0;
    projection_->getValueAtTime(args.time, projIdx);
    borderMode_->getValueAtTime(args.time, borderIdx);
    p.projection = static_cast<Projection>(projIdx);
    p.borderMode = static_cast<BorderMode>(borderIdx);
    double br, bg, bb;
    borderColor_->getValueAtTime(args.time, br, bg, bb);
    p.borderRGB[0] = static_cast<float>(br);
    p.borderRGB[1] = static_cast<float>(bg);
    p.borderRGB[2] = static_cast<float>(bb);
    p.vScale = 1.0f;
    p.vOffset = 0.0f;

    const float deg = kPi / 180.0f;
    const float pan = static_cast<float>(pan_->getValueAtTime(args.time)) * deg;
    const float tilt = static_cast<float>(tilt_->getValueAtTime(args.time)) * deg;
    const float roll = static_cast<float>(roll_->getValueAtTime(args.time)) * deg;
    const float sYaw = static_cast<float>(srcYaw_->getValueAtTime(args.time)) * deg;
    const float sPitch = static_cast<float>(srcPitch_->getValueAtTime(args.time)) * deg;
    const float sRoll = static_cast<float>(srcRoll_->getValueAtTime(args.time)) * deg;
    const float fovDeg = static_cast<float>(fov_->getValueAtTime(args.time));
    const float distortion = static_cast<float>(distortion_->getValueAtTime(args.time));

    const Mat4 srcFix =
        multiply(multiply(rotationY(sYaw), rotationX(sPitch)), rotationZ(sRoll));
    const Mat4 view =
        multiply(multiply(rotationZ(roll), rotationX(tilt)), rotationY(pan));
    const Mat4 orient = multiply(view, srcFix);

    // Dewarp consumes the plain 3x3, everything else goes through the MVP path.
    const float *o = orient.m;
    const float r3[9] = {o[0], o[1], o[2], o[4], o[5], o[6], o[8], o[9], o[10]};
    std::memcpy(p.rot3, r3, sizeof(r3));

    // Pulling the eye back off the sphere centre is what produces the
    // crystal-ball / tiny-planet look; the ray-sphere hit does the rest.
    float eyeDist = 0.0f;
    float effectiveFov = fovDeg;
    switch (p.projection) {
      case kProjWide:
        break;
      case kProjUltraWide:
        effectiveFov = std::min(fovDeg * 1.6f, 170.0f);
        break;
      case kProjCrystalBall:
        eyeDist = 0.75f + 0.75f * distortion;
        effectiveFov = std::min(fovDeg, 140.0f);
        break;
      case kProjTinyPlanet:
        eyeDist = 1.9f + 1.5f * distortion;
        effectiveFov = std::min(fovDeg, 140.0f);
        break;
      case kProjDewarp:
        break;
    }

    const OfxRectI win = args.renderWindow;
    float aspect = 1.0f;
    if (win.y2 > win.y1)
      aspect = static_cast<float>(win.x2 - win.x1) /
               static_cast<float>(win.y2 - win.y1);
    aspect *= static_cast<float>(args.renderScale.x / args.renderScale.y);

    const float fovY = 2.0f * std::atan(std::tan(effectiveFov * deg * 0.5f) /
                                        std::max(aspect, 1e-3f));
    const Mat4 proj = perspective(std::clamp(fovY, 0.02f, 3.0f), aspect, 0.01f, 100.0f);
    const Mat4 mvp = multiply(proj, multiply(translation(0.0f, 0.0f, -eyeDist), orient));

    p.invMVP = identity();
    invert(mvp, p.invMVP);
    return p;
  }

  OFX::Clip *dstClip_ = nullptr;
  OFX::Clip *srcClip_ = nullptr;
  OFX::ChoiceParam *projection_ = nullptr;
  OFX::DoubleParam *pan_ = nullptr;
  OFX::DoubleParam *tilt_ = nullptr;
  OFX::DoubleParam *roll_ = nullptr;
  OFX::DoubleParam *fov_ = nullptr;
  OFX::DoubleParam *distortion_ = nullptr;
  OFX::DoubleParam *srcYaw_ = nullptr;
  OFX::DoubleParam *srcPitch_ = nullptr;
  OFX::DoubleParam *srcRoll_ = nullptr;
  OFX::ChoiceParam *borderMode_ = nullptr;
  OFX::RGBParam *borderColor_ = nullptr;
  OFX::RGBParam *overlayColor_ = nullptr;
  OFX::PushButtonParam *reset_ = nullptr;
  OFX::ChoiceParam *resolution_ = nullptr;
  OFX::IntParam *customWidth_ = nullptr;
  OFX::IntParam *customHeight_ = nullptr;
};

// Drag in the viewer to fly the camera; guide lines mark the thirds.
class ReframeOverlay : public OFX::OverlayInteract {
 public:
  ReframeOverlay(OfxInteractHandle handle, OFX::ImageEffect *effect)
      : OFX::OverlayInteract(handle),
        plugin_(dynamic_cast<ReframePlugin *>(effect)) {}

  bool draw(const OFX::DrawArgs &args) override {
    if (plugin_ == nullptr) return false;
    double r = 1.0, g = 1.0, b = 1.0;
    plugin_->overlayColorParam()->getValueAtTime(args.time, r, g, b);

    const OfxRectD rod = plugin_->getRegionOfDefinitionForOverlay(args.time);
    const double w = rod.x2 - rod.x1, h = rod.y2 - rod.y1;
    if (w <= 0.0 || h <= 0.0) return false;

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glColor4d(r, g, b, dragging_ ? 0.9 : 0.45);
    glLineWidth(1.0f);

    glBegin(GL_LINES);
    for (int i = 1; i < 3; ++i) {
      const double x = rod.x1 + w * i / 3.0;
      glVertex2d(x, rod.y1);
      glVertex2d(x, rod.y2);
      const double y = rod.y1 + h * i / 3.0;
      glVertex2d(rod.x1, y);
      glVertex2d(rod.x2, y);
    }
    glEnd();

    const double cx = rod.x1 + w * 0.5, cy = rod.y1 + h * 0.5;
    const double s = 0.02 * std::min(w, h);
    glBegin(GL_LINES);
    glVertex2d(cx - s, cy);
    glVertex2d(cx + s, cy);
    glVertex2d(cx, cy - s);
    glVertex2d(cx, cy + s);
    glEnd();
    glDisable(GL_BLEND);
    return true;
  }

  bool penDown(const OFX::PenArgs &args) override {
    if (plugin_ == nullptr) return false;
    dragging_ = true;
    last_ = args.penPosition;
    return true;
  }

  bool penMotion(const OFX::PenArgs &args) override {
    if (!dragging_ || plugin_ == nullptr) return false;

    const OfxRectD rod = plugin_->getRegionOfDefinitionForOverlay(args.time);
    const double w = rod.x2 - rod.x1, h = rod.y2 - rod.y1;
    if (w <= 0.0 || h <= 0.0) return false;

    // Scale by FOV so a drag moves the scene roughly under the cursor.
    const double fov = plugin_->fovParam()->getValueAtTime(args.time);
    const double dx = (args.penPosition.x - last_.x) / w * fov;
    const double dy = (args.penPosition.y - last_.y) / h * fov;
    last_ = args.penPosition;

    plugin_->panParam()->setValue(
        plugin_->panParam()->getValueAtTime(args.time) - dx);
    plugin_->tiltParam()->setValue(std::clamp(
        plugin_->tiltParam()->getValueAtTime(args.time) + dy, -90.0, 90.0));
    return true;
  }

  bool penUp(const OFX::PenArgs & /*args*/) override {
    if (!dragging_) return false;
    dragging_ = false;
    return true;
  }

 private:
  ReframePlugin *plugin_ = nullptr;
  bool dragging_ = false;
  OfxPointD last_{0.0, 0.0};
};

class ReframeOverlayDescriptor
    : public OFX::DefaultEffectOverlayDescriptor<ReframeOverlayDescriptor,
                                                 ReframeOverlay> {};

class ReframeFactory : public OFX::PluginFactoryHelper<ReframeFactory> {
 public:
  ReframeFactory()
      : OFX::PluginFactoryHelper<ReframeFactory>(kPluginIdentifier,
                                                 kPluginVersionMajor,
                                                 kPluginVersionMinor) {}

  void describe(OFX::ImageEffectDescriptor &desc) override {
    desc.setLabels(kPluginName, kPluginName, kPluginName);
    desc.setPluginGrouping(kPluginGrouping);
    desc.setPluginDescription("Lets you animate a virtual camera in your 360 footage.");
    desc.setOverlayInteractDescriptor(new ReframeOverlayDescriptor);
    desc.addSupportedContext(OFX::eContextFilter);
    desc.addSupportedContext(OFX::eContextGeneral);
    desc.addSupportedBitDepth(OFX::eBitDepthFloat);
    // Resolve skips CPU-only plugins on Linux with "not supported on this
    // platform", so GPU render support has to be advertised here.
    desc.setSupportsOpenCLBuffersRender(true);
    desc.setSingleInstance(false);
    desc.setHostFrameThreading(false);
    desc.setSupportsMultiResolution(true);
    desc.setSupportsTiles(true);
    desc.setTemporalClipAccess(false);
    desc.setRenderTwiceAlways(false);
    desc.setSupportsMultipleClipPARs(false);
  }

  void describeInContext(OFX::ImageEffectDescriptor &desc,
                         OFX::ContextEnum /*context*/) override {
    OFX::ClipDescriptor *src = desc.defineClip(kOfxImageEffectSimpleSourceClipName);
    src->addSupportedComponent(OFX::ePixelComponentRGBA);
    src->setTemporalClipAccess(false);
    src->setSupportsTiles(false);
    src->setIsMask(false);

    OFX::ClipDescriptor *dst = desc.defineClip(kOfxImageEffectOutputClipName);
    dst->addSupportedComponent(OFX::ePixelComponentRGBA);
    dst->setSupportsTiles(true);

    OFX::PageParamDescriptor *page = desc.definePageParam("Controls");

    auto group = [&](const char *name, const char *label, bool open) {
      OFX::GroupParamDescriptor *g = desc.defineGroupParam(name);
      g->setLabel(label);
      g->setOpen(open);
      page->addChild(*g);
      return g;
    };

    OFX::GroupParamDescriptor *gRes = group("Group_Resolution", "Resolution", true);
    {
      OFX::ChoiceParamDescriptor *p = desc.defineChoiceParam("resolution");
      p->setLabel("Resolution");
      for (const ResolutionPreset &r : kResolutions) p->appendOption(r.label);
      p->setDefault(0);
      p->setAnimates(false);
      p->setParent(*gRes);
    }
    {
      OFX::IntParamDescriptor *p = desc.defineIntParam("custom_width");
      p->setLabel("Custom Width");
      p->setHint("Custom output width");
      p->setDefault(7680);
      p->setRange(1, 16384);
      p->setDisplayRange(1, 16384);
      p->setAnimates(false);
      p->setParent(*gRes);
    }
    {
      OFX::IntParamDescriptor *p = desc.defineIntParam("custom_height");
      p->setLabel("Custom Height");
      p->setHint("Custom output height");
      p->setDefault(3840);
      p->setRange(1, 16384);
      p->setDisplayRange(1, 16384);
      p->setAnimates(false);
      p->setParent(*gRes);
    }

    OFX::GroupParamDescriptor *gCam =
        group("Group_Camera", "Camera Operation", true);
    {
      OFX::ChoiceParamDescriptor *p = desc.defineChoiceParam("camera_projection_type");
      p->setLabel("Projection");
      p->appendOption("Wide");
      p->appendOption("CrystalBall");
      p->appendOption("TinyPlanet");
      p->appendOption("UltraWide");
      p->appendOption("Dewarp");
      p->setDefault(kProjWide);
      p->setAnimates(true);
      p->setParent(*gCam);
    }

    auto angle = [&](const char *name, const char *label, double lo, double hi,
                     OFX::GroupParamDescriptor *g) {
      OFX::DoubleParamDescriptor *p = desc.defineDoubleParam(name);
      p->setLabel(label);
      p->setDoubleType(OFX::eDoubleTypeAngle);
      p->setDefault(0.0);
      p->setRange(lo, hi);
      p->setDisplayRange(lo, hi);
      p->setAnimates(true);
      p->setParent(*g);
    };

    angle("camera_pan", "Pan", -3600.0, 3600.0, gCam);
    angle("camera_tilt", "Tilt", -90.0, 90.0, gCam);
    angle("camera_rotate", "Rotate", -180.0, 180.0, gCam);

    {
      OFX::DoubleParamDescriptor *p = desc.defineDoubleParam("camera_fov_control");
      p->setLabel("FOV Control");
      p->setDefault(90.0);
      p->setRange(5.0, 170.0);
      p->setDisplayRange(5.0, 170.0);
      p->setAnimates(true);
      p->setParent(*gCam);
    }
    {
      OFX::DoubleParamDescriptor *p =
          desc.defineDoubleParam("camera_distortion_control");
      p->setLabel("Distortion Control");
      p->setDefault(0.0);
      p->setRange(-1.0, 1.0);
      p->setDisplayRange(-1.0, 1.0);
      p->setAnimates(true);
      p->setParent(*gCam);
    }
    {
      OFX::PushButtonParamDescriptor *p = desc.definePushButtonParam("reset_button");
      p->setLabel("Reset");
      p->setHint("Reset camera parameters");
      p->setParent(*gCam);
    }

    OFX::GroupParamDescriptor *gSrc = group("Group_SourceFile", "Source File", true);
    angle("source_file_yaw", "Pan", -180.0, 180.0, gSrc);
    angle("source_file_pitch", "Tilt", -180.0, 180.0, gSrc);
    angle("source_file_roll", "Rotate", -180.0, 180.0, gSrc);

    OFX::GroupParamDescriptor *gAdv = group("Group_Advanced", "Advanced", true);
    {
      OFX::GroupParamDescriptor *gBg = desc.defineGroupParam("Group_Background");
      gBg->setLabel("Background");
      gBg->setOpen(true);
      gBg->setParent(*gAdv);

      OFX::ChoiceParamDescriptor *p = desc.defineChoiceParam("border_mode");
      p->setLabel("Border Mode");
      p->appendOption("Transparent");
      p->appendOption("Color");
      p->appendOption("Mirror/Extend");
      p->setDefault(kBorderTransparent);
      p->setAnimates(false);
      p->setParent(*gBg);

      OFX::RGBParamDescriptor *c = desc.defineRGBParam("border_color");
      c->setLabel("Border Color");
      c->setDefault(0.0, 0.0, 0.0);
      c->setAnimates(true);
      c->setParent(*gBg);
    }
    {
      OFX::GroupParamDescriptor *gOv = desc.defineGroupParam("Group_OverlayColor");
      gOv->setLabel("Guide Lines");
      gOv->setOpen(true);
      gOv->setParent(*gAdv);

      OFX::RGBParamDescriptor *c = desc.defineRGBParam("overlay_color");
      c->setLabel("Guide Line Color");
      c->setDefault(0.8, 0.8, 0.8);
      c->setAnimates(true);
      c->setParent(*gOv);
    }
  }

  OFX::ImageEffect *createInstance(OfxImageEffectHandle handle,
                                   OFX::ContextEnum /*context*/) override {
    return new ReframePlugin(handle);
  }
};

namespace OFX {
namespace Plugin {
void getPluginIDs(OFX::PluginFactoryArray &ids) {
  static ReframeFactory factory;
  ids.push_back(&factory);
}
}  // namespace Plugin
}  // namespace OFX
