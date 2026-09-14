#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "ofxsImageEffect.h"
#include "ofxsMultiThread.h"
#include "ofxsProcessing.h"

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

// Source is bottom-up (OFX) while the reference kernels assume a top-down
// equirect texture, so v is flipped at sample time rather than in the maths.
void sampleBilinear(const float *src, int srcW, int srcH, int srcRowFloats,
                    float u, float v, float *out) {
  u = std::clamp(u, 0.0f, 1.0f);
  v = std::clamp(v, 0.0f, 1.0f);
  const float fx = u * static_cast<float>(srcW - 1);
  const float fy = (1.0f - v) * static_cast<float>(srcH - 1);
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
    v = 1.0f - theta / kPi;
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

    projection_ = fetchChoiceParam("projection");
    pan_ = fetchDoubleParam("pan");
    tilt_ = fetchDoubleParam("tilt");
    roll_ = fetchDoubleParam("roll");
    fov_ = fetchDoubleParam("fov");
    distortion_ = fetchDoubleParam("distortion");
    srcYaw_ = fetchDoubleParam("srcYaw");
    srcPitch_ = fetchDoubleParam("srcPitch");
    srcRoll_ = fetchDoubleParam("srcRoll");
    borderMode_ = fetchChoiceParam("borderMode");
    borderColor_ = fetchRGBParam("borderColor");
    reset_ = fetchPushButtonParam("reset");
  }

  void render(const OFX::RenderArguments &args) override {
    std::unique_ptr<OFX::Image> dst(dstClip_->fetchImage(args.time));
    std::unique_ptr<OFX::Image> src(srcClip_->fetchImage(args.time));
    if (!dst || !src) return;

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
    if (name != "reset" || args.reason != OFX::eChangeUserEdit) return;
    pan_->setValue(0.0);
    tilt_->setValue(0.0);
    roll_->setValue(0.0);
    fov_->setValue(90.0);
    distortion_->setValue(0.0);
  }

 private:
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
  OFX::PushButtonParam *reset_ = nullptr;
};

class ReframeFactory : public OFX::PluginFactoryHelper<ReframeFactory> {
 public:
  ReframeFactory()
      : OFX::PluginFactoryHelper<ReframeFactory>(kPluginIdentifier,
                                                 kPluginVersionMajor,
                                                 kPluginVersionMinor) {}

  void describe(OFX::ImageEffectDescriptor &desc) override {
    desc.setLabels(kPluginName, kPluginName, kPluginName);
    desc.setPluginGrouping(kPluginGrouping);
    desc.setPluginDescription(
        "Reframe equirectangular 360 footage to a flat view. "
        "Expects an equirectangular (2:1) source.");
    desc.addSupportedContext(OFX::eContextFilter);
    desc.addSupportedContext(OFX::eContextGeneral);
    desc.addSupportedBitDepth(OFX::eBitDepthFloat);
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

    {
      OFX::ChoiceParamDescriptor *p = desc.defineChoiceParam("projection");
      p->setLabel("Projection");
      p->appendOption("Wide");
      p->appendOption("Crystal Ball");
      p->appendOption("Tiny Planet");
      p->appendOption("Ultra Wide");
      p->appendOption("Dewarp");
      p->setDefault(kProjWide);
      p->setAnimates(true);
      page->addChild(*p);
    }

    auto angle = [&](const char *name, const char *label, double def, double lo,
                     double hi) {
      OFX::DoubleParamDescriptor *p = desc.defineDoubleParam(name);
      p->setLabel(label);
      p->setDoubleType(OFX::eDoubleTypeAngle);
      p->setDefault(def);
      p->setRange(lo, hi);
      p->setDisplayRange(lo, hi);
      p->setAnimates(true);
      page->addChild(*p);
    };

    angle("pan", "Pan", 0.0, -3600.0, 3600.0);
    angle("tilt", "Tilt", 0.0, -180.0, 180.0);
    angle("roll", "Roll", 0.0, -180.0, 180.0);

    {
      OFX::DoubleParamDescriptor *p = desc.defineDoubleParam("fov");
      p->setLabel("FOV");
      p->setDefault(90.0);
      p->setRange(5.0, 170.0);
      p->setDisplayRange(5.0, 170.0);
      p->setAnimates(true);
      page->addChild(*p);
    }
    {
      OFX::DoubleParamDescriptor *p = desc.defineDoubleParam("distortion");
      p->setLabel("Distortion");
      p->setDefault(0.0);
      p->setRange(-1.0, 1.0);
      p->setDisplayRange(-1.0, 1.0);
      p->setAnimates(true);
      page->addChild(*p);
    }

    {
      OFX::GroupParamDescriptor *g = desc.defineGroupParam("sourceOrientation");
      g->setLabel("Source Orientation");
      g->setOpen(false);
      page->addChild(*g);

      auto srcAngle = [&](const char *name, const char *label) {
        OFX::DoubleParamDescriptor *p = desc.defineDoubleParam(name);
        p->setLabel(label);
        p->setDoubleType(OFX::eDoubleTypeAngle);
        p->setDefault(0.0);
        p->setRange(-180.0, 180.0);
        p->setDisplayRange(-180.0, 180.0);
        p->setAnimates(true);
        p->setParent(*g);
      };
      srcAngle("srcYaw", "Source Yaw");
      srcAngle("srcPitch", "Source Pitch");
      srcAngle("srcRoll", "Source Roll");
    }

    {
      OFX::ChoiceParamDescriptor *p = desc.defineChoiceParam("borderMode");
      p->setLabel("Border");
      p->appendOption("Transparent");
      p->appendOption("Color");
      p->appendOption("Mirror / Extend");
      p->setDefault(kBorderTransparent);
      p->setAnimates(false);
      page->addChild(*p);
    }
    {
      OFX::RGBParamDescriptor *p = desc.defineRGBParam("borderColor");
      p->setLabel("Border Color");
      p->setDefault(0.0, 0.0, 0.0);
      p->setAnimates(true);
      page->addChild(*p);
    }
    {
      OFX::PushButtonParamDescriptor *p = desc.definePushButtonParam("reset");
      p->setLabel("Reset View");
      page->addChild(*p);
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
