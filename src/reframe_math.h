#pragma once

#include <cmath>

namespace reframe {

constexpr float kPi = 3.14159265358979323846f;

struct Vec3 {
  float x, y, z;
};

struct Mat4 {
  float m[16];  // column-major, m[col*4 + row]
};

inline Mat4 identity() {
  Mat4 r{};
  r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f;
  return r;
}

inline Mat4 multiply(const Mat4 &a, const Mat4 &b) {
  Mat4 r{};
  for (int c = 0; c < 4; ++c)
    for (int row = 0; row < 4; ++row) {
      float s = 0.0f;
      for (int k = 0; k < 4; ++k) s += a.m[k * 4 + row] * b.m[c * 4 + k];
      r.m[c * 4 + row] = s;
    }
  return r;
}

inline Mat4 perspective(float fovYRad, float aspect, float zn, float zf) {
  Mat4 r{};
  const float f = 1.0f / std::tan(fovYRad * 0.5f);
  r.m[0] = f / aspect;
  r.m[5] = f;
  r.m[10] = (zf + zn) / (zn - zf);
  r.m[11] = -1.0f;
  r.m[14] = (2.0f * zf * zn) / (zn - zf);
  return r;
}

inline Mat4 rotationX(float a) {
  Mat4 r = identity();
  const float c = std::cos(a), s = std::sin(a);
  r.m[5] = c;
  r.m[6] = s;
  r.m[9] = -s;
  r.m[10] = c;
  return r;
}

inline Mat4 rotationY(float a) {
  Mat4 r = identity();
  const float c = std::cos(a), s = std::sin(a);
  r.m[0] = c;
  r.m[2] = -s;
  r.m[8] = s;
  r.m[10] = c;
  return r;
}

inline Mat4 rotationZ(float a) {
  Mat4 r = identity();
  const float c = std::cos(a), s = std::sin(a);
  r.m[0] = c;
  r.m[1] = s;
  r.m[4] = -s;
  r.m[5] = c;
  return r;
}

inline Mat4 translation(float x, float y, float z) {
  Mat4 r = identity();
  r.m[12] = x;
  r.m[13] = y;
  r.m[14] = z;
  return r;
}

inline bool invert(const Mat4 &src, Mat4 &out) {
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

inline void dirToEquirectUV(const Vec3 &d, float &u, float &v) {
  const float theta = std::acos(d.y < -1.0f ? -1.0f : (d.y > 1.0f ? 1.0f : d.y));
  float phi = std::atan2(d.x, d.z);
  if (phi < 0.0f) phi += 2.0f * kPi;
  u = phi / (2.0f * kPi);
  v = 1.0f - theta / kPi;
}

// Unproject an NDC point, intersect the unit sphere, return the equirect UV.
inline bool rayToEquirectUV(const Mat4 &invMVP, float ndcX, float ndcY, float &u,
                            float &v) {
  const float *M = invMVP.m;
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

  dirToEquirectUV({nearPt.x + t * dir.x, nearPt.y + t * dir.y, nearPt.z + t * dir.z},
                  u, v);
  return true;
}

}  // namespace reframe
