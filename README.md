# dji-reframe-linux

A native Linux OFX plugin that reframes equirectangular 360 footage in DaVinci Resolve,
reimplementing DJI's `DJI Reframe for DaVinci Resolve` plugin, which ships Windows and
macOS binaries only.

The official installer contains a single `Win64/DJIReframe.ofx` PE DLL. An OFX bundle can
hold a `Linux-x86-64/` payload, DJI just doesn't build one, and a PE plugin cannot be
loaded by Resolve's native Linux binary. This is a clean reimplementation of the same
projection maths, not a binary wrapper.

Plugin identifier and version match the original (`com.pano360.reframe` 1.0), so projects
moved between platforms keep their node links.

## Build

```
git clone https://github.com/AcademySoftwareFoundation/openfx.git vendor/openfx
make -j$(nproc)
sudo make install     # -> /usr/OFX/Plugins/DJIReframe.ofx.bundle
```

Alternatively point `OFX_PLUGIN_PATH` at `build/`.

## Parameters

| Param | Notes |
| --- | --- |
| Projection | Wide, Crystal Ball, Tiny Planet, Ultra Wide, Dewarp |
| Pan / Tilt / Roll | View orientation, keyframable |
| FOV | 5-170 degrees |
| Distortion | Eye distance from the sphere centre, drives the ball/planet look |
| Source Yaw / Pitch / Roll | Horizon correction for the source pano |
| Border | Transparent, Color, Mirror |

## Input format

The source clip must be **equirectangular** (2:1). Raw Osmo 360 `.OSV` files are two
3000x3000 fisheye HEVC streams in one MP4 and must be stitched by DJI Studio first, on
Windows or macOS, or by the Mimo app. That step is unrelated to this plugin and is
required on Windows too.

## Rendering

CPU only, multithreaded via the OFX host. The original also carries CUDA and OpenCL
paths; the OpenCL kernels are embedded as plain text in the shipped DLL and match the
maths here one for one.
