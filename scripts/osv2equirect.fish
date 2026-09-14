#!/usr/bin/env fish
# Rough dual-fisheye -> equirectangular stitch for Osmo 360 .OSV files.
# No lens calibration is applied, so seams stay visible. Use DJI Studio for finals.

set -l ih_fov 196
argparse 'f/fov=' 'w/width=' 'p/preview' -- $argv; or exit 1
set -q _flag_fov; and set ih_fov $_flag_fov
set -l width (set -q _flag_width; and echo $_flag_width; or echo 3840)
set -l height (math "$width / 2")

if test (count $argv) -lt 1
    echo "usage: osv2equirect.fish [-f FOV] [-w WIDTH] [-p] <file.OSV>... "
    exit 1
end

for src in $argv
    set -l out (string replace -r '\.OSV$' '' -- $src)_equirect

    if set -q _flag_preview
        ffmpeg -hide_banner -loglevel error -y \
            -filter_complex "[0:v:0][0:v:1]hstack=inputs=2[d];[d]v360=dfisheye:e:ih_fov=$ih_fov:iv_fov=$ih_fov:w=1600:h=800[o]" \
            -ss 5 -i $src -map "[o]" -frames:v 1 $out.png
        echo "$out.png"
        continue
    end

    ffmpeg -hide_banner -y -i $src \
        -filter_complex "[0:v:0][0:v:1]hstack=inputs=2[d];[d]v360=dfisheye:e:ih_fov=$ih_fov:iv_fov=$ih_fov:w=$width:h=$height[o]" \
        -map "[o]" -map 0:a:0 \
        -c:v dnxhd -profile:v dnxhr_hq -pix_fmt yuv422p -c:a pcm_s16le \
        $out.mov
    echo "$out.mov"
end
