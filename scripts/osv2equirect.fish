#!/usr/bin/env fish
# Osmo 360 .OSV (dual fisheye, two HEVC streams) -> equirectangular intermediate
# for DaVinci Resolve.
#
# No lens calibration is applied, so the two stitch seams stay visible. DJI
# Studio is the only thing that can remove them properly.

argparse 'q/quality=' 'f/fov=' 'c/codec=' 's/start=' 'd/duration=' 'y/yaw=' 't/tilt=' 'r/roll=' 'p/preview' 'g/grid' 'n/dry-run' -- $argv; or exit 1

# v360 only accepts -180..180, so angles wrap around instead of clamping.
function _wrap -a deg
    math "(($deg + 180) % 360 + 360) % 360 - 180"
end

if test (count $argv) -lt 1
    echo "usage: osv2equirect.fish [options] <file.OSV>..."
    echo
    echo "  -q full|high|edit   output size (default high)"
    echo "       full  7680x3840  native angular resolution, ~22 GB/min"
    echo "       high  5760x2880  82% of native,             ~13 GB/min"
    echo "       edit  3840x1920  54% of native,             ~5.5 GB/min"
    echo "  -c dnxhr|prores     intermediate codec (default dnxhr)"
    echo "  -f FOV              fisheye field of view, default 196"
    echo "  -y/-t/-r DEGREES    yaw/tilt/roll baked into the stitch so the clip"
    echo "                      lands level (default yaw 180). Any value works,"
    echo "                      270 and -90 mean the same thing."
    echo "  -g                  write a labelled grid of orientations to pick from"
    echo "  -s SECONDS          start offset"
    echo "  -d SECONDS          duration"
    echo "  -p                  write a single preview PNG instead"
    echo "  -n                  print the ffmpeg command without running it"
    exit 1
end

set -l ih_fov (set -q _flag_fov; and echo $_flag_fov; or echo 196)
# Baked in so the Source File controls in the OFX plugin can stay at zero.
set -l yaw (_wrap (set -q _flag_yaw; and echo $_flag_yaw; or echo 180))
set -l tilt (_wrap (set -q _flag_tilt; and echo $_flag_tilt; or echo 0))
set -l roll (_wrap (set -q _flag_roll; and echo $_flag_roll; or echo 0))
set -l quality (set -q _flag_quality; and echo $_flag_quality; or echo high)
set -l codec (set -q _flag_codec; and echo $_flag_codec; or echo dnxhr)
set -l seek (set -q _flag_start; and echo $_flag_start; or echo 5)

switch $quality
    case full
        set -g W 7680; set -g H 3840
    case high
        set -g W 5760; set -g H 2880
    case edit
        set -g W 3840; set -g H 1920
    case '*'
        echo "unknown quality '$quality' (use full, high or edit)"; exit 1
end

# 10-bit 4:2:2 either way, which matches the 10-bit HEVC source.
switch $codec
    case dnxhr
        set -g VOPTS -c:v dnxhd -profile:v dnxhr_hqx -pix_fmt yuv422p10le
    case prores
        set -g VOPTS -c:v prores_ks -profile:v 3 -pix_fmt yuv422p10le
    case '*'
        echo "unknown codec '$codec' (use dnxhr or prores)"; exit 1
end

set -l trim
set -q _flag_start; and set trim $trim -ss $_flag_start
set -q _flag_duration; and set trim $trim -t $_flag_duration

set -l lbl "[o]"

for src in $argv
    if not test -f "$src"
        echo "missing: $src"; continue
    end

    set -l base (string replace -ir '\.OSV$' '' -- $src)
    set -l stitch "[0:v:0][0:v:1]hstack=inputs=2[d];[d]v360=dfisheye:e:ih_fov=$ih_fov:iv_fov=$ih_fov"

    if set -q _flag_grid
        # Render every yaw/roll combination once so the right one can be read off.
        set -l out "$base"_orientations.png
        set -l parts
        set -l ins
        set -l i 0
        for gy in 0 180
            for gr in 0 90 180 -90
                set -l t "yaw $gy  roll $gr"
                set parts $parts "$stitch:yaw=$gy:pitch=0:roll=$gr:w=640:h=320,drawtext=text='$t':x=8:y=8:fontsize=28:fontcolor=yellow:box=1:boxcolor=black@0.6[v$i]"
                set ins $ins "[v$i]"
                set i (math $i + 1)
            end
        end
        set -l layout "0_0|w0_0|w0+w0_0|w0+w0+w0_0|0_h0|w0_h0|w0+w0_h0|w0+w0+w0_h0"
        set -l graph (string join ';' $parts)";"(string join '' $ins)"xstack=inputs=8:layout=$layout$lbl"
        ffmpeg -hide_banner -loglevel error -ss $seek -i $src -filter_complex $graph -map "[o]" -frames:v 1 -y $out
        or begin; echo "grid failed for $src"; continue; end
        echo $out
        continue
    end

    set -l rot "yaw=$yaw:pitch=$tilt:roll=$roll"
    set -l graph "$stitch:$rot:w=$W:h=$H$lbl"

    if set -q _flag_preview
        set -l out "$base"_equirect.png
        ffmpeg -hide_banner -loglevel error -ss $seek -i $src \
            -filter_complex $graph -map "[o]" -frames:v 1 -y $out
        echo $out
        continue
    end

    set -l out "$base"_equirect.mov
    set -l cmd ffmpeg -hide_banner $trim -i $src -filter_complex $graph \
        -map "[o]" -map 0:a:0 $VOPTS -c:a pcm_s16le -y $out

    if set -q _flag_dry_run
        echo $cmd
        continue
    end

    echo "-> $out  ($W"x"$H, $quality, $codec, yaw $yaw tilt $tilt roll $roll)"
    $cmd; or begin
        echo "failed: $src"
        continue
    end
    echo "   "(du -h $out | cut -f1)
end
