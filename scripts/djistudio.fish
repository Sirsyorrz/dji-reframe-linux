#!/usr/bin/env fish
# Launch DJI Studio under Wine. Prefix is set up with DXVK for D3D11.

set -x WINEPREFIX /home/sirs/.local/share/wineprefixes/djistudio
set -x WINEDEBUG -all
set -x DXVK_LOG_PATH /tmp

if test "$argv[1]" = "-k"
    wineserver -k
    exit 0
end

cd "$WINEPREFIX/drive_c/Program Files/DJI Studio"; or exit 1
exec wine "DJI Studio.exe" $argv
