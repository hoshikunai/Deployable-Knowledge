{
  pkgs ? import <nixpkgs> { },
}:
let
  runtimeLibs =
    with pkgs;
    (lib.makeLibraryPath [
      stdenv.cc.cc.lib
      glib
      zlib
      libGL
      libxcb
    ]);
in
pkgs.mkShell {
  packages = with pkgs; [
    python313
    python313Packages.black
    python313Packages.pylint
    python313Packages.pyinstaller
    nodejs
    electron_40-bin
    gnumake
    sqlite
  ];

  shellHook = ''
    export LD_LIBRARY_PATH="${runtimeLibs}:''${LD_LIBRARY_PATH:-}"
    for electron_dist in "${pkgs.electron_40-bin}/libexec/electron" "${pkgs.electron_40-bin}/share/electron" "${pkgs.electron_40-bin}/bin"; do
      if [ -x "$electron_dist/electron" ]; then
        export ELECTRON_OVERRIDE_DIST_PATH="$electron_dist"
        break
      fi
    done
  '';
}
