{
  nixConfig = {
    extra-substituters = [
      "https://nix-cache.fossi-foundation.org"
    ];
    extra-trusted-public-keys = [
      "nix-cache.fossi-foundation.org:3+K59iFwXqKsL7BNu6Guy0v+uTlwsxYQxjspXzqLYQs="
    ];
  };

  inputs = {
    librelane.url = "github:librelane/librelane/dev";
  };

  outputs =
    {
      self,
      librelane,
      ...
    }:
    let
      nix-eda = librelane.inputs.nix-eda;
      devshell = librelane.inputs.devshell;
      nixpkgs = nix-eda.inputs.nixpkgs;
      lib = nixpkgs.lib;
    in
    {
      # Outputs
      legacyPackages = nix-eda.forAllSystems (
        system:
        import nixpkgs {
          inherit system;
          overlays = [
            nix-eda.overlays.default
            devshell.overlays.default
            librelane.overlays.default
          ];
        }
      );

      packages = nix-eda.forAllSystems (system: {
        inherit (self.legacyPackages.${system}.python3.pkgs) ;
      });

      devShells = nix-eda.forAllSystems (
        system:
        let
          pkgs = (self.legacyPackages.${system});
          callPackage = lib.callPackageWith pkgs;
        in
        {
          default = pkgs.librelane-shell.override ({
            extra-packages = with pkgs; [
              # Utilities
              gnumake
              gnugrep
              gawk

              # Simulation
              iverilog
              verilator

              # Waveform viewing
              gtkwave
              surfer
            ];

            extra-python-packages =
              ps:
              let
                cocotbext-spi = ps.buildPythonPackage {
                  pname = "cocotbext-spi";
                  version = "0-unstable-20251215";
                  src = pkgs.fetchFromGitHub {
                    owner = "schang412";
                    repo = "cocotbext-spi";
                    rev = "be5761be54796a64607c4b115e2e7cfe8ac6bc83";
                    hash = "sha256-IOmeta+Gc4t+KV237xlTEV+f0z364mq8k3Z3ViiLeJM=";
                  };
                  pyproject = true;
                  build-system = with ps; [ setuptools setuptools-scm wheel ];
                  dependencies = with ps; [ cocotb ];
                  env.SETUPTOOLS_SCM_PRETEND_VERSION = "0.1";
                };
              in
              with ps; [
                # Verification
                cocotb
                pytest
                cocotbext-spi

                # For KLayout Python DRC runner
                docopt

                # For logo generation
                pillow
              ];
          });
        }
      );
    };
}
