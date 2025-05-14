{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    zig-overlay = {
      url = "github:mitchellh/zig-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    shaderc-overlay = {
      url = "github:DontEatOreo/nixpkgs/update-koboldcpp";
      flake = false;
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      zig-overlay,
      shaderc-overlay,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        overlays = [
          zig-overlay.overlays.default
	  (final: prev: {
	    shaderc = (import shaderc-overlay { inherit system; }).shaderc;
	  })
        ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };
      in
      with pkgs;
      {
        devShells.default = mkShell rec {
          buildInputs = [
	    zigpkgs.master
	    zls
	    shaderc #glslc
	    glfw
	    vulkan-tools
	    vulkan-headers #vk.xml
	    vulkan-validation-layers
          ];
	  VULKAN_SDK="${vulkan-headers}";
      	  VK_LAYER_PATH="${vulkan-validation-layers}/share/vulkan/explicit_layer.d";
        };

        formatter = nixfmt-rfc-style;
      }
    );
}
