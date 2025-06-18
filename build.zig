const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const maybe_vulkan_sdk = std.process.getEnvVarOwned(b.allocator, "VULKAN_SDK") catch null;
    defer if (maybe_vulkan_sdk) |vulkan_sdk| b.allocator.free(vulkan_sdk);

    const default_registry = "/usr/share/vulkan/registry/vk.xml";

    const exe_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .link_libc = true,
        .optimize = optimize,
    });

    const registry_path: std.Build.LazyPath = .{
        .cwd_relative = if (maybe_vulkan_sdk) |vulkan_sdk| b.pathJoin(&.{
            vulkan_sdk,
            "share",
            "vulkan",
            "registry",
            "vk.xml",
        }) else default_registry,
    };

    const vulkan = b.dependency("vulkan_zig", .{
        .registry = registry_path,
    }).module("vulkan-zig");

    exe_mod.addImport("vulkan", vulkan);

    const zalgebra = b.dependency("zalgebra", .{}).module("zalgebra");

    exe_mod.addImport("zalgebra", zalgebra);

    const cgltf = b.dependency("cgltf", .{});

    exe_mod.addIncludePath(cgltf.path("."));

    exe_mod.addCSourceFile(.{
        .file = cgltf.path("cgltf.h"),
        .language = .c,
        .flags = &.{"-DCGLTF_IMPLEMENTATION"},
    });

    const vert_cmd = b.addSystemCommand(&.{
        "glslc",
        "--target-env=vulkan1.4",
        "-o",
    });
    const vert_spv = vert_cmd.addOutputFileArg("vert.spv");
    vert_cmd.addFileArg(b.path("shaders/triangle.vert"));
    exe_mod.addAnonymousImport("vertex_shader", .{
        .root_source_file = vert_spv,
    });

    const frag_cmd = b.addSystemCommand(&.{
        "glslc",
        "--target-env=vulkan1.4",
        "-o",
    });
    const frag_spv = frag_cmd.addOutputFileArg("frag.spv");
    frag_cmd.addFileArg(b.path("shaders/triangle.frag"));
    exe_mod.addAnonymousImport("fragment_shader", .{
        .root_source_file = frag_spv,
    });

    const exe = b.addExecutable(.{
        .name = "triangle",
        .root_module = exe_mod,
    });
    b.installArtifact(exe);
    exe.linkSystemLibrary("glfw");

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);
}
