const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const maybe_override_registry = b.option([]const u8, "override-registry", "Override the path to the Vulkan registry used for the examples");

    const registry = b.dependency("vulkan_headers", .{}).path("registry/vk.xml");

    const exe_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .link_libc = true,
        .optimize = optimize,
    });

    const registry_path: std.Build.LazyPath = if (maybe_override_registry) |override_registry|
        .{ .cwd_relative = override_registry }
    else
        registry;

    const vulkan = b.dependency("vulkan_zig", .{
        .registry = registry_path,
    }).module("vulkan-zig");

    exe_mod.addImport("vulkan", vulkan);

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
