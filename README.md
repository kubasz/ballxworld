# BallX World

A voxel-based sandbox game using Vulkan, SDL2 and Rust.

## Building on ArchLinux

#### Dependencies
Make sure to have the latest Rust (stable) and graphics drivers
with Vulkan support installed and working first.

Other dependencies:
```shell
sudo pacman -Syu --needed vulkan-devel shaderc sdl2
```

#### Downloading & building
```shell
git clone https://github.com/kubasz/ballxworld.git ballxworld
cd ballxworld
cargo update
cargo build [--release] # Add release if you want a fully optimized build
```

#### Running

Running debug build: `cargo run`

Running debug build directly: `./target/debug/ballxworld`

For RenderDoc, set the executable to `PROJECT/target/debug/ballxworld`
working directory to `PROJECT`, and arguments to `-renderdoc`

You can adjust rendering settings in the `settings.toml` file, which should be auto-generated after the first start of the game.

#### Development

To perform lint checks on the code: `cargo clippy [--package bxw_world]`

To reformat code before a commit: `cargo fmt [--package bxw_world]`

To recompile modified GLSL shaders into SPIR-V: `make`
