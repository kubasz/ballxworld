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

## Implementation notes

### Coordinate system

The game uses a left-handed coordinate system, where x+ points to the right, y+ up and z+ into the monitor (towards the back direction).

Voxels with integer coordinates (x,y,z) have an extent of (x-0.5,...) to (x+0.5,...).
Chunks are 32x32x32, with the first one having the extent of voxels (0,0,0) to (31,31,31) inclusive.

Calculations between block integer positions, floating point positions and chunk positions are provided in the bxw_world module.

### Block orientations

Orientations are internally specified by a pair of vectors `right` and `up`, with `forward` defined as `f = -r x u`
using the standard right-handed vector cross product. Various constructors from other forms are provided,
some use extra redundant data to verify the input and return an Option<> rather than an orientation to allow for verification.

The default orientation (id = 0) is right=x+ and up=y+ (front/forward=z-).

Slopes have their triangle sides facing left and right, with the slope being between top and front.

Corner slopes are meant to connect with other slopes with their left and back faces. Their top, front and right faces
are the "filler" for a neat finish.

## License

The source code of the project is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0-standalone.html).

The assets in the `res/` folder are licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0) unless specified otherwise in a file located in the same folder as the asset.

