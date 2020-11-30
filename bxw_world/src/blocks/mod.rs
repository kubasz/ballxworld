pub mod stdshapes;

use crate::voxregistry::VoxelRegistry;
use crate::TextureMapping;

pub fn register_standard_blocks(vxreg: &mut VoxelRegistry, texmapper: &dyn Fn(&str) -> u32) {
    vxreg
        .build_definition()
        .name("core:grass")
        .texture_names(
            texmapper,
            TextureMapping::new_tsb("grass_top", "dirt_grass", "dirt"),
        )
        .finish()
        .unwrap();
    vxreg
        .build_definition()
        .name("core:snow_grass")
        .texture_names(
            texmapper,
            TextureMapping::new_tsb("snow", "dirt_snow", "dirt"),
        )
        .finish()
        .unwrap();
    vxreg
        .build_definition()
        .name("core:dirt")
        .texture_names(texmapper, TextureMapping::new_single("dirt"))
        .finish()
        .unwrap();
    vxreg
        .build_definition()
        .name("core:stone")
        .texture_names(texmapper, TextureMapping::new_single("stone"))
        .finish()
        .unwrap();
    vxreg
        .build_definition()
        .name("core:diamond_ore")
        .texture_names(texmapper, TextureMapping::new_single("stone_diamond"))
        .finish()
        .unwrap();
    vxreg
        .build_definition()
        .name("core:debug")
        .texture_names(
            texmapper,
            TextureMapping::new([
                "dbg_left",
                "dbg_right",
                "dbg_down",
                "dbg_up",
                "dbg_front",
                "dbg_back",
            ]),
        )
        .finish()
        .unwrap();
    vxreg
        .build_definition()
        .name("core:table")
        .texture_names(texmapper, TextureMapping::new_tsb("table", "wood", "table"))
        .finish()
        .unwrap();
}
