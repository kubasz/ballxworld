use crate::registry::VoxelRegistry;
use crate::TextureMapping;

pub fn register_standard_blocks(vxreg: &mut VoxelRegistry, texmapper: &dyn Fn(&str) -> u32) {
    vxreg
        .build_definition()
        .name("core:grass")
        .texture_names(
            texmapper,
            TextureMapping::TiledTSB {
                top: "grass_top",
                side: "dirt_grass",
                bottom: "dirt",
            },
        )
        .has_physical_properties()
        .finish()
        .unwrap();
    vxreg
        .build_definition()
        .name("core:snow_grass")
        .texture_names(
            texmapper,
            TextureMapping::TiledTSB {
                top: "snow",
                side: "dirt_snow",
                bottom: "dirt",
            },
        )
        .has_physical_properties()
        .finish()
        .unwrap();
    vxreg
        .build_definition()
        .name("core:dirt")
        .texture_names(texmapper, TextureMapping::TiledSingle("dirt"))
        .has_physical_properties()
        .finish()
        .unwrap();
    vxreg
        .build_definition()
        .name("core:stone")
        .texture_names(texmapper, TextureMapping::TiledSingle("stone"))
        .has_physical_properties()
        .finish()
        .unwrap();
    vxreg
        .build_definition()
        .name("core:diamond_ore")
        .texture_names(texmapper, TextureMapping::TiledSingle("stone_diamond"))
        .has_physical_properties()
        .finish()
        .unwrap();
    vxreg
        .build_definition()
        .name("core:water")
        .texture_names(texmapper, TextureMapping::TiledSingle("water"))
        .has_physical_properties()
        .finish()
        .unwrap();
}
