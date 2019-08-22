use crate::client::render::VoxelRenderer;
use crate::world::registry::{VoxelDefinitionBuilder, VoxelRegistry};
use crate::world::TextureMapping;

pub fn register_standard_blocks(vxreg: &mut VoxelRegistry, vctx: Option<&VoxelRenderer>) {
    vxreg
        .build_definition()
        .name("core:grass")
        .opt_texture_names(
            vctx,
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
        .name("core:dirt")
        .opt_texture_names(vctx, TextureMapping::TiledSingle("dirt"))
        .has_physical_properties()
        .finish()
        .unwrap();
    vxreg
        .build_definition()
        .name("core:stone")
        .opt_texture_names(vctx, TextureMapping::TiledSingle("stone"))
        .has_physical_properties()
        .finish()
        .unwrap();
    vxreg
        .build_definition()
        .name("core:diamond_ore")
        .opt_texture_names(vctx, TextureMapping::TiledSingle("stone_diamond"))
        .has_physical_properties()
        .finish()
        .unwrap();
    vxreg
        .build_definition()
        .name("core:border")
        .opt_texture_names(vctx, TextureMapping::TiledSingle("table"))
        .has_physical_properties()
        .finish()
        .unwrap();
}

trait OptionalTextureNames {
    fn opt_texture_names(self, vctx: Option<&VoxelRenderer>, t: TextureMapping<&str>) -> Self;
}

impl OptionalTextureNames for VoxelDefinitionBuilder<'_> {
    fn opt_texture_names(self, vctx: Option<&VoxelRenderer>, t: TextureMapping<&str>) -> Self {
        if let Some(vctx) = vctx {
            self.texture_names(vctx, t)
        } else {
            self
        }
    }
}
