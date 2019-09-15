#version 450

layout(push_constant) uniform PushConstants {
    vec3 chunk_offset;
    int highlight_index;
} push_constants;

layout(set = 1, binding = 0) uniform sampler2DArray voxel_tarray;

layout(location = 0) in vec4 v_position;
layout(location = 1) in vec4 v_color;
layout(location = 2) in vec3 v_texcoord;
layout(location = 3) in flat int v_index;

layout(location = 0) out vec4 f_color;

void main() {
    vec4 base_color = v_color * texture(voxel_tarray, v_texcoord);
    if (push_constants.highlight_index == v_index) {
        f_color = clamp(base_color*1.5, 0.0, 1.0);
    } else {
        f_color = base_color;
    }
}