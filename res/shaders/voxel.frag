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
layout(location = 4) in vec4 v_barycentric_color_offset;
layout(location = 5) in vec2 v_barycentric;

layout(location = 0) out vec4 f_color;

const float sel_border = 0.02;

void main() {
    vec4 corrected_v_color = v_color + v_barycentric.x * v_barycentric.y * v_barycentric_color_offset;
    vec4 base_color = corrected_v_color * texture(voxel_tarray, v_texcoord);
    if (push_constants.highlight_index == v_index) {
        //f_color = clamp(base_color*1.1, 0.0, 1.0);
        if (v_texcoord.x < sel_border || v_texcoord.x > 1.0-sel_border || v_texcoord.y < sel_border || v_texcoord.y > 1.0-sel_border) {
            f_color = vec4(0, 0, 0, 1);
        } else {
            f_color = base_color;
        }
    } else {
        f_color = base_color;
    }
}
