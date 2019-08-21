#version 450

layout(set = 1, binding = 0) uniform sampler2DArray voxel_tarray;

layout(location = 0) in vec4 v_position;
layout(location = 1) in vec4 v_color;
layout(location = 2) in vec3 v_texcoord;

layout(location = 0) out vec4 f_color;

void main() {
    f_color = v_color * texture(voxel_tarray, v_texcoord);
}
