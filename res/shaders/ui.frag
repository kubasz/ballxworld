#version 450

layout(set = 0, binding = 0) uniform sampler2D texatlas;

layout(location = 0) in vec2 v_position;
layout(location = 1) in vec2 v_texcoord;
layout(location = 2) in vec4 v_color;

layout(location = 0) out vec4 f_color;

void main() {
    f_color = v_color * texture(texatlas, v_texcoord);
}
