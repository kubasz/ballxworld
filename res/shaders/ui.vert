#version 450

layout(location = 0) in vec2 position;
layout(location = 1) in vec3 texcoord;
layout(location = 2) in vec4 color;
layout(location = 3) in int texselect;

layout(location = 0) out vec2 v_position;
layout(location = 1) out vec3 v_texcoord;
layout(location = 2) out vec4 v_color;
layout(location = 3) flat out int v_texselect;

void main() {
    gl_Position = vec4(position, 1.0, 1.0);
    v_position = position;
    v_color = color;
    v_texcoord = texcoord;
    v_texselect = texselect;
}
