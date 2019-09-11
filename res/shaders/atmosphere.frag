#version 450

layout(location = 0) in vec3 v_position;

layout(location = 0) out vec4 f_color;

void main() {
    vec3 col = (v_position + vec3(1.0)) / 2.2;
    f_color = vec4(col, 1.0);
}
