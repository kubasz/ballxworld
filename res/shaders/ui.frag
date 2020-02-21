#version 450

layout(set = 0, binding = 0) uniform sampler2D guiatlas;
layout(set = 0, binding = 1) uniform sampler2D fontatlas;
layout(set = 0, binding = 2) uniform sampler2DArray voxelatlas;

layout(location = 0) in vec2 v_position;
layout(location = 1) in vec3 v_texcoord;
layout(location = 2) in vec4 v_color;
layout(location = 3) flat in int v_texselect;

layout(location = 0) out vec4 f_color;

void main() {
    vec4 tcolor = vec4(1.0, 1.0, 1.0, 1.0);
    if(v_texselect == 0) {
        tcolor = texture(guiatlas, v_texcoord.xy);
    } else if (v_texselect == 1) {
        tcolor = texture(fontatlas, v_texcoord.xy);
    } else if (v_texselect == 2) {
        tcolor = texture(voxelatlas, v_texcoord);
    }
    f_color = v_color * tcolor;
}
