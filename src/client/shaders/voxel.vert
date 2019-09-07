#version 450

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout(push_constant) uniform PushConstants {
    vec3 chunk_offset;
    int highlight_index;
} push_constants;

layout(location = 0) in vec4 position;
layout(location = 1) in vec4 color;
layout(location = 2) in vec3 texcoord;
layout(location = 3) in int index;

layout(location = 0) out vec4 v_position;
layout(location = 1) out vec4 v_color;
layout(location = 2) out vec3 v_texcoord;
layout(location = 3) out int v_index;

void main() {
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(position.xyz + push_constants.chunk_offset, 1.0);
    v_position = position;
    v_color = color;
    v_texcoord = texcoord;
    v_index = index;
}
