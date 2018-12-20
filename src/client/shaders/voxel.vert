#version 450

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout(location = 0) in vec4 position;
layout(location = 1) in vec4 color;

layout(location = 0) out vec4 v_color;

void main() {
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(position.xyz, 1.0);
    v_color = color;
}
