#version 450

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
} ubo;

const float vtable[8*3] = float[](
1.f, 1.f, 1.f,
-1.f, 1.f, 1.f,
-1.f, 1.f, -1.f,
1.f, 1.f, -1.f,
1.f, -1.f, 1.f,
-1.f, -1.f, 1.f,
-1.f, -1.f, -1.f,
1.f, -1.f, -1.f
);

const int itable[12*3] = int[](
0, 1, 3,
3, 1, 2,
2, 6, 7,
7, 3, 2,
7, 6, 5,
5, 4, 7,
5, 1, 4,
4, 1, 0,
4, 3, 7,
3, 4, 0,
5, 6, 2,
5, 2, 1
);

layout(location = 0) out vec3 v_position;

void main() {
    int id;
    if(gl_VertexIndex >= 36) {id = 0;}
    else {id = 3*itable[gl_VertexIndex];}
    vec3 pos = vec3(vtable[id], vtable[id+1], vtable[id+2]);
    v_position = pos;
    gl_Position = (ubo.proj * vec4((ubo.view * vec4(pos, 0.0)).xyz, 1.0)).xyww;
}
