"""
OpenGL Shaders for rendering 3D Meshes in ModernGL.
"""

SCENE_VERTEX_SHADER = """
#version 330 core

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;

in vec3 in_position;
in vec3 in_normal;
in vec2 in_uv;

out vec3 v_world_pos;
out vec3 v_normal;
out vec2 v_uv;

void main() {
    // TODO: Transform vertex position and normal to world space
    // TODO: Pass UV coordinates to fragment shader
}
"""

SCENE_FRAGMENT_SHADER = """
#version 330 core

uniform vec3 u_light_pos;
uniform vec3 u_cam_pos;
uniform vec3 u_color;
uniform sampler2D u_texture;
uniform bool u_use_texture;

in vec3 v_world_pos;
in vec3 v_normal;
in vec2 v_uv;

layout(location = 0) out vec4 frag_color;

void main() {
    
    //TODO: implement light direction and normal to continue next step N dot light_direction

    //TODO: ambient
    vec3 ambient = 

    //TODO: diffuse
    vec3 diffuse = 

    //TODO: specular

    vec3 specular = 

    //TODO: Set the fragment color
    frag_color = 
}
"""
