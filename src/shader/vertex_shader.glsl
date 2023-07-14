#version 330
layout(location = 0) in vec4 position;
out vec2 UV;
void main()
{
  UV = position.xy*0.5+0.5;
  gl_Position = position;
}
