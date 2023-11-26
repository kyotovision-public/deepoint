### Some of the visualization code in this file is copied and edited from that of Gaze360.
### You can see the original code by accessing "Updated online demo" on https://github.com/erkil1452/gaze360

from matplotlib.pyplot import imshow, show
from OpenGL.GL import shaders
import glfw
import numpy as np
import OpenGL.GL as gl
from pathlib import Path

WIDTH, HEIGHT = 960, 720
glfw.init()
window = glfw.create_window(WIDTH, HEIGHT, "Draw Arrow Window", None, None)
if not window:
    glfw.terminate()
    print("Failed to create window")
    exit()

glfw.make_context_current(window)

vertexPositions = np.float32([[-1, -1], [1, -1], [-1, 1], [1, 1]])
with (Path(__file__).parent / "shader/vertex_shader.glsl").open("r") as f:
    VERTEX_SHADER = shaders.compileShader(
        f.read(),
        gl.GL_VERTEX_SHADER,
    )


with (Path(__file__).parent / "shader/fragment_shader.glsl").open("r") as f:
    FRAGMENT_SHADER = shaders.compileShader(
        f.read(),
        gl.GL_FRAGMENT_SHADER,
    )

shader = shaders.compileProgram(VERTEX_SHADER, FRAGMENT_SHADER)

xpos = gl.glGetUniformLocation(shader, "xpos")
ypos = gl.glGetUniformLocation(shader, "ypos")

vdir_x = gl.glGetUniformLocation(shader, "vdir_x")
vdir_y = gl.glGetUniformLocation(shader, "vdir_y")
vdir_z = gl.glGetUniformLocation(shader, "vdir_z")

arrow_color = gl.glGetUniformLocation(shader, "arrow_color")

arrow_size = gl.glGetUniformLocation(shader, "size")

arrow_length_mul = gl.glGetUniformLocation(shader, "arrow_length_mul")

res_loc = gl.glGetUniformLocation(shader, "iResolution")


def render_frame(
    x_position: float,
    y_position: float,
    vx: float,
    vy: float,
    vz: float,
    acolor: tuple[float, float, float] = (1.0, 0.0, 0.0),
    asize: float = 0.05,
    alength: float = 1.0,
    offset: float = 0.03,
) -> np.ndarray:
    """
    Draw arrow on image sized (HEIGHT,WIDTH) and returns it.
    Params:
        x_position: x coordinate of arrow root in [-1,1]; right is positive
        y_position: y coordinate of arrow root in [-1,1]; up is positive
        vx: horizontal component of arrow direction. right is positive
        vy: vertical component of arrow direction. up is positive
        vz: depth component of arrow direction. near side is positive
        acolor: RGB color
        asize: up to 0.05 is recommended
        alength: arrow length
        offset: How much the arrow is displaced to the arrow direction
    """
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
    with shader:
        x_position += vx * offset * (asize / 0.05)
        y_position += vy * offset * (asize / 0.05)

        x_position = x_position * 0.89
        y_position = y_position * 0.67
        gl.glUniform1f(xpos, x_position)
        gl.glUniform1f(ypos, y_position)

        gl.glUniform1f(vdir_x, vx)
        gl.glUniform1f(vdir_y, vy)
        gl.glUniform1f(vdir_z, vz)
        gl.glUniform1f(arrow_size, asize)
        gl.glUniform1f(arrow_length_mul, alength)

        gl.glUniform3f(res_loc, WIDTH, HEIGHT, 1.0)

        gl.glUniform3f(arrow_color, acolor[0], acolor[1], acolor[2])

        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, False, 0, vertexPositions)
        gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)
    img_buf = gl.glReadPixels(0, 0, WIDTH, HEIGHT, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
    img = np.frombuffer(img_buf, np.uint8).reshape(HEIGHT, WIDTH, 3)[::-1]
    return img


# For debugging.
if __name__ == "__main__":
    import cv2
    from math import sqrt

    x = 1
    y = 0
    z = 0
    r = sqrt(x**2 + y**2 + z**2)

    img_arrow = (
        render_frame(
            -0.8,
            0,
            x / r,
            y / r,
            z / r,
            acolor=(1, 0, 0),
            asize=0.25,
            offset=0,
        )
        / 255
    )

    cv2.imwrite("arrow.png", img_arrow * 255)

    arrow_mask = (
        (img_arrow[:, :, 0] + img_arrow[:, :, 1] + img_arrow[:, :, 2]) == 0.0
    ).astype(float)[:, :, None]
    img_arrow_alpha = np.concatenate((img_arrow, 1 - arrow_mask), axis=2)
    cv2.imwrite("arrow_alpha.png", img_arrow_alpha * 255)
