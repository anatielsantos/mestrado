# Import PyGame, PyOpenGL and other public libraries.
import pygame
import numpy as np
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from PIL import Image as Image

def main():

    # Inicialização do PyGame.
    pygame.init()
    display = (800,800)

    # Ligação do PyGame com o OpenGL
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    # gluPerspective( FOV (deg), aspect ratio, z-near, z-far)
    # z-near and z-far referem-se a quando a imagem desaparece
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)

    # Mover o objeto: X, Y, Z (zoom em Z)
    glTranslatef(0.0, 0.0, -5.0)

    # Matriz de rotação (velocity, X, Y, Z do eixo de rotação principal)
    glRotate(0, 0, 0, 0)

    # Importar o arquivo de textura
    # opengl/glass.jpg
    texture_list = ['opengl/earth.jpg', 'opengl/venus.jpg', 'opengl/plutao.jpg', 'opengl/wood.jpg', 'opengl/orange.jpg', 'opengl/glass.jpg']
    textura = Read_Texture(texture_list[0])


    # Loop principal do código
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        
        # controles de transladação e zoom
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                glRotate(0.5,0,0,1)
            if event.key == pygame.K_RIGHT:
                glRotate(0.5,0,0,-1)
            if event.key == pygame.K_UP:
                glRotate(0.5,0,1,0)
            if event.key == pygame.K_DOWN:
                glRotate(0.5,0,-1,0)
            if event.key == pygame.K_w:
                glTranslatef(0.0, -0.1, 0.0)
            if event.key == pygame.K_s:
                glTranslatef(0.0, 0.1, 0.0)
            if event.key == pygame.K_d:
                glTranslatef(-0.1, 0.0, 0)
            if event.key == pygame.K_a:
                glTranslatef(0.1, 0.0, 0)
            if event.key == pygame.K_z:
                glTranslatef(0.0, 0.0, 0.01)
            if event.key == pygame.K_x:
                glTranslatef(0.0, 0.0, -0.01)
            if event.key == pygame.K_r:
                #gluPerspective(45, 0, 0, 0.0)
                glTranslatef((display[0] // 2), (display[1] // 2), 50.0)
                glRotate(0, 0, 0, 0)

        # Cria a esfera e adiciona textura
        # Adicionar segundo parâmetro (r,g,b) para adicionar cor
        cria_esfera(textura, (1,0,1))

        # Iliminação
        light_ambient = [1.0, 0.0, 1.0, 1.0] # Tom da luz ambiente (R, G, B, A)
        light_diffuse = [1.0, 1.0, 1.0, 1.0] # Difusão da luz (RGBA)
        light_specular = [0.0, 0.0, 0.0, 1.0] # Intensidade da luz (RGBA)
        light_position = [2.0, 0.0, 1.0, 0.0] # Posição da luz

        glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient) # Tom
        glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse) # Difusão
        glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular) # Intensidade
        glLightfv(GL_LIGHT0, GL_POSITION, light_position) # Posição

        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_DEPTH_TEST)

        # Atualiza a tela
        pygame.display.flip()
        pygame.time.wait(5)

# Cria uma esfera
def cria_esfera(texture, color=False):
    # Limpa a tela depois de cada frame
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glPushMatrix()

    if (color):
        # cor
        glColor3fv((color))
        esfera = gluNewQuadric()
        gluSphere(esfera, 1, 50, 50)
        gluDeleteQuadric(esfera)
    else:
        # textura
        glEnable(GL_TEXTURE_2D)
        esfera = gluNewQuadric()
        gluQuadricTexture(esfera, GL_TRUE)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, texture)
        gluSphere(esfera, 1, 50, 50)
        gluDeleteQuadric(esfera)
        glDisable(GL_TEXTURE_2D)

    glPopMatrix()

# Função genérica para ler texturas com OpenGL
def Read_Texture(filename):
    img = Image.open(filename)
    img_data = np.array(list(img.getdata()), np.int8)
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.size[0], img.size[1], 0,
                 GL_RGB, GL_UNSIGNED_BYTE, img_data)
    return texture_id

if __name__ == '__main__':
    main()