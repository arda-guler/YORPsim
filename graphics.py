import OpenGL
from OpenGL.GL import *
from OpenGL.GLU import *

import numpy as np

def drawOrigin():
    glBegin(GL_LINES)
    glColor(1,0,0)
    glVertex3f(0,0,0)
    glVertex3f(10000,0,0)
    glColor(0,1,0)
    glVertex3f(0,0,0)
    glVertex3f(0,10000,0)
    glColor(0,0,1)
    glVertex3f(0,0,0)
    glVertex3f(0,0,10000)
    glEnd()

def drawForces(forces, scaler = 50):
    
    for f in forces:
        glPushMatrix()

        start_position = f.pos
        end_position = f.pos + f.force
        f_vector = f.force * scaler
        
        f_dir = f_vector / np.linalg.norm(f_vector)
        arrowhead_start = f.force * scaler * 0.8

        if not np.linalg.norm(np.cross(f_dir, np.array([1, 0, 0]))) == 0:
            arrowhead_vector1 = np.cross(f_dir, np.array([1, 0, 0]))
        else:
            arrowhead_vector1 = np.cross(f_dir, np.array([0, 1, 0]))

        arrowhead_vector2 = np.cross(arrowhead_vector1, f_dir)

        arrowhead_vector1 = arrowhead_vector1 * np.linalg.norm(f.force) * scaler * 0.1
        arrowhead_vector2 = arrowhead_vector2 * np.linalg.norm(f.force) * scaler * 0.1
            
        arrowhead_pt1 = arrowhead_start + arrowhead_vector1
        arrowhead_pt2 = arrowhead_start - arrowhead_vector1

        arrowhead_pt3 = arrowhead_start + arrowhead_vector2
        arrowhead_pt4 = arrowhead_start - arrowhead_vector2
        
        glTranslate(start_position[0], start_position[1], start_position[2])
        glColor(1,0,1)

        glBegin(GL_LINES)

        glVertex3f(0,0,0)
        glVertex3f(f_vector[0], f_vector[1], f_vector[2])

        glVertex3f(arrowhead_pt1[0], arrowhead_pt1[1], arrowhead_pt1[2])
        glVertex3f(arrowhead_pt3[0], arrowhead_pt3[1], arrowhead_pt3[2])

        glVertex3f(arrowhead_pt2[0], arrowhead_pt2[1], arrowhead_pt2[2])
        glVertex3f(arrowhead_pt4[0], arrowhead_pt4[1], arrowhead_pt4[2])

        glVertex3f(arrowhead_pt2[0], arrowhead_pt2[1], arrowhead_pt2[2])
        glVertex3f(arrowhead_pt3[0], arrowhead_pt3[1], arrowhead_pt3[2])

        glVertex3f(arrowhead_pt1[0], arrowhead_pt1[1], arrowhead_pt1[2])
        glVertex3f(arrowhead_pt4[0], arrowhead_pt4[1], arrowhead_pt4[2])

        glVertex3f(arrowhead_pt1[0], arrowhead_pt1[1], arrowhead_pt1[2])
        glVertex3f(f_vector[0], f_vector[1], f_vector[2])

        glVertex3f(arrowhead_pt2[0], arrowhead_pt2[1], arrowhead_pt2[2])
        glVertex3f(f_vector[0], f_vector[1], f_vector[2])

        glVertex3f(arrowhead_pt3[0], arrowhead_pt3[1], arrowhead_pt3[2])
        glVertex3f(f_vector[0], f_vector[1], f_vector[2])

        glVertex3f(arrowhead_pt4[0], arrowhead_pt4[1], arrowhead_pt4[2])
        glVertex3f(f_vector[0], f_vector[1], f_vector[2])

        glEnd()

        glPopMatrix()

def drawAsteroid(a):
    glPushMatrix()
    multarray = np.array([[a.orient[0][0], a.orient[0][1], a.orient[0][2], 0],
                          [a.orient[1][0], a.orient[1][1], a.orient[1][2], 0],
                          [a.orient[2][0], a.orient[2][1], a.orient[2][2], 0],
                          [0, 0, 0, 1]])
    glMultMatrixf(multarray)

    for idx_face, face in enumerate(a.faces):
        glColor(a.colors[idx_face][0], a.colors[idx_face][1], a.colors[idx_face][2], 1)
        glBegin(GL_LINES)
        glVertex3f(face.pts[0][0], face.pts[0][1], face.pts[0][2])
        glVertex3f(face.pts[1][0], face.pts[1][1], face.pts[1][2])

        glVertex3f(face.pts[0][0], face.pts[0][1], face.pts[0][2])
        glVertex3f(face.pts[2][0], face.pts[2][1], face.pts[2][2])

        glVertex3f(face.pts[1][0], face.pts[1][1], face.pts[1][2])
        glVertex3f(face.pts[2][0], face.pts[2][1], face.pts[2][2])
        glEnd()

    glPopMatrix()

def drawSol(sol, a):
    glColor(0.99, 0.99, 0.66)
    glPointSize(5)
    glBegin(GL_POINTS)
    glVertex3f(a.pos[1], a.pos[2], -a.pos[0]) # some coordinate transformation happens here
                                              # TODO: Recheck this!
    glEnd()
    glPointSize(1)

def drawScene(a, sol):
    drawOrigin()
    drawAsteroid(a)
    drawSol(sol, a)
    drawForces(a.fvises)
