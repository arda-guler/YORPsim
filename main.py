import numpy as np

import OpenGL
from OpenGL.GL import *
from OpenGL.GLU import *
import glfw
import keyboard as kbd

from load_asteroid import *
from sol import *
from graphics import *
from camera import *
from ui import *

vp_size_changed = False
def resize_cb(window, w, h):
    global vp_size_changed
    vp_size_changed = True

def main():
    global vp_size_changed

    def window_resize(window, width, height):
        try:
            # glfw.get_framebuffer_size(window)
            glViewport(0, 0, width, height)
            glLoadIdentity()
            gluPerspective(fov, width/height, near_clip, far_clip)
            glTranslate(main_cam.pos[0], main_cam.pos[1], main_cam.pos[2])
            main_cam.orient = np.eye(3)
            main_cam.rotate([0, 180, 0])
        except ZeroDivisionError:
            # if the window is minimized it makes height = 0, but we don't need to update projection in that case anyway
            pass

    def move_cam(movement):
        main_cam.move(movement)

    def rotate_cam(rotation):
        main_cam.rotate(rotation)
    
    pos = np.array([-5.450685741567448E+12, 1.490701393404249E+12, -7.878225128952154E+11])
    vel = np.array([-1.872560167799135E+03, -4.824057774345532E+03, 6.744106862311416E+02])

    mass = 2162 * 0.333 * 3.14159 * 50000**3
    moment_of_inertia = np.eye(3) * 1.4e30

    orient = np.eye(3)
    ang_vel = np.array([0.0, 0.0, 0.0])

    faces = read_model_file("data/models/test.obj")
    
    a = Body(pos, vel, orient, ang_vel, mass, moment_of_inertia, faces)

    sol = Sol()

    start_time = 0
    dt = 1e2
    end_time = 1e30
    step_count = int((end_time - start_time) / dt)

    enable_rotation_model = True
    enable_orbit_model = False
    enable_text_output = True

    glfw.init()
    window_x = 1280
    window_y = 720
    mwin = glfw.create_window(window_x, window_y, "YORPsim", None, None)
    glfw.set_window_pos(mwin, 50, 50)
    glfw.make_context_current(mwin)
    glfw.set_window_size_callback(mwin, resize_cb)

    gluPerspective(50, window_x/window_y, 0.1, 10000000)
    glEnable(GL_CULL_FACE)
    glEnable(GL_POINT_SMOOTH)
    glClearColor(0, 0, 0, 1)

    cam_pos = np.array([0, 0, 0])
    cam_orient = np.array([[-1, 0, 0],
                           [0, 1, 0],
                           [0, 0, -1]])
    main_cam = Camera("main_cam", cam_pos, cam_orient, True)

    cam_pitch_up = "K"
    cam_pitch_dn = "I"
    cam_yaw_left = "J"
    cam_yaw_right = "L"
    cam_roll_cw = "O"
    cam_roll_ccw = "U"

    cam_move_up = "R"
    cam_move_dn = "F"
    cam_move_right = "D"
    cam_move_left = "A"
    cam_move_fwd = "W"
    cam_move_bwd = "S"

    cam_rot_speed = 1
    cam_move_speed = 1000

    move_cam([0, 0, 100000])

    for cycle in range(step_count):

        glfw.poll_events()
        if glfw.window_should_close(mwin):
            break

        if vp_size_changed:
            vp_size_changed = False
            w, h = glfw.get_framebuffer_size(mwin)
            glViewport(0, 0, w, h)

        if kbd.is_pressed(cam_pitch_up):
            rotate_cam([cam_rot_speed, 0, 0])
        if kbd.is_pressed(cam_pitch_dn):
            rotate_cam([-cam_rot_speed, 0, 0])
        if kbd.is_pressed(cam_yaw_left):
            rotate_cam([0, cam_rot_speed, 0])
        if kbd.is_pressed(cam_yaw_right):
            rotate_cam([0, -cam_rot_speed, 0])
        if kbd.is_pressed(cam_roll_cw):
            rotate_cam([0, 0, cam_rot_speed])
        if kbd.is_pressed(cam_roll_ccw):
            rotate_cam([0, 0, -cam_rot_speed])

        if kbd.is_pressed(cam_move_up):
            move_cam([0, -cam_move_speed, 0])
        if kbd.is_pressed(cam_move_dn):
            move_cam([0, cam_move_speed, 0])
        if kbd.is_pressed(cam_move_left):
            move_cam([-cam_move_speed, 0, 0])
        if kbd.is_pressed(cam_move_right):
            move_cam([cam_move_speed, 0, 0])
        if kbd.is_pressed(cam_move_fwd):
            move_cam([0, 0, -cam_move_speed])
        if kbd.is_pressed(cam_move_bwd):
            move_cam([0, 0, cam_move_speed])
        
        time = start_time + cycle * dt
        solar_irradiance = sol.luminosity / a.pos**2 # W m-2
        ray = a.pos / np.linalg.norm(a.pos) * solar_irradiance

        solar_forces = []
        application_pts = []
        for face in a.faces:
            face_total_force, face_reflection_force, face_absorbtion_force, face_emission_force\
                              = face.calc_radiation_force(a.orient, ray)
            solar_forces.append(face_total_force)

            application_pts.append(a.calc_face_rel_pos(face))

        solar_torques = []
        solar_lin_forces = []
        for idx_face, face in enumerate(a.faces):
            face_lin_force, face_torque = a.calc_torque_and_force(solar_forces[idx_face], application_pts[idx_face])
            solar_torques.append(face_torque)
            solar_lin_forces.append(face_lin_force)

        if enable_rotation_model:
            ang_accel = np.array([0.0, 0.0, 0.0])
            for torque in solar_torques:
                ang_accel += a.calc_ang_accel(torque)

            ang_vel += ang_accel * dt
            a.orient = a.rotate(ang_vel * dt)
            row_norms = row_norms = np.linalg.norm(a.orient, axis=1, keepdims=True)
            a.orient = a.orient / row_norms

        if enable_orbit_model:
            accel = np.array([0.0, 0.0, 0.0])
            for force in solar_lin_forces:
                accel += force / a.mass

            accel += -(a.pos / np.linalg.norm(a.pos)) * sol.mu / np.linalg.norm(a.pos)**2

            vel += accel * dt
            pos += vel * dt

        a.calc_face_colors()
        a.calc_face_forces()

        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        drawScene(a, sol)
        render_AN("TEST", (1, 1, 1), [0, 0], main_cam, 0.1)
        glfw.swap_buffers(mwin)

        if enable_text_output and cycle % 100 == 0:
            print("------------------")
            print("Pos:", a.pos)
            print("Vel:", a.vel)
            print("Orient:", a.orient)
            print("Ang. Vel.:", a.ang_vel)
            print("------------------")
            print("")

    glfw.destroy_window(mwin)

main()
