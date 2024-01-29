import math
import numpy as np

c = 299792458
G = 6.67408e-11
stefanboltzmann = 6.570367e-8

class Face:
    def __init__(self, area, normal, rel_pos, reflectivity, local_heat_capacity, init_temp = 273):
        self.area = area
        self.normal = normal
        self.antinormal = -normal
        self.rel_pos = rel_pos
        self.reflectivity = reflectivity
        self.absorbtivity = 1 - reflectivity
        self.local_heat_capacity = local_heat_capacity
        self.temp = init_temp

    def __repr__(self):
        output = ""
        output += "Area: " + str(self.area) + "\n"
        output += "Normal: " + str(self.normal) + "\n"
        output += "Rel. pos: " + str(self.rel_pos) + "\n"
        output += "\n"

        return output

    def calc_radiation_force(self, orient, ray):
        antinormal = orient * self.antinormal
        
        if np.dot(ray, antinormal) < 0:
            return np.array([0, 0, 0])

        ## REFLECTION FORCE (in antinormal direction of the surface)
        ray_mag = np.linalg.norm(ray)
        cos_ray_normal_angle = np.dot(ray, antinormal) / ray_mag
        force_mag = ray_mag / c
        
        # Power / c = W / (m/s) = Nm/s / (m/s) = N = Force
        reflection_force = antinormal * force_mag * cos_ray_normal_angle * self.reflectivity

        ## ABOSORBTION FORCE (in direction of the ray)
        ray_dir = ray / ray_mag
        absorbtion_force = ray_dir * force_mag * self.absorbtivity

        ## EMISSION FORCE
        emission_power = emissivity * stefanboltzmann * self.temp * self.area
        emission_force_mag = emission_power / c
        emission_force = self.normal * emission_force_mag

        total_force = reflection_force + absorbtion_force + emission_force
        return total_force

    def calc_heating(self, delta_energy):
        return delta_energy / self.local_heat_capacity

class Body:
    def __init__(self, pos, vel, orient, ang_vel, mass, moment_of_inertia, mu, faces):
        self.pos = pos
        self.vel = vel
        self.orient = orient
        self.ang_vel = ang_vel
        self.mass = mass
        self.moment_of_inertia = moment_of_inertia
        self.mu = G * mass
        self.faces = faces

    def calc_gravitational_accel(self, other_body):
        dist = np.linalg.norm(self.pos - other_body.pos)
        grav_accel_dir = (other_body.pos - self.pos) / dist
        grav_accel_mag = other_body.mu / dist**2
        grav_accel = grav_accel_dir * grav_accel_mag
        return grav_accel

    def calc_torque_and_force(self, force, point_of_application):
        force_mag = np.linalg.norm(force)
        force_dir = force / force_mag

        torque_arm = point_of_application
        torque = np.cross(torque_arm, force)

        linear_force = force

        return linear_force, torque

    def calc_face_rel_pos(self, face):
        face_pos = face.rel_pos
        global_rel_pos = self.orient * face_pos
        return global_rel_pos

    def calc_ang_accel(self, torque):
        ang_accel = np.linalg.solve(self.moment_of_inertia, torque)
        return ang_accel

    def rotate(self, rotation):
        rot_x = rotation[0]
        rot_y = rotation[1]
        rot_z = rotation[2]

        Rx = np.array([[1, 0,             0             ],
                       [0, np.cos(rot_x), -np.sin(rot_x)],
                       [0, np.sin(rot_x), np.cos(rot_x)]])

        Ry = np.array([[np.cos(rot_y), 0, np.sin(rot_y)],
                       [0,             1, 0            ],
                       [-np.sin(rot_y),0, np.cos(rot_y)]])

        Rz = np.array([[np.cos(rot_z), -np.sin(rot_z), 0],
                       [np.sin(rot_z), np.cos(rot_z),  0],
                       [0,             0,              1]])

        Rrot = Rz * Ry * Rz

        Rnew = Rrot * self.orient
        return Rnew
