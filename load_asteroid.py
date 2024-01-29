import numpy as np

from body import *

def read_model_file(filename):
    points = []
    normals = []
    face_points = []
    
    with open(filename, "r") as f:
        lines = f.readlines()

        for line in lines:
            if line.startswith("v "):
                new_pt = np.array(eval("[" + line[2:].replace(" ", ",") + "]"))
                points.append(new_pt)

            elif line.startswith("vn "):
                new_normal = np.array(eval("[" + line[3:].replace(" ", ",") + "]"))
                normals.append(new_normal)

            elif line.startswith("f "):
                new_face_pts = []
                line = line.split(" ")
                line = line[1:]

                for idx_i, _ in enumerate(line):
                    line[idx_i] = line[idx_i].split("/")
                    new_face_pts.append(int(line[idx_i][0]) - 1)

                face_points.append(new_face_pts)

    face_point_positions = []
    for pointset in face_points:
        face_point_positions.append([])

        for point in pointset:
            face_point_positions[-1].append(points[point])

    areas = []
    face_positions = []

    for idx_face, face in enumerate(face_point_positions):
        ab = face[1] - face[0]
        ac = face[2] - face[0]
        crs = np.cross(ab, ac)
        mag = np.linalg.norm(crs)
        area = abs(0.5 * mag)

        face_pos = (face[0] + face[1] + face[2]) / 3
        
        areas.append(area)
        face_positions.append(face_pos)

    face_instances = []
    for i in range(len(face_positions)):
        new_face = Face(areas[i], normals[i], face_positions[i], 0.2, 0.2, 1000, face_point_positions[i], 273)
        face_instances.append(new_face)

    return face_instances
