# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ##### END GPL LICENSE BLOCK #####

bl_info = {"name": "Point Cloud Visualizer",
           "description": "Display, edit, filter, render, convert, generate and export colored point cloud PLY files.",
           "author": "Jakub Uhlik",
           "version": (0, 9, 30),
           "blender": (2, 81, 0),
           "location": "View3D > Sidebar > Point Cloud Visualizer",
           "warning": "",
           "wiki_url": "https://github.com/uhlik/bpy",
           "tracker_url": "https://github.com/uhlik/bpy/issues",
           "category": "3D View", }

import os
import sys
import os.path
import bpy
import bmesh
import math
import struct
import uuid
import datetime
import math
import numpy as np
import time
import shutil
import random
from bpy.props import PointerProperty, BoolProperty, StringProperty, FloatProperty, IntProperty, FloatVectorProperty, \
    EnumProperty, CollectionProperty
from bpy.types import PropertyGroup, Panel, Operator, AddonPreferences, UIList
import gpu
from gpu.types import GPUOffScreen, GPUShader, GPUBatch, GPUVertBuf, GPUVertFormat
from gpu_extras.batch import batch_for_shader
from bpy.app.handlers import persistent
import bgl
from mathutils import Matrix, Vector, Quaternion, Color
from bpy_extras.object_utils import world_to_camera_view
from bpy_extras.io_utils import axis_conversion, ExportHelper
from mathutils.kdtree import KDTree
from mathutils.geometry import barycentric_transform
from mathutils.interpolate import poly_3d_calc
from mathutils.bvhtree import BVHTree
import mathutils.geometry

def log(msg, indent=0, prefix='>', ):
    m = "{}{} {}".format("    " * indent, prefix, msg)
    if (debug_mode()):
        print(m)

def debug_mode():
    # return True
    return (bpy.app.debug_value != 0)


class data_Storage():
    uuid = None
    points = None
    depth_Camera = None
    point_Cloud_Axis = bpy.data.objects['point_Cloud_Axis']
    FLAG_PROGRAM_START = False
    FLAG_GET_CAMERA_DATA = False
    FLAG_PLY_LOAD_FINISHED = False
    FLAG_DELETE_POINT_CLOUD = False
    FLAG_KEEP_COLLECT_DEPTH_INFO = False
    FLAG_RESTRICT_MAKE_POINT_CLOUD = False

class PlyPointCloudReader():
    _supported_formats = ('binary_little_endian', 'binary_big_endian', 'ascii',)
    _supported_versions = ('1.0',)
    _byte_order = {'binary_little_endian': '<', 'binary_big_endian': '>', 'ascii': None, }
    # _types = {'char': 'c', 'uchar': 'B', 'short': 'h', 'ushort': 'H', 'int': 'i', 'uint': 'I', 'float': 'f', 'double': 'd', }
    _types = {
        'char': 'b',
        'uchar': 'B',
        'int8': 'b',
        'uint8': 'B',
        'int16': 'h',
        'uint16': 'H',
        'short': 'h',
        'ushort': 'H',
        'int': 'i',
        'int32': 'i',
        'uint': 'I',
        'uint32': 'I',
        'float': 'f',
        'float32': 'f',
        'float64': 'd',
        'double': 'd',
        'string': 's',
    }

    def __init__(self, path, ):
        log("{}:".format(self.__class__.__name__), 0)
        if (os.path.exists(path) is False or os.path.isdir(path) is True):
            raise OSError("did you point me to an imaginary file? ('{}')".format(path))

        self.path = path

        self._header()
        if (self._ply_format == 'ascii'):
            self._data_ascii()
        else:
            self._data_binary()
        print("loaded {} vertices".format(len(self.points)))

        # remove alpha if present (meshlab adds it)
        self.points = self.points[[b for b in list(self.points.dtype.names) if b != 'alpha']]

        # rename diffuse_rgb to rgb, if present
        user_rgb = ('diffuse_red', 'diffuse_green', 'diffuse_blue',)
        names = self.points.dtype.names
        ls = list(names)
        if (set(user_rgb).issubset(names)):
            for ci, uc in enumerate(user_rgb):
                for i, v in enumerate(ls):
                    if (v == uc):
                        ls[i] = ls[i].replace('diffuse_', '', )
        self.points.dtype.names = tuple(ls)

        # remove anything that is not (x, y, z, nx, ny, nz, red, green, blue) to prevent problems later
        self.points = self.points[[b for b in list(self.points.dtype.names) if
                                   b in ('x', 'y', 'z', 'nx', 'ny', 'nz', 'red', 'green', 'blue',)]]

        # some info
        nms = self.points.dtype.names
        self.has_vertices = True
        self.has_normals = True
        self.has_colors = True
        if (not set(('x', 'y', 'z')).issubset(nms)):
            self.has_vertices = False
        if (not set(('nx', 'ny', 'nz')).issubset(nms)):
            self.has_normals = False
        if (not set(('red', 'green', 'blue')).issubset(nms)):
            self.has_colors = False
        # print('has_vertices: {}'.format(self.has_vertices))
        # print('has_normals: {}'.format(self.has_normals))
        # print('has_colors: {}'.format(self.has_colors))

    def _header(self):
        raw = []
        h = []
        with open(self.path, mode='rb') as f:
            for l in f:
                raw.append(l)
                a = l.decode('ascii').rstrip()
                h.append(a)
                if (a == "end_header"):
                    break

        if (h[0] != 'ply'):
            raise TypeError("not a ply file")
        for i, l in enumerate(h):
            if (l.startswith('format')):
                _, f, v = l.split(' ')
                if (f not in self._supported_formats):
                    raise TypeError("unsupported ply format")
                if (v not in self._supported_versions):
                    raise TypeError("unsupported ply file version")
                self._ply_format = f
                self._ply_version = v
                if (self._ply_format != 'ascii'):
                    self._endianness = self._byte_order[self._ply_format]

        self._elements = []
        current_element = None
        for i, l in enumerate(h):
            if (l.startswith('ply')):
                pass
            elif (l.startswith('format')):
                pass
            elif (l.startswith('comment')):
                pass
            elif (l.startswith('element')):
                _, t, c = l.split(' ')
                a = {'type': t, 'count': int(c), 'props': [], }
                self._elements.append(a)
                current_element = a
            elif (l.startswith('property')):
                if (l.startswith('property list')):
                    _, _, c, t, n = l.split(' ')
                    if (self._ply_format == 'ascii'):
                        current_element['props'].append((n, self._types[c], self._types[t],))
                    else:
                        current_element['props'].append((n, self._types[c], self._types[t],))
                else:
                    _, t, n = l.split(' ')
                    if (self._ply_format == 'ascii'):
                        current_element['props'].append((n, self._types[t]))
                    else:
                        current_element['props'].append((n, self._types[t]))
            elif (l.startswith('end_header')):
                pass
            else:
                log('unknown header line: {}'.format(l))

        if (self._ply_format == 'ascii'):
            skip = False
            flen = 0
            hlen = 0
            with open(self.path, mode='r', encoding='utf-8') as f:
                for i, l in enumerate(f):
                    flen += 1
                    if (skip):
                        continue
                    hlen += 1
                    if (l.rstrip() == 'end_header'):
                        skip = True
            self._header_length = hlen
            self._file_length = flen
        else:
            self._header_length = sum([len(i) for i in raw])

    def _data_binary(self):
        self.points = []

        read_from = self._header_length
        for ie, element in enumerate(self._elements):
            if (element['type'] != 'vertex'):
                continue

            dtp = []
            for i, p in enumerate(element['props']):
                n, t = p
                dtp.append((n, '{}{}'.format(self._endianness, t),))
            dt = np.dtype(dtp)
            with open(self.path, mode='rb') as f:
                f.seek(read_from)
                a = np.fromfile(f, dtype=dt, count=element['count'], )

            self.points = a
            read_from += element['count']

    def _data_ascii(self):
        self.points = []

        skip_header = self._header_length
        skip_footer = self._file_length - self._header_length
        for ie, element in enumerate(self._elements):
            if (element['type'] != 'vertex'):
                continue

            skip_footer = skip_footer - element['count']
            with open(self.path, mode='r', encoding='utf-8') as f:
                a = np.genfromtxt(f, dtype=np.dtype(element['props']), skip_header=skip_header,
                                  skip_footer=skip_footer, )
            self.points = a
            skip_header += element['count']


class PCVShaders():
    vertex_shader_illumination = '''
        in vec3 position;
        in vec3 normal;
        in vec4 color;

        // uniform float show_illumination;
        uniform vec3 light_direction;
        uniform vec3 light_intensity;
        uniform vec3 shadow_direction;
        uniform vec3 shadow_intensity;
        // uniform float show_normals;

        uniform mat4 perspective_matrix;
        uniform mat4 object_matrix;
        uniform float point_size;
        uniform float alpha_radius;
        uniform float global_alpha;

        out vec4 f_color;
        out float f_alpha_radius;
        out vec3 f_normal;

        out vec3 f_light_direction;
        out vec3 f_light_intensity;
        out vec3 f_shadow_direction;
        out vec3 f_shadow_intensity;
        // out float f_show_normals;
        // out float f_show_illumination;

        void main()
        {
            gl_Position = perspective_matrix * object_matrix * vec4(position, 1.0f);
            gl_PointSize = point_size;
            f_normal = normal;
            // f_color = color;
            f_color = vec4(color[0], color[1], color[2], global_alpha);
            f_alpha_radius = alpha_radius;

            // f_light_direction = normalize(vec3(inverse(object_matrix) * vec4(light_direction, 1.0)));
            f_light_direction = light_direction;
            f_light_intensity = light_intensity;
            // f_shadow_direction = normalize(vec3(inverse(object_matrix) * vec4(shadow_direction, 1.0)));
            f_shadow_direction = shadow_direction;
            f_shadow_intensity = shadow_intensity;
            // f_show_normals = show_normals;
            // f_show_illumination = show_illumination;
        }
    '''
    fragment_shader_illumination = '''
        in vec4 f_color;
        in vec3 f_normal;
        in float f_alpha_radius;

        in vec3 f_light_direction;
        in vec3 f_light_intensity;
        in vec3 f_shadow_direction;
        in vec3 f_shadow_intensity;
        // in float f_show_normals;
        // in float f_show_illumination;

        out vec4 fragColor;

        void main()
        {
            float r = 0.0f;
            float a = 1.0f;
            vec2 cxy = 2.0f * gl_PointCoord - 1.0f;
            r = dot(cxy, cxy);
            if(r > f_alpha_radius){
                discard;
            }
            // fragColor = f_color * a;

            vec4 col;

            // if(f_show_normals > 0.5){
            //     col = vec4(f_normal, 1.0) * a;
            // }else if(f_show_illumination > 0.5){

            // if(f_show_illumination > 0.5){
            //     vec4 light = vec4(max(dot(f_light_direction, -f_normal), 0) * f_light_intensity, 1);
            //     vec4 shadow = vec4(max(dot(f_shadow_direction, -f_normal), 0) * f_shadow_intensity, 1);
            //     col = (f_color + light - shadow) * a;
            // }else{
            //     col = f_color * a;
            // }

            vec4 light = vec4(max(dot(f_light_direction, -f_normal), 0) * f_light_intensity, 1);
            vec4 shadow = vec4(max(dot(f_shadow_direction, -f_normal), 0) * f_shadow_intensity, 1);
            col = (f_color + light - shadow) * a;

            fragColor = col;
        }
    '''

    vertex_shader_simple = '''
        in vec3 position;
        in vec4 color;
        uniform mat4 perspective_matrix;
        uniform mat4 object_matrix;
        uniform float point_size;
        uniform float alpha_radius;
        uniform float global_alpha;
        out vec4 f_color;
        out float f_alpha_radius;
        void main()
        {
            gl_Position = perspective_matrix * object_matrix * vec4(position, 1.0f);
            gl_PointSize = point_size;
            // f_color = color;
            f_color = vec4(color[0], color[1], color[2], global_alpha);
            f_alpha_radius = alpha_radius;
        }
    '''
    fragment_shader_simple = '''
        in vec4 f_color;
        in float f_alpha_radius;
        out vec4 fragColor;
        void main()
        {
            float r = 0.0f;
            float a = 1.0f;
            vec2 cxy = 2.0f * gl_PointCoord - 1.0f;
            r = dot(cxy, cxy);
            if(r > f_alpha_radius){
                discard;
            }
            fragColor = f_color * a;
        }
    '''

    normals_vertex_shader = '''
        layout(location = 0) in vec3 position;
        layout(location = 1) in vec3 normal;

        out vec3 vertex_normal;

        void main()
        {
            vertex_normal = normal;
            gl_Position = vec4(position, 1.0);
        }
    '''
    normals_fragment_shader = '''
        layout(location = 0) out vec4 frag_color;

        uniform float global_alpha;
        in vec4 vertex_color;

        void main()
        {
            // frag_color = vertex_color;
            frag_color = vec4(vertex_color[0], vertex_color[1], vertex_color[2], vertex_color[3] * global_alpha);
        }
    '''
    normals_geometry_shader = '''
        layout(points) in;
        layout(line_strip, max_vertices = 2) out;

        uniform mat4 perspective_matrix;
        uniform mat4 object_matrix;
        uniform float length = 1.0;
        uniform vec4 color = vec4(1.0, 1.0, 1.0, 1.0);

        in vec3 vertex_normal[];

        out vec4 vertex_color;

        void main()
        {
            vec3 normal = vertex_normal[0];

            vertex_color = color;

            vec4 v0 = gl_in[0].gl_Position;
            gl_Position = perspective_matrix * object_matrix * v0;
            EmitVertex();

            vec4 v1 = v0 + vec4(normal * length, 0.0);
            gl_Position = perspective_matrix * object_matrix * v1;
            EmitVertex();

            EndPrimitive();
        }
    '''

    depth_vertex_shader_simple = '''
        in vec3 position;
        uniform mat4 perspective_matrix;
        uniform mat4 object_matrix;
        uniform vec3 center;
        uniform float point_size;
        uniform float maxdist;
        out float f_depth;
        void main()
        {
            gl_Position = perspective_matrix * object_matrix * vec4(position, 1.0);
            gl_PointSize = point_size;
            vec4 pp = perspective_matrix * object_matrix * vec4(position, 1.0);
            vec4 op = perspective_matrix * object_matrix * vec4(center, 1.0);
            float d = op.z - pp.z;
            f_depth = ((d - (-maxdist)) / (maxdist - d)) / 2;
        }
    '''
    depth_fragment_shader_simple = '''
        in float f_depth;
        uniform float brightness;
        uniform float contrast;
        uniform float alpha_radius;
        uniform float global_alpha;
        out vec4 fragColor;
        void main()
        {
            float r = 0.0f;
            float a = 1.0f;
            vec2 cxy = 2.0f * gl_PointCoord - 1.0f;
            r = dot(cxy, cxy);
            if(r > alpha_radius){
                discard;
            }
            vec3 color = vec3(f_depth, f_depth, f_depth);
            color = (color - 0.5) * contrast + 0.5 + brightness;
            fragColor = vec4(color, global_alpha) * a;
        }
    '''

    selection_vertex_shader = '''
        in vec3 position;
        uniform mat4 perspective_matrix;
        uniform mat4 object_matrix;
        uniform float point_size;
        void main()
        {
            gl_Position = perspective_matrix * object_matrix * vec4(position, 1.0f);
            gl_PointSize = point_size;
        }
    '''
    selection_fragment_shader = '''
        uniform vec4 color;
        uniform float alpha_radius;
        out vec4 fragColor;
        void main()
        {
            float r = 0.0f;
            float a = 1.0f;
            vec2 cxy = 2.0f * gl_PointCoord - 1.0f;
            r = dot(cxy, cxy);
            if(r > alpha_radius){
                discard;
            }
            fragColor = color * a;
        }
    '''

    normal_colors_vertex_shader = '''
        in vec3 position;
        in vec3 normal;
        uniform mat4 perspective_matrix;
        uniform mat4 object_matrix;
        uniform float point_size;
        out vec3 f_color;
        void main()
        {
            gl_Position = perspective_matrix * object_matrix * vec4(position, 1.0f);
            gl_PointSize = point_size;
            f_color = normal * 0.5 + 0.5;
            // f_color = normal;
        }
    '''
    normal_colors_fragment_shader = '''
        // uniform vec4 color;
        in vec3 f_color;
        uniform float alpha_radius;
        uniform float global_alpha;
        out vec4 fragColor;
        void main()
        {
            float r = 0.0f;
            float a = 1.0f;
            vec2 cxy = 2.0f * gl_PointCoord - 1.0f;
            r = dot(cxy, cxy);
            if(r > alpha_radius){
                discard;
            }
            fragColor = vec4(f_color, global_alpha) * a;
        }
    '''

    depth_vertex_shader_illumination = '''
        in vec3 position;
        in vec3 normal;
        uniform mat4 perspective_matrix;
        uniform mat4 object_matrix;
        uniform vec3 center;
        uniform float point_size;
        uniform float maxdist;
        uniform vec3 light_direction;
        uniform vec3 light_intensity;
        uniform vec3 shadow_direction;
        uniform vec3 shadow_intensity;
        out float f_depth;
        out vec3 f_light_direction;
        out vec3 f_light_intensity;
        out vec3 f_shadow_direction;
        out vec3 f_shadow_intensity;
        out vec3 f_normal;
        void main()
        {
            gl_Position = perspective_matrix * object_matrix * vec4(position, 1.0);
            gl_PointSize = point_size;
            vec4 pp = perspective_matrix * object_matrix * vec4(position, 1.0);
            vec4 op = perspective_matrix * object_matrix * vec4(center, 1.0);
            float d = op.z - pp.z;
            f_depth = ((d - (-maxdist)) / (maxdist - d)) / 2;
            f_normal = normal;
            f_light_direction = light_direction;
            f_light_intensity = light_intensity;
            f_shadow_direction = shadow_direction;
            f_shadow_intensity = shadow_intensity;
        }
    '''
    depth_fragment_shader_illumination = '''
        in float f_depth;
        in vec3 f_normal;
        in vec3 f_light_direction;
        in vec3 f_light_intensity;
        in vec3 f_shadow_direction;
        in vec3 f_shadow_intensity;
        uniform float alpha_radius;
        uniform float global_alpha;
        uniform float brightness;
        uniform float contrast;
        uniform vec3 color_a;
        uniform vec3 color_b;
        out vec4 fragColor;
        void main()
        {
            float r = 0.0f;
            float a = 1.0f;
            vec2 cxy = 2.0f * gl_PointCoord - 1.0f;
            r = dot(cxy, cxy);
            if(r > alpha_radius){
                discard;
            }
            vec3 l = vec3(max(dot(f_light_direction, -f_normal), 0) * f_light_intensity);
            vec3 s = vec3(max(dot(f_shadow_direction, -f_normal), 0) * f_shadow_intensity);
            vec3 color = mix(color_b, color_a, f_depth);
            // brightness/contrast after illumination
            // vec3 c = color + l - s;
            // vec3 cc = (c - 0.5) * contrast + 0.5 + brightness;
            // fragColor = vec4(cc, global_alpha) * a;

            // brightness/contrast before illumination
            vec3 cc = (color - 0.5) * contrast + 0.5 + brightness;
            vec3 c = cc + l - s;
            fragColor = vec4(c, global_alpha) * a;
        }
    '''
    depth_vertex_shader_false_colors = '''
        in vec3 position;
        uniform mat4 perspective_matrix;
        uniform mat4 object_matrix;
        uniform vec3 center;
        uniform float point_size;
        uniform float maxdist;
        out float f_depth;
        void main()
        {
            gl_Position = perspective_matrix * object_matrix * vec4(position, 1.0);
            gl_PointSize = point_size;
            vec4 pp = perspective_matrix * object_matrix * vec4(position, 1.0);
            vec4 op = perspective_matrix * object_matrix * vec4(center, 1.0);
            float d = op.z - pp.z;
            f_depth = ((d - (-maxdist)) / (maxdist - d)) / 2;
        }
    '''
    depth_fragment_shader_false_colors = '''
        in float f_depth;
        uniform float alpha_radius;
        uniform float global_alpha;
        uniform float brightness;
        uniform float contrast;
        uniform vec3 color_a;
        uniform vec3 color_b;
        out vec4 fragColor;
        void main()
        {
            float r = 0.0f;
            float a = 1.0f;
            vec2 cxy = 2.0f * gl_PointCoord - 1.0f;
            r = dot(cxy, cxy);
            if(r > alpha_radius){
                discard;
            }
            vec3 color = mix(color_b, color_a, f_depth);
            color = (color - 0.5) * contrast + 0.5 + brightness;
            fragColor = vec4(color, global_alpha) * a;
        }
    '''

    position_colors_vertex_shader = '''
        in vec3 position;
        uniform mat4 perspective_matrix;
        uniform mat4 object_matrix;
        uniform float point_size;
        out vec3 f_color;
        void main()
        {
            gl_Position = perspective_matrix * object_matrix * vec4(position, 1.0f);
            gl_PointSize = point_size;
            // f_color = position * 0.5 + 0.5;
            f_color = position;
        }
    '''
    position_colors_fragment_shader = '''
        in vec3 f_color;
        uniform float alpha_radius;
        uniform float global_alpha;
        out vec4 fragColor;
        void main()
        {
            float r = 0.0f;
            float a = 1.0f;
            vec2 cxy = 2.0f * gl_PointCoord - 1.0f;
            r = dot(cxy, cxy);
            if(r > alpha_radius){
                discard;
            }
            fragColor = vec4(mod(f_color, 1.0), global_alpha) * a;
        }
    '''

    bbox_vertex_shader = '''
        layout(location = 0) in vec3 position;

        void main()
        {
            gl_Position = vec4(position, 1.0);
        }
    '''
    bbox_fragment_shader = '''
        layout(location = 0) out vec4 frag_color;

        uniform float global_alpha;
        in vec4 vertex_color;

        void main()
        {
            frag_color = vec4(vertex_color.rgb, vertex_color[3] * global_alpha);
        }
    '''
    bbox_geometry_shader = '''
        layout(points) in;
        layout(line_strip, max_vertices = 256) out;

        uniform mat4 perspective_matrix;
        uniform mat4 object_matrix;
        uniform vec4 color = vec4(1.0, 1.0, 1.0, 1.0);

        uniform float length = 0.1;
        uniform vec3 center = vec3(0.0, 0.0, 0.0);
        uniform vec3 dimensions = vec3(1.0, 1.0, 1.0);

        out vec4 vertex_color;

        void line();

        void line(vec4 o, vec3 a, vec3 b)
        {
            gl_Position = perspective_matrix * object_matrix * (o + vec4(a, 0.0));
            EmitVertex();
            gl_Position = perspective_matrix * object_matrix * (o + vec4(b, 0.0));
            EmitVertex();
            EndPrimitive();
        }

        void main()
        {
            vertex_color = color;

            //vec4 o = gl_in[0].gl_Position;
            vec4 o = vec4(center, 1.0);

            float w = dimensions[0] / 2;
            float h = dimensions[1] / 2;
            float d = dimensions[2] / 2;
            float l = length;

            vec3 p00 = vec3(-(w - l),       -h,       -d);
            vec3 p01 = vec3(      -w,       -h,       -d);
            vec3 p02 = vec3(      -w,       -h, -(d - l));
            vec3 p03 = vec3(      -w, -(h - l),       -d);
            vec3 p04 = vec3(-(w - l),       -h,        d);
            vec3 p05 = vec3(      -w,       -h,        d);
            vec3 p06 = vec3(      -w, -(h - l),        d);
            vec3 p07 = vec3(      -w,       -h,  (d - l));
            vec3 p08 = vec3(      -w,  (h - l),       -d);
            vec3 p09 = vec3(      -w,        h,       -d);
            vec3 p10 = vec3(      -w,        h, -(d - l));
            vec3 p11 = vec3(-(w - l),        h,       -d);
            vec3 p12 = vec3(-(w - l),        h,        d);
            vec3 p13 = vec3(      -w,        h,        d);
            vec3 p14 = vec3(      -w,        h,  (d - l));
            vec3 p15 = vec3(      -w,  (h - l),        d);
            vec3 p16 = vec3(       w, -(h - l),       -d);
            vec3 p17 = vec3(       w,       -h,       -d);
            vec3 p18 = vec3(       w,       -h, -(d - l));
            vec3 p19 = vec3( (w - l),       -h,       -d);
            vec3 p20 = vec3( (w - l),       -h,        d);
            vec3 p21 = vec3(       w,       -h,        d);
            vec3 p22 = vec3(       w,       -h,  (d - l));
            vec3 p23 = vec3(       w, -(h - l),        d);
            vec3 p24 = vec3( (w - l),        h,       -d);
            vec3 p25 = vec3(       w,        h,       -d);
            vec3 p26 = vec3(       w,        h, -(d - l));
            vec3 p27 = vec3(       w,  (h - l),       -d);
            vec3 p28 = vec3(       w,  (h - l),        d);
            vec3 p29 = vec3(       w,        h,        d);
            vec3 p30 = vec3(       w,        h,  (d - l));
            vec3 p31 = vec3( (w - l),        h,        d);

            line(o, p00, p01);
            line(o, p01, p03);
            line(o, p02, p01);
            line(o, p04, p05);
            line(o, p05, p07);
            line(o, p06, p05);
            line(o, p08, p09);
            line(o, p09, p11);
            line(o, p10, p09);
            line(o, p12, p13);
            line(o, p13, p15);
            line(o, p14, p13);
            line(o, p16, p17);
            line(o, p17, p19);
            line(o, p18, p17);
            line(o, p20, p21);
            line(o, p21, p23);
            line(o, p22, p21);
            line(o, p24, p25);
            line(o, p25, p27);
            line(o, p26, p25);
            line(o, p28, p29);
            line(o, p29, p31);
            line(o, p30, p29);

        }
    '''

    vertex_shader_color_adjustment = '''
        in vec3 position;
        in vec4 color;
        uniform mat4 perspective_matrix;
        uniform mat4 object_matrix;
        uniform float point_size;
        uniform float alpha_radius;
        uniform float global_alpha;

        uniform float exposure;
        uniform float gamma;
        uniform float brightness;
        uniform float contrast;
        uniform float hue;
        uniform float saturation;
        uniform float value;
        uniform float invert;

        out vec4 f_color;
        out float f_alpha_radius;

        out float f_exposure;
        out float f_gamma;
        out float f_brightness;
        out float f_contrast;
        out float f_hue;
        out float f_saturation;
        out float f_value;
        out float f_invert;

        void main()
        {
            gl_Position = perspective_matrix * object_matrix * vec4(position, 1.0f);
            gl_PointSize = point_size;
            f_color = vec4(color[0], color[1], color[2], global_alpha);
            f_alpha_radius = alpha_radius;

            f_exposure = exposure;
            f_gamma = gamma;
            f_brightness = brightness;
            f_contrast = contrast;
            f_hue = hue;
            f_saturation = saturation;
            f_value = value;
            f_invert = invert;
        }
    '''
    fragment_shader_color_adjustment = '''
        in vec4 f_color;
        in float f_alpha_radius;
        out vec4 fragColor;

        in float f_exposure;
        in float f_gamma;
        in float f_brightness;
        in float f_contrast;
        in float f_hue;
        in float f_saturation;
        in float f_value;
        in float f_invert;

        // https://stackoverflow.com/questions/15095909/from-rgb-to-hsv-in-opengl-glsl
        vec3 rgb2hsv(vec3 c)
        {
            vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
            vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
            vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));

            float d = q.x - min(q.w, q.y);
            float e = 1.0e-10;
            return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
        }
        vec3 hsv2rgb(vec3 c)
        {
            vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
            vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
            return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
        }

        void main()
        {
            float r = 0.0f;
            float a = 1.0f;
            vec2 cxy = 2.0f * gl_PointCoord - 1.0f;
            r = dot(cxy, cxy);
            if(r > f_alpha_radius){
                discard;
            }
            fragColor = f_color * a;

            // adjustments
            vec3 rgb = fragColor.rgb;
            float alpha = fragColor.a;
            vec3 color = rgb;

            // exposure
            color = clamp(color * pow(2, f_exposure), 0.0, 1.0);
            // gamma
            color = clamp(vec3(pow(color[0], 1 / f_gamma), pow(color[1], 1 / f_gamma), pow(color[2], 1 / f_gamma)), 0.0, 1.0);

            // brightness/contrast
            color = clamp((color - 0.5) * f_contrast + 0.5 + f_brightness, 0.0, 1.0);

            // hue/saturation/value
            vec3 hsv = rgb2hsv(color);
            float hue = f_hue;
            if(hue > 1.0){
                hue = mod(hue, 1.0);
            }
            hsv[0] = mod((hsv[0] + hue), 1.0);
            hsv[1] += f_saturation;
            hsv[2] += f_value;
            hsv = clamp(hsv, 0.0, 1.0);
            color = hsv2rgb(hsv);

            if(f_invert > 0.0){
                color = vec3(1.0 - color[0], 1.0 - color[1], 1.0 - color[2]);
            }

            fragColor = vec4(color, alpha);

        }
    '''

    vertex_shader_simple_render_smooth = '''
        in vec3 position;
        in vec4 color;
        uniform mat4 perspective_matrix;
        uniform mat4 object_matrix;
        uniform float point_size;
        uniform float alpha_radius;
        uniform float global_alpha;
        out vec4 f_color;
        out float f_alpha_radius;
        void main()
        {
            gl_Position = perspective_matrix * object_matrix * vec4(position, 1.0f);
            gl_PointSize = point_size;
            f_color = vec4(color[0], color[1], color[2], global_alpha);
            f_alpha_radius = alpha_radius;
        }
    '''
    fragment_shader_simple_render_smooth = '''
        in vec4 f_color;
        in float f_alpha_radius;
        out vec4 fragColor;
        void main()
        {
            float r = 0.0f;
            float d = 0.0f;
            float a = 1.0f;
            vec2 cxy = 2.0f * gl_PointCoord - 1.0f;
            r = dot(cxy, cxy);
            d = fwidth(r);
            a = 1.0 - smoothstep(1.0 - (d / 2), 1.0 + (d / 2), r);
            //fragColor = f_color * a;
            fragColor = vec4(f_color.rgb, f_color.a * a);
        }
    '''

    vertex_shader_illumination_render_smooth = '''
        in vec3 position;
        in vec3 normal;
        in vec4 color;

        // uniform float show_illumination;
        uniform vec3 light_direction;
        uniform vec3 light_intensity;
        uniform vec3 shadow_direction;
        uniform vec3 shadow_intensity;
        // uniform float show_normals;

        uniform mat4 perspective_matrix;
        uniform mat4 object_matrix;
        uniform float point_size;
        uniform float alpha_radius;
        uniform float global_alpha;

        out vec4 f_color;
        out float f_alpha_radius;
        out vec3 f_normal;

        out vec3 f_light_direction;
        out vec3 f_light_intensity;
        out vec3 f_shadow_direction;
        out vec3 f_shadow_intensity;
        // out float f_show_normals;
        // out float f_show_illumination;

        void main()
        {
            gl_Position = perspective_matrix * object_matrix * vec4(position, 1.0f);
            gl_PointSize = point_size;
            f_normal = normal;
            // f_color = color;
            f_color = vec4(color[0], color[1], color[2], global_alpha);
            f_alpha_radius = alpha_radius;

            // f_light_direction = normalize(vec3(inverse(object_matrix) * vec4(light_direction, 1.0)));
            f_light_direction = light_direction;
            f_light_intensity = light_intensity;
            // f_shadow_direction = normalize(vec3(inverse(object_matrix) * vec4(shadow_direction, 1.0)));
            f_shadow_direction = shadow_direction;
            f_shadow_intensity = shadow_intensity;
            // f_show_normals = show_normals;
            // f_show_illumination = show_illumination;
        }
    '''
    fragment_shader_illumination_render_smooth = '''
        in vec4 f_color;
        in vec3 f_normal;
        in float f_alpha_radius;

        in vec3 f_light_direction;
        in vec3 f_light_intensity;
        in vec3 f_shadow_direction;
        in vec3 f_shadow_intensity;
        // in float f_show_normals;
        // in float f_show_illumination;

        out vec4 fragColor;

        void main()
        {
            // float r = 0.0f;
            // float a = 1.0f;
            // vec2 cxy = 2.0f * gl_PointCoord - 1.0f;
            // r = dot(cxy, cxy);
            // if(r > f_alpha_radius){
            //     discard;
            // }
            // // fragColor = f_color * a;
            //
            // vec4 col;
            //
            // // if(f_show_normals > 0.5){
            // //     col = vec4(f_normal, 1.0) * a;
            // // }else if(f_show_illumination > 0.5){
            //
            // // if(f_show_illumination > 0.5){
            // //     vec4 light = vec4(max(dot(f_light_direction, -f_normal), 0) * f_light_intensity, 1);
            // //     vec4 shadow = vec4(max(dot(f_shadow_direction, -f_normal), 0) * f_shadow_intensity, 1);
            // //     col = (f_color + light - shadow) * a;
            // // }else{
            // //     col = f_color * a;
            // // }
            //
            // vec4 light = vec4(max(dot(f_light_direction, -f_normal), 0) * f_light_intensity, 1);
            // vec4 shadow = vec4(max(dot(f_shadow_direction, -f_normal), 0) * f_shadow_intensity, 1);
            // col = (f_color + light - shadow) * a;
            //
            // fragColor = col;

            float r = 0.0f;
            float d = 0.0f;
            float a = 1.0f;
            vec2 cxy = 2.0f * gl_PointCoord - 1.0f;
            r = dot(cxy, cxy);
            d = fwidth(r);
            a = 1.0 - smoothstep(1.0 - (d / 2), 1.0 + (d / 2), r);
            //fragColor = vec4(f_color.rgb, f_color.a * a);

            vec4 col;
            vec4 light = vec4(max(dot(f_light_direction, -f_normal), 0) * f_light_intensity, 1);
            vec4 shadow = vec4(max(dot(f_shadow_direction, -f_normal), 0) * f_shadow_intensity, 1);
            //col = (f_color + light - shadow) * a;
            col = (f_color + light - shadow) * 1.0;
            //fragColor = col;
            fragColor = vec4(col.rgb, f_color.a * a);

        }
    '''

    vertex_shader_minimal = '''
        in vec3 position;
        in vec4 color;
        uniform mat4 perspective_matrix;
        uniform mat4 object_matrix;
        uniform float point_size;
        uniform float global_alpha;
        out vec3 f_color;
        out float f_alpha;
        void main()
        {
            gl_Position = perspective_matrix * object_matrix * vec4(position, 1.0f);
            gl_PointSize = point_size;
            f_color = color.rgb;
            f_alpha = global_alpha;
        }
    '''
    fragment_shader_minimal = '''
        in vec3 f_color;
        in float f_alpha;
        out vec4 fragColor;
        void main()
        {
            fragColor = vec4(f_color, f_alpha);
        }
    '''

    vertex_shader_minimal_variable_size = '''
        in vec3 position;
        in vec4 color;
        // in float size;
        in int size;
        uniform mat4 perspective_matrix;
        uniform mat4 object_matrix;
        uniform float global_alpha;
        out vec3 f_color;
        out float f_alpha;
        void main()
        {
            gl_Position = perspective_matrix * object_matrix * vec4(position, 1.0f);
            gl_PointSize = size;
            f_color = color.rgb;
            f_alpha = global_alpha;
        }
    '''
    fragment_shader_minimal_variable_size = '''
        in vec3 f_color;
        in float f_alpha;
        out vec4 fragColor;
        void main()
        {
            fragColor = vec4(f_color, f_alpha);
        }
    '''

    vertex_shader_minimal_variable_size_and_depth = '''
        in vec3 position;
        in vec4 color;
        in int size;
        uniform mat4 perspective_matrix;
        uniform mat4 object_matrix;
        uniform float global_alpha;

        uniform vec3 center;
        uniform float maxdist;

        out vec3 f_color;
        out float f_alpha;

        out float f_depth;

        void main()
        {
            gl_Position = perspective_matrix * object_matrix * vec4(position, 1.0f);
            gl_PointSize = size;
            f_color = color.rgb;
            f_alpha = global_alpha;

            vec4 pp = perspective_matrix * object_matrix * vec4(position, 1.0);
            vec4 op = perspective_matrix * object_matrix * vec4(center, 1.0);
            float d = op.z - pp.z;
            f_depth = ((d - (-maxdist)) / (maxdist - d)) / 2;
        }
    '''
    fragment_shader_minimal_variable_size_and_depth = '''
        in vec3 f_color;
        in float f_alpha;

        in float f_depth;
        uniform float brightness;
        uniform float contrast;
        uniform float blend;

        out vec4 fragColor;
        void main()
        {
            // fragColor = vec4(f_color, f_alpha);

            vec3 depth_color = vec3(f_depth, f_depth, f_depth);
            depth_color = (depth_color - 0.5) * contrast + 0.5 + brightness;
            // fragColor = vec4(depth_color, global_alpha) * a;

            depth_color = mix(depth_color, vec3(1.0, 1.0, 1.0), blend);

            fragColor = vec4(f_color * depth_color, f_alpha);

        }
    '''

    billboard_vertex = '''
        layout(location = 0) in vec3 position;
        layout(location = 1) in vec4 color;

        uniform mat4 object_matrix;
        uniform float alpha;

        out vec4 vcolor;
        out float valpha;

        void main()
        {
            gl_Position = object_matrix * vec4(position, 1.0);
            vcolor = color;
            valpha = alpha;
        }
    '''
    billboard_fragment = '''
        layout(location = 0) out vec4 frag_color;

        in vec4 fcolor;
        in float falpha;

        void main()
        {
            frag_color = vec4(fcolor.rgb, falpha);
        }
    '''
    billboard_geometry = '''
        layout (points) in;
        layout (triangle_strip, max_vertices = 4) out;

        in vec4 vcolor[];
        in float valpha[];

        uniform mat4 view_matrix;
        uniform mat4 window_matrix;

        uniform float size[];

        out vec4 fcolor;
        out float falpha;

        void main()
        {
            fcolor = vcolor[0];
            falpha = valpha[0];
            // value is diameter, i need radius
            float s = size[0] / 2;

            vec4 pos = view_matrix * gl_in[0].gl_Position;
            vec2 xyloc = vec2(-1 * s, -1 * s);
            gl_Position = window_matrix * (pos + vec4(xyloc, 0, 0));
            EmitVertex();

            xyloc = vec2(1 * s, -1 * s);
            gl_Position = window_matrix * (pos + vec4(xyloc, 0, 0));
            EmitVertex();

            xyloc = vec2(-1 * s, 1 * s);
            gl_Position = window_matrix * (pos + vec4(xyloc, 0, 0));
            EmitVertex();

            xyloc = vec2(1 * s, 1 * s);
            gl_Position = window_matrix * (pos + vec4(xyloc, 0, 0));
            EmitVertex();

            EndPrimitive();
        }
    '''

    billboard_geometry_disc = '''
        layout (points) in;
        // 3 * 16 = 48
        layout (triangle_strip, max_vertices = 48) out;

        in vec4 vcolor[];
        in float valpha[];

        uniform mat4 view_matrix;
        uniform mat4 window_matrix;

        uniform float size[];

        out vec4 fcolor;
        out float falpha;

        vec2 disc_coords(float radius, int step, int steps)
        {
            const float PI = 3.1415926535897932384626433832795;
            float angstep = 2 * PI / steps;
            float x = sin(step * angstep) * radius;
            float y = cos(step * angstep) * radius;
            return vec2(x, y);
        }

        void main()
        {
            fcolor = vcolor[0];
            falpha = valpha[0];
            float s = size[0];

            vec4 pos = view_matrix * gl_in[0].gl_Position;
            float r = s / 2;
            int steps = 16;

            for(int i = 0; i < steps; i++)
            {

                gl_Position = window_matrix * (pos);
                EmitVertex();

                vec2 xyloc = disc_coords(r, i, steps);
                gl_Position = window_matrix * (pos + vec4(xyloc, 0, 0));
                EmitVertex();

                xyloc = disc_coords(r, i + 1, steps);
                gl_Position = window_matrix * (pos + vec4(xyloc, 0, 0));
                EmitVertex();

                EndPrimitive();
            }

        }
    '''

    billboard_vertex_with_depth_and_size = '''
        layout(location = 0) in vec3 position;
        layout(location = 1) in vec4 color;
        layout(location = 2) in float sizef;

        uniform mat4 object_matrix;
        uniform mat4 perspective_matrix;

        uniform float alpha;
        uniform vec3 center;
        uniform float maxdist;

        out vec4 vcolor;
        out float valpha;
        out float vsizef;
        out float vdepth;

        void main()
        {
            gl_Position = object_matrix * vec4(position, 1.0);
            vcolor = color;
            valpha = alpha;
            vsizef = sizef;

            vec4 pp = perspective_matrix * object_matrix * vec4(position, 1.0);
            vec4 op = perspective_matrix * object_matrix * vec4(center, 1.0);
            float d = op.z - pp.z;
            vdepth = ((d - (-maxdist)) / (maxdist - d)) / 2;
        }
    '''
    billboard_fragment_with_depth_and_size = '''
        layout(location = 0) out vec4 frag_color;

        in vec4 fcolor;
        in float falpha;

        in float fdepth;
        uniform float brightness;
        uniform float contrast;
        uniform float blend;

        void main()
        {
            vec3 depth_color = vec3(fdepth, fdepth, fdepth);
            depth_color = (depth_color - 0.5) * contrast + 0.5 + brightness;
            depth_color = mix(depth_color, vec3(1.0, 1.0, 1.0), blend);
            frag_color = vec4(fcolor.rgb * depth_color, falpha);
        }
    '''
    billboard_geometry_with_depth_and_size = '''
        layout (points) in;
        layout (triangle_strip, max_vertices = 4) out;

        in vec4 vcolor[];
        in float valpha[];
        in float vsizef[];
        in float vdepth[];

        uniform mat4 view_matrix;
        uniform mat4 window_matrix;

        uniform float size[];

        out vec4 fcolor;
        out float falpha;
        out float fdepth;

        void main()
        {
            fcolor = vcolor[0];
            falpha = valpha[0];
            fdepth = vdepth[0];

            // value is diameter, i need radius, then multiply by individual point size
            float s = (size[0] / 2) * vsizef[0];

            vec4 pos = view_matrix * gl_in[0].gl_Position;
            vec2 xyloc = vec2(-1 * s, -1 * s);
            gl_Position = window_matrix * (pos + vec4(xyloc, 0, 0));
            EmitVertex();

            xyloc = vec2(1 * s, -1 * s);
            gl_Position = window_matrix * (pos + vec4(xyloc, 0, 0));
            EmitVertex();

            xyloc = vec2(-1 * s, 1 * s);
            gl_Position = window_matrix * (pos + vec4(xyloc, 0, 0));
            EmitVertex();

            xyloc = vec2(1 * s, 1 * s);
            gl_Position = window_matrix * (pos + vec4(xyloc, 0, 0));
            EmitVertex();

            EndPrimitive();
        }
    '''

    phong_vs = '''
        layout (location = 0) in vec3 position;
        layout (location = 1) in vec3 normal;
        layout (location = 2) in vec4 color;

        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        uniform float point_size;
        uniform float alpha_radius;

        out vec3 f_position;
        out vec3 f_normal;
        out vec4 f_color;
        out float f_alpha_radius;

        void main()
        {
            gl_Position = projection * view * model * vec4(position, 1.0);
            gl_PointSize = point_size;
            f_position = vec3(model * vec4(position, 1.0));
            f_normal = mat3(transpose(inverse(model))) * normal;
            f_color = color;
            f_alpha_radius = alpha_radius;
        }
    '''
    phong_fs = '''
        in vec3 f_position;
        in vec3 f_normal;
        in vec4 f_color;
        in float f_alpha_radius;

        uniform float alpha;
        uniform vec3 light_position;
        uniform vec3 light_color;
        uniform vec3 view_position;
        uniform float ambient_strength;
        uniform float specular_strength;
        uniform float specular_exponent;

        out vec4 frag_color;

        void main()
        {
            vec3 ambient = ambient_strength * light_color;

            vec3 nor = normalize(f_normal);
            vec3 light_direction = normalize(light_position - f_position);
            vec3 diffuse = max(dot(nor, light_direction), 0.0) * light_color;

            vec3 view_direction = normalize(view_position - f_position);
            vec3 reflection_direction = reflect(-light_direction, nor);
            float spec = pow(max(dot(view_direction, reflection_direction), 0.0), specular_exponent);
            vec3 specular = specular_strength * spec * light_color;

            vec3 col = (ambient + diffuse + specular) * f_color.rgb;

            float r = 0.0f;
            float a = 1.0f;
            vec2 cxy = 2.0f * gl_PointCoord - 1.0f;
            r = dot(cxy, cxy);
            if(r > f_alpha_radius){
                discard;
            }

            frag_color = vec4(col, alpha) * a;
        }
    '''

    billboard_vertex_with_no_depth_and_size = '''
        layout(location = 0) in vec3 position;
        layout(location = 1) in vec4 color;
        layout(location = 2) in float sizef;

        uniform mat4 object_matrix;

        uniform float alpha;

        out vec4 vcolor;
        out float valpha;
        out float vsizef;

        void main()
        {
            gl_Position = object_matrix * vec4(position, 1.0);
            vcolor = color;
            valpha = alpha;
            vsizef = sizef;
        }
    '''
    billboard_fragment_with_no_depth_and_size = '''
        layout(location = 0) out vec4 frag_color;

        in vec4 fcolor;
        in float falpha;

        void main()
        {
            frag_color = vec4(fcolor.rgb, falpha);
        }
    '''
    billboard_geometry_with_no_depth_and_size = '''
        layout (points) in;
        layout (triangle_strip, max_vertices = 4) out;

        in vec4 vcolor[];
        in float valpha[];
        in float vsizef[];

        uniform mat4 view_matrix;
        uniform mat4 window_matrix;

        uniform float size[];

        out vec4 fcolor;
        out float falpha;

        void main()
        {
            fcolor = vcolor[0];
            falpha = valpha[0];

            // value is diameter, i need radius, then multiply by individual point size
            float s = (size[0] / 2) * vsizef[0];

            vec4 pos = view_matrix * gl_in[0].gl_Position;
            vec2 xyloc = vec2(-1 * s, -1 * s);
            gl_Position = window_matrix * (pos + vec4(xyloc, 0, 0));
            EmitVertex();

            xyloc = vec2(1 * s, -1 * s);
            gl_Position = window_matrix * (pos + vec4(xyloc, 0, 0));
            EmitVertex();

            xyloc = vec2(-1 * s, 1 * s);
            gl_Position = window_matrix * (pos + vec4(xyloc, 0, 0));
            EmitVertex();

            xyloc = vec2(1 * s, 1 * s);
            gl_Position = window_matrix * (pos + vec4(xyloc, 0, 0));
            EmitVertex();

            EndPrimitive();
        }
    '''

    vertex_shader_simple_clip = '''
        in vec3 position;
        in vec4 color;
        uniform mat4 perspective_matrix;
        uniform mat4 object_matrix;
        uniform float point_size;
        uniform float alpha_radius;
        uniform float global_alpha;
        out vec4 f_color;
        out float f_alpha_radius;

        uniform vec4 clip_plane0;
        uniform vec4 clip_plane1;
        uniform vec4 clip_plane2;
        uniform vec4 clip_plane3;
        uniform vec4 clip_plane4;
        uniform vec4 clip_plane5;

        void main()
        {
            gl_Position = perspective_matrix * object_matrix * vec4(position, 1.0f);
            gl_PointSize = point_size;
            f_color = vec4(color[0], color[1], color[2], global_alpha);
            f_alpha_radius = alpha_radius;

            vec4 pos = vec4(position, 1.0f);
            gl_ClipDistance[0] = dot(clip_plane0, pos);
            gl_ClipDistance[1] = dot(clip_plane1, pos);
            gl_ClipDistance[2] = dot(clip_plane2, pos);
            gl_ClipDistance[3] = dot(clip_plane3, pos);
            gl_ClipDistance[4] = dot(clip_plane4, pos);
            gl_ClipDistance[5] = dot(clip_plane5, pos);
        }
    '''
    fragment_shader_simple_clip = '''
        in vec4 f_color;
        in float f_alpha_radius;
        out vec4 fragColor;
        void main()
        {
            float r = 0.0f;
            float a = 1.0f;
            vec2 cxy = 2.0f * gl_PointCoord - 1.0f;
            r = dot(cxy, cxy);
            if(r > f_alpha_radius){
                discard;
            }
            fragColor = f_color * a;
        }
    '''

    billboard_phong_vs = '''
        layout (location = 0) in vec3 position;
        layout (location = 1) in vec3 normal;
        layout (location = 2) in vec4 color;

        uniform mat4 model;
        out vec3 g_position;
        out vec3 g_normal;
        out vec4 g_color;

        void main()
        {
            gl_Position = model * vec4(position, 1.0);
            g_position = vec3(model * vec4(position, 1.0));
            g_normal = mat3(transpose(inverse(model))) * normal;
            g_color = color;
        }
    '''
    billboard_phong_circles_gs = '''
        layout (points) in;
        layout (triangle_strip, max_vertices = 48) out;

        in vec3 g_position[];
        in vec3 g_normal[];
        in vec4 g_color[];
        uniform mat4 view_matrix;
        uniform mat4 window_matrix;
        uniform float size[];
        out vec3 f_position;
        out vec3 f_normal;
        out vec4 f_color;

        vec2 disc_coords(float radius, int step, int steps)
        {
            const float PI = 3.1415926535897932384626433832795;
            float angstep = 2 * PI / steps;
            float x = sin(step * angstep) * radius;
            float y = cos(step * angstep) * radius;
            return vec2(x, y);
        }

        void main()
        {
            f_position = g_position[0];
            f_normal = g_normal[0];
            f_color = g_color[0];

            float s = size[0];
            vec4 pos = view_matrix * gl_in[0].gl_Position;
            float r = s / 2;
            // 3 * 16 = max_vertices 48
            int steps = 16;
            for (int i = 0; i < steps; i++)
            {
                gl_Position = window_matrix * (pos);
                EmitVertex();

                vec2 xyloc = disc_coords(r, i, steps);
                gl_Position = window_matrix * (pos + vec4(xyloc, 0, 0));
                EmitVertex();

                xyloc = disc_coords(r, i + 1, steps);
                gl_Position = window_matrix * (pos + vec4(xyloc, 0, 0));
                EmitVertex();

                EndPrimitive();
            }
        }
    '''
    billboard_phong_fast_gs = '''
        layout (points) in;
        layout (triangle_strip, max_vertices = 4) out;

        in vec3 g_position[];
        in vec3 g_normal[];
        in vec4 g_color[];
        uniform mat4 view_matrix;
        uniform mat4 window_matrix;
        uniform float size[];
        out vec3 f_position;
        out vec3 f_normal;
        out vec4 f_color;

        void main()
        {
            f_position = g_position[0];
            f_normal = g_normal[0];
            f_color = g_color[0];

            float s = size[0] / 2;

            vec4 pos = view_matrix * gl_in[0].gl_Position;
            vec2 xyloc = vec2(-1 * s, -1 * s);
            gl_Position = window_matrix * (pos + vec4(xyloc, 0, 0));
            EmitVertex();

            xyloc = vec2(1 * s, -1 * s);
            gl_Position = window_matrix * (pos + vec4(xyloc, 0, 0));
            EmitVertex();

            xyloc = vec2(-1 * s, 1 * s);
            gl_Position = window_matrix * (pos + vec4(xyloc, 0, 0));
            EmitVertex();

            xyloc = vec2(1 * s, 1 * s);
            gl_Position = window_matrix * (pos + vec4(xyloc, 0, 0));
            EmitVertex();

            EndPrimitive();
        }
    '''
    billboard_phong_fs = '''
        layout (location = 0) out vec4 frag_color;

        in vec3 f_position;
        in vec3 f_normal;
        in vec4 f_color;
        uniform float alpha;
        uniform vec3 light_position;
        uniform vec3 light_color;
        uniform vec3 view_position;
        uniform float ambient_strength;
        uniform float specular_strength;
        uniform float specular_exponent;

        void main()
        {
            vec3 ambient = ambient_strength * light_color;

            vec3 nor = normalize(f_normal);
            vec3 light_direction = normalize(light_position - f_position);
            vec3 diffuse = max(dot(nor, light_direction), 0.0) * light_color;

            vec3 view_direction = normalize(view_position - f_position);
            vec3 reflection_direction = reflect(-light_direction, nor);
            float spec = pow(max(dot(view_direction, reflection_direction), 0.0), specular_exponent);
            vec3 specular = specular_strength * spec * light_color;

            vec3 col = (ambient + diffuse + specular) * f_color.rgb;
            frag_color = vec4(col, alpha);
        }
    '''

    vertex_shader_simple_skip_point_vertices = '''
        in vec3 position;
        in vec4 color;
        in int index;
        uniform mat4 perspective_matrix;
        uniform mat4 object_matrix;
        uniform float point_size;
        uniform float alpha_radius;
        uniform float global_alpha;
        uniform float skip_index;
        out vec4 f_color;
        out float f_alpha_radius;
        void main()
        {
            gl_Position = perspective_matrix * object_matrix * vec4(position, 1.0f);

            if(skip_index <= index){
                gl_Position = vec4(2.0, 0.0, 0.0, 1.0);
            }

            gl_PointSize = point_size;
            f_color = vec4(color[0], color[1], color[2], global_alpha);
            f_alpha_radius = alpha_radius;
        }
    '''
    fragment_shader_simple_skip_point_vertices = '''
        in vec4 f_color;
        in float f_alpha_radius;
        out vec4 fragColor;
        void main()
        {
            float r = 0.0f;
            float a = 1.0f;
            vec2 cxy = 2.0f * gl_PointCoord - 1.0f;
            r = dot(cxy, cxy);
            if(r > f_alpha_radius){
                discard;
            }
            fragColor = f_color * a;
        }
    '''

class BinPlyPointCloudWriter():
    """Save binary ply file from data numpy array

    Args:
        path: path to ply file
        points: strucured array of points as (x, y, z, nx, ny, nz, red, green, blue) (normals and colors are optional)

    Attributes:
        path (str): real path to ply file

    """

    _types = {'c': 'char', 'B': 'uchar', 'h': 'short', 'H': 'ushort', 'i': 'int', 'I': 'uint', 'f': 'float',
              'd': 'double', }
    _byte_order = {'little': 'binary_little_endian', 'big': 'binary_big_endian', }
    _comment = "created with Point Cloud Visualizer"

    def __init__(self, path, points, ):
        log("{}:".format(self.__class__.__name__), 0)
        self.path = os.path.realpath(path)

        # write
        log("will write to: {}".format(self.path), 1)
        # write to temp file first
        n = os.path.splitext(os.path.split(self.path)[1])[0]
        t = "{}.temp.ply".format(n)
        p = os.path.join(os.path.dirname(self.path), t)

        l = len(points)

        with open(p, 'wb') as f:
            # write header
            log("writing header..", 2)
            dt = points.dtype
            h = "ply\n"
            # x should be a float of some kind, therefore we can get endianess
            bo = dt['x'].byteorder
            if (bo != '='):
                # not native byteorder
                if (bo == '>'):
                    h += "format {} 1.0\n".format(self._byte_order['big'])
                else:
                    h += "format {} 1.0\n".format(self._byte_order['little'])
            else:
                # byteorder was native, use what sys.byteorder says..
                h += "format {} 1.0\n".format(self._byte_order[sys.byteorder])
            h += "element vertex {}\n".format(l)
            # construct header from data names/types in points array
            for n in dt.names:
                t = self._types[dt[n].char]
                h += "property {} {}\n".format(t, n)
            h += "comment {}\n".format(self._comment)
            h += "end_header\n"
            f.write(h.encode('ascii'))

            # write data
            log("writing data.. ({} points)".format(l), 2)
            f.write(points.tobytes())

        # remove original file (if needed) and rename temp
        if (os.path.exists(self.path)):
            os.remove(self.path)
        shutil.move(p, self.path)

        log("done.", 1)
        print("┌────────────────────┐")
        print("│ ply file is saved. │")
        print("└────────────────────┘")

class PCVManager():
    cache = {}
    handle = None
    initialized = False

    @classmethod
    def load_ply_to_cache(cls, operator, context):
        PCVManager.init()
        obj = data_Storage.point_Cloud_Axis
        pcv = obj.point_cloud_visualizer
        filepath = None

        __t = time.time()
        _t = time.time()

        # FIXME ply loading might not work with all ply files, for example, file spec seems does not forbid having two or more blocks of vertices with different props, currently i load only first block of vertices. maybe construct some messed up ply and test how for example meshlab behaves
        points = []
        try:
            points = data_Storage.points
            data_Storage.points = []
        except Exception as e:
            if (operator is not None):
                print("ERROR: " + str(e))
            else:
                raise e
        if (len(points) == 0):
            print("No vertices loaded from file at {}".format(filepath))
            return False

        _d = datetime.timedelta(seconds=time.time() - _t)
        # print("completed in {}.".format(_d))

        _t = time.time()

        np.random.shuffle(points)

        _d = datetime.timedelta(seconds=time.time() - _t)
        # print("completed in {}.".format(_d))

        # print('process data..')
        _t = time.time()

        if (not set(('x', 'y', 'z')).issubset(points.dtype.names)):
            # this is very unlikely..
            operator.report({'ERROR'}, "Loaded data seems to miss vertex locations.")
            return False

        # FIXME checking for normals/colors in points is kinda scattered all over.. chceck should be upon loading / setting from external script
        normals = True
        if (not set(('nx', 'ny', 'nz')).issubset(points.dtype.names)):
            normals = False
        pcv.has_normals = normals
        if (not pcv.has_normals):
            pcv.illumination = False
        vcols = True
        if (not set(('red', 'green', 'blue')).issubset(points.dtype.names)):
            vcols = False
        pcv.has_vcols = vcols

        vs = np.column_stack((points['x'], points['y'], points['z'],))
        if (vs.dtype != np.float32):
            vs = vs.astype(np.float32)

        if (normals):
            ns = np.column_stack((points['nx'], points['ny'], points['nz'],))
            if (ns.dtype != np.float32):
                ns = ns.astype(np.float32)

        else:
            n = len(points)
            ns = np.column_stack((np.full(n, 0.0, dtype=np.float32, ),
                                  np.full(n, 0.0, dtype=np.float32, ),
                                  np.full(n, 1.0, dtype=np.float32, ),))

        if (vcols):
            if (points['red'].dtype == 'uint16'):
                r8 = (points['red'] / 256).astype('uint8')
                g8 = (points['green'] / 256).astype('uint8')
                b8 = (points['blue'] / 256).astype('uint8')
                cs = np.column_stack((r8 / 255, g8 / 255, b8 / 255, np.ones(len(points), dtype=float, ),))
                cs = cs.astype(np.float32)
            else:
                # 'uint8'
                cs = np.column_stack((points['red'] / 255, points['green'] / 255, points['blue'] / 255,
                                      np.ones(len(points), dtype=float, ),))
                cs = cs.astype(np.float32)
                # ======================================================================= #
        else:
            n = len(points)
            col = (0.65, 0.65, 0.65,)
            col = tuple([c ** (1 / 2.2) for c in col]) + (1.0,)
            cs = np.column_stack((np.full(n, col[0], dtype=np.float32, ),
                                  np.full(n, col[1], dtype=np.float32, ),
                                  np.full(n, col[2], dtype=np.float32, ),
                                  np.ones(n, dtype=np.float32, ),))

        u = str(uuid.uuid1())
        o = data_Storage.point_Cloud_Axis

        pcv.uuid = u

        d = PCVManager.new()
        d['filepath'] = filepath

        d['points'] = points

        d['uuid'] = u
        d['stats'] = len(vs)
        d['vertices'] = vs
        d['colors'] = cs
        d['normals'] = ns

        d['length'] = len(vs)
        dp = pcv.display_percent
        l = int((len(vs) / 100) * dp)
        if (dp >= 99):
            l = len(vs)
        d['display_length'] = l
        d['current_display_length'] = l

        ienabled = pcv.illumination
        d['illumination'] = ienabled
        if (ienabled):
            shader = GPUShader(PCVShaders.vertex_shader_illumination, PCVShaders.fragment_shader_illumination)
            batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], "normal": ns[:l], })
        else:
            shader = GPUShader(PCVShaders.vertex_shader_simple, PCVShaders.fragment_shader_simple)
            batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], })

        d['shader'] = shader
        d['batch'] = batch
        d['ready'] = True
        d['object'] = o
        d['name'] = o.name

        PCVManager.add(d)

        _d = datetime.timedelta(seconds=time.time() - _t)
        # print("completed in {}.".format(_d))

        # print("-" * 50)
        __d = datetime.timedelta(seconds=time.time() - __t)
        print("load and process completed in {}.".format(__d))
        # print("-" * 50)

        # with new file browser in 2.81, screen is not redrawn, so i have to do it manually..
        cls._redraw()

        return True

    @classmethod
    def render(cls, uuid, ):
        bgl.glEnable(bgl.GL_PROGRAM_POINT_SIZE)
        bgl.glEnable(bgl.GL_DEPTH_TEST)
        bgl.glEnable(bgl.GL_BLEND)

        # TODO: replace all 'batch_for_shader' (2.80/scripts/modules/gpu_extras/batch.py) calls with something custom made and keep buffer cached. faster shader switching, less memory used, etc..

        ci = PCVManager.cache[uuid]

        shader = ci['shader']
        batch = ci['batch']

        if (ci['current_display_length'] != ci['display_length']):
            l = ci['display_length']
            ci['current_display_length'] = l
            vs = ci['vertices']
            cs = ci['colors']
            ns = ci['normals']
            if (ci['illumination']):
                batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], "normal": ns[:l], })
            else:
                batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], })
            ci['batch'] = batch

        o = ci['object']
        try:
            pcv = o.point_cloud_visualizer
        except ReferenceError:
            # FIXME undo still doesn't work in some cases, from what i've seen, only when i am undoing operations on parent object, especially when you undo/redo e.g. transforms around load/draw operators, filepath property gets reset and the whole thing is drawn, but ui looks like loding never happened, i've added a quick fix storing path in cache, but it all depends on object name and this is bad.
            # NOTE parent object reference check should be before drawing, not in the middle, it's not that bad, it's pretty early, but it's still messy, this will require rewrite of handler and render functions in manager.. so don't touch until broken
            log("PCVManager.render: ReferenceError (possibly after undo/redo?)")
            # blender on undo/redo swaps whole scene to different one stored in memory and therefore stored object references are no longer valid
            # so find object with the same name, not the best solution, but lets see how it goes..
            o = bpy.data.objects[ci['name']]
            # update stored reference
            ci['object'] = o
            pcv = o.point_cloud_visualizer
            # push back correct uuid, since undo changed it, why? WHY? why do i even bother?
            pcv.uuid = uuid
            # push back filepath, it might get lost during undo/redo
            pcv.filepath = ci['filepath']

        if (not o.visible_get()):
            # if parent object is not visible, skip drawing
            # this should checked earlier, but until now i can't be sure i have correct object reference

            # NOTE: use bpy.context.view_layer.objects.active instead of context.active_object and add option to not hide cloud when parent object is hidden? seems like this is set when object is clicked in outliner even when hidden, at least properties buttons are changed.. if i unhide and delete the object, props buttons are not drawn, if i click on another already hidden object, correct buttons are back, so i need to check if there is something active.. also this would require rewriting all panels polls, now they check for context.active_object and if None, which is when object is hidden, panel is not drawn..

            bgl.glDisable(bgl.GL_PROGRAM_POINT_SIZE)
            bgl.glDisable(bgl.GL_DEPTH_TEST)
            bgl.glDisable(bgl.GL_BLEND)
            return

        if (ci['illumination'] != pcv.illumination):
            vs = ci['vertices']
            cs = ci['colors']
            ns = ci['normals']
            l = ci['current_display_length']
            if (pcv.illumination):
                shader = GPUShader(PCVShaders.vertex_shader_illumination, PCVShaders.fragment_shader_illumination)
                batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], "normal": ns[:l], })
                ci['illumination'] = True
            else:
                shader = GPUShader(PCVShaders.vertex_shader_simple, PCVShaders.fragment_shader_simple)
                batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], })
                ci['illumination'] = False
            ci['shader'] = shader
            ci['batch'] = batch

        shader.bind()
        pm = bpy.context.region_data.perspective_matrix
        shader.uniform_float("perspective_matrix", pm)
        shader.uniform_float("object_matrix", o.matrix_world)
        shader.uniform_float("point_size", pcv.point_size)
        shader.uniform_float("alpha_radius", pcv.alpha_radius)
        shader.uniform_float("global_alpha", pcv.global_alpha)

        if (pcv.illumination and pcv.has_normals and ci['illumination']):
            cm = Matrix(
                ((-1.0, 0.0, 0.0, 0.0,), (0.0, -0.0, 1.0, 0.0,), (0.0, -1.0, -0.0, 0.0,), (0.0, 0.0, 0.0, 1.0,),))
            _, obrot, _ = o.matrix_world.decompose()
            mr = obrot.to_matrix().to_4x4()
            mr.invert()
            direction = cm @ pcv.light_direction
            direction = mr @ direction
            shader.uniform_float("light_direction", direction)

            # def get_space3dview():
            #     for a in bpy.context.screen.areas:
            #         if(a.type == "VIEW_3D"):
            #             return a.spaces[0]
            #     return None
            #
            # s3dv = get_space3dview()
            # region3d = s3dv.region_3d
            # eye = region3d.view_matrix[2][:3]
            #
            # # shader.uniform_float("light_direction", Vector(eye) * -1)
            # shader.uniform_float("light_direction", Vector(eye))

            inverted_direction = direction.copy()
            inverted_direction.negate()

            c = pcv.light_intensity
            shader.uniform_float("light_intensity", (c, c, c,))
            shader.uniform_float("shadow_direction", inverted_direction)
            c = pcv.shadow_intensity
            shader.uniform_float("shadow_intensity", (c, c, c,))
            # shader.uniform_float("show_normals", float(pcv.show_normals))
            # shader.uniform_float("show_illumination", float(pcv.illumination))
        else:
            pass

        if (not pcv.override_default_shader):
            # NOTE: just don't draw default shader, quick and easy solution, other shader will be drawn instead, would better to not create it..
            batch.draw(shader)

            # # remove extra if present, will be recreated if needed and if left stored it might cause problems
            # if('extra' in ci.keys()):
            #     del ci['extra']

        if (pcv.vertex_normals and pcv.has_normals):
            def make(ci):
                l = ci['current_display_length']
                vs = ci['vertices'][:l]
                ns = ci['normals'][:l]

                shader = GPUShader(PCVShaders.normals_vertex_shader, PCVShaders.normals_fragment_shader,
                                   geocode=PCVShaders.normals_geometry_shader, )
                batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "normal": ns[:l], }, )

                d = {'shader': shader,
                     'batch': batch,
                     'position': vs,
                     'normal': ns,
                     'current_display_length': l, }
                ci['vertex_normals'] = d

                return shader, batch

            if ("vertex_normals" not in ci.keys()):
                shader, batch = make(ci)
            else:
                d = ci['vertex_normals']
                shader = d['shader']
                batch = d['batch']
                ok = True
                if (ci['current_display_length'] != d['current_display_length']):
                    ok = False
                if (not ok):
                    shader, batch = make(ci)

            shader.bind()
            pm = bpy.context.region_data.perspective_matrix
            shader.uniform_float("perspective_matrix", pm)
            shader.uniform_float("object_matrix", o.matrix_world)
            col = bpy.context.preferences.addons[__name__].preferences.normal_color[:]
            col = tuple([c ** (1 / 2.2) for c in col]) + (pcv.vertex_normals_alpha,)
            shader.uniform_float("color", col, )
            shader.uniform_float("length", pcv.vertex_normals_size, )
            shader.uniform_float("global_alpha", pcv.global_alpha)
            batch.draw(shader)

        if (pcv.dev_depth_enabled):

            # if(debug_mode()):
            #     import cProfile
            #     import pstats
            #     import io
            #     pr = cProfile.Profile()
            #     pr.enable()

            vs = ci['vertices']
            ns = ci['normals']
            l = ci['current_display_length']

            use_stored = False
            if ('extra' in ci.keys()):
                t = 'DEPTH'
                for k, v in ci['extra'].items():
                    if (k == t):
                        if (v['length'] == l):
                            if (v['illumination'] == pcv.illumination and v[
                                'false_colors'] == pcv.dev_depth_false_colors):
                                use_stored = True
                                batch = v['batch']
                                shader = v['shader']
                                break

            if (not use_stored):
                if (pcv.illumination):
                    shader = GPUShader(PCVShaders.depth_vertex_shader_illumination,
                                       PCVShaders.depth_fragment_shader_illumination, )
                    batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "normal": ns[:l], })
                elif (pcv.dev_depth_false_colors):
                    shader = GPUShader(PCVShaders.depth_vertex_shader_false_colors,
                                       PCVShaders.depth_fragment_shader_false_colors, )
                    batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], })
                else:
                    shader = GPUShader(PCVShaders.depth_vertex_shader_simple, PCVShaders.depth_fragment_shader_simple, )
                    batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], })

                if ('extra' not in ci.keys()):
                    ci['extra'] = {}
                d = {'shader': shader,
                     'batch': batch,
                     'illumination': pcv.illumination,
                     'false_colors': pcv.dev_depth_false_colors,
                     'length': l, }
                ci['extra']['DEPTH'] = d

            shader.bind()
            pm = bpy.context.region_data.perspective_matrix
            shader.uniform_float("perspective_matrix", pm)
            shader.uniform_float("object_matrix", o.matrix_world)

            # NOTE: precalculating and storing following should speed up things a bit, but then it won't reflect edits..
            # TODO: calculating center of all points is not quite correct, visually it works, but (as i've seen in bounding box shader) it's not working when distribution of points is uneven, so have a check if it might be a bit better..
            cx = np.sum(vs[:, 0]) / len(vs)
            cy = np.sum(vs[:, 1]) / len(vs)
            cz = np.sum(vs[:, 2]) / len(vs)
            _, _, s = o.matrix_world.decompose()
            l = s.length
            maxd = abs(np.max(vs))
            mind = abs(np.min(vs))
            maxdist = maxd
            if (mind > maxd):
                maxdist = mind
            shader.uniform_float("maxdist", float(maxdist) * l)
            shader.uniform_float("center", (cx, cy, cz,))

            shader.uniform_float("brightness", pcv.dev_depth_brightness)
            shader.uniform_float("contrast", pcv.dev_depth_contrast)

            shader.uniform_float("point_size", pcv.point_size)
            shader.uniform_float("alpha_radius", pcv.alpha_radius)
            shader.uniform_float("global_alpha", pcv.global_alpha)

            if (pcv.illumination):
                cm = Matrix(
                    ((-1.0, 0.0, 0.0, 0.0,), (0.0, -0.0, 1.0, 0.0,), (0.0, -1.0, -0.0, 0.0,), (0.0, 0.0, 0.0, 1.0,),))
                _, obrot, _ = o.matrix_world.decompose()
                mr = obrot.to_matrix().to_4x4()
                mr.invert()
                direction = cm @ pcv.light_direction
                direction = mr @ direction
                shader.uniform_float("light_direction", direction)
                inverted_direction = direction.copy()
                inverted_direction.negate()
                c = pcv.light_intensity
                shader.uniform_float("light_intensity", (c, c, c,))
                shader.uniform_float("shadow_direction", inverted_direction)
                c = pcv.shadow_intensity
                shader.uniform_float("shadow_intensity", (c, c, c,))
                if (pcv.dev_depth_false_colors):
                    shader.uniform_float("color_a", pcv.dev_depth_color_a)
                    shader.uniform_float("color_b", pcv.dev_depth_color_b)
                else:
                    shader.uniform_float("color_a", (1.0, 1.0, 1.0))
                    shader.uniform_float("color_b", (0.0, 0.0, 0.0))
            else:
                if (pcv.dev_depth_false_colors):
                    shader.uniform_float("color_a", pcv.dev_depth_color_a)
                    shader.uniform_float("color_b", pcv.dev_depth_color_b)

            batch.draw(shader)

            # if(debug_mode()):
            #     pr.disable()
            #     s = io.StringIO()
            #     sortby = 'cumulative'
            #     ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            #     ps.print_stats()
            #     print(s.getvalue())

        if (pcv.dev_normal_colors_enabled):

            vs = ci['vertices']
            ns = ci['normals']
            l = ci['current_display_length']

            use_stored = False
            if ('extra' in ci.keys()):
                t = 'NORMAL'
                for k, v in ci['extra'].items():
                    if (k == t):
                        if (v['length'] == l):
                            use_stored = True
                            batch = v['batch']
                            shader = v['shader']
                            break

            if (not use_stored):
                shader = GPUShader(PCVShaders.normal_colors_vertex_shader, PCVShaders.normal_colors_fragment_shader, )
                batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "normal": ns[:l], })

                if ('extra' not in ci.keys()):
                    ci['extra'] = {}
                d = {'shader': shader,
                     'batch': batch,
                     'length': l, }
                ci['extra']['NORMAL'] = d

            # shader = GPUShader(PCVShaders.normal_colors_vertex_shader, PCVShaders.normal_colors_fragment_shader, )
            # batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "normal": ns[:l], })
            shader.bind()
            pm = bpy.context.region_data.perspective_matrix
            shader.uniform_float("perspective_matrix", pm)
            shader.uniform_float("object_matrix", o.matrix_world)
            shader.uniform_float("point_size", pcv.point_size)
            shader.uniform_float("alpha_radius", pcv.alpha_radius)
            shader.uniform_float("global_alpha", pcv.global_alpha)
            batch.draw(shader)

        if (pcv.dev_position_colors_enabled):

            vs = ci['vertices']
            l = ci['current_display_length']

            use_stored = False
            if ('extra' in ci.keys()):
                t = 'POSITION'
                for k, v in ci['extra'].items():
                    if (k == t):
                        if (v['length'] == l):
                            use_stored = True
                            batch = v['batch']
                            shader = v['shader']
                            break

            if (not use_stored):
                shader = GPUShader(PCVShaders.position_colors_vertex_shader,
                                   PCVShaders.position_colors_fragment_shader, )
                batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], })

                if ('extra' not in ci.keys()):
                    ci['extra'] = {}
                d = {'shader': shader,
                     'batch': batch,
                     'length': l, }
                ci['extra']['POSITION'] = d

            # shader = GPUShader(PCVShaders.normal_colors_vertex_shader, PCVShaders.normal_colors_fragment_shader, )
            # batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "normal": ns[:l], })
            shader.bind()
            pm = bpy.context.region_data.perspective_matrix
            shader.uniform_float("perspective_matrix", pm)
            shader.uniform_float("object_matrix", o.matrix_world)
            shader.uniform_float("point_size", pcv.point_size)
            shader.uniform_float("alpha_radius", pcv.alpha_radius)
            shader.uniform_float("global_alpha", pcv.global_alpha)
            batch.draw(shader)

        if (pcv.dev_selection_shader_display):
            vs = ci['vertices']
            l = ci['current_display_length']
            shader = GPUShader(PCVShaders.selection_vertex_shader, PCVShaders.selection_fragment_shader, )
            batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], })
            shader.bind()
            pm = bpy.context.region_data.perspective_matrix
            shader.uniform_float("perspective_matrix", pm)
            shader.uniform_float("object_matrix", o.matrix_world)
            shader.uniform_float("color", pcv.dev_selection_shader_color)
            shader.uniform_float("point_size", pcv.point_size)
            shader.uniform_float("alpha_radius", pcv.alpha_radius)
            bgl.glClear(bgl.GL_DEPTH_BUFFER_BIT)
            batch.draw(shader)

        if (pcv.color_adjustment_shader_enabled):
            vs = ci['vertices']
            cs = ci['colors']
            l = ci['current_display_length']

            use_stored = False
            if ('extra' in ci.keys()):
                t = 'COLOR_ADJUSTMENT'
                for k, v in ci['extra'].items():
                    if (k == t):
                        if (v['length'] == l):
                            use_stored = True
                            batch = v['batch']
                            shader = v['shader']
                            break

            if (not use_stored):
                shader = GPUShader(PCVShaders.vertex_shader_color_adjustment,
                                   PCVShaders.fragment_shader_color_adjustment, )
                batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], })

                if ('extra' not in ci.keys()):
                    ci['extra'] = {}
                d = {'shader': shader,
                     'batch': batch,
                     'length': l, }
                ci['extra']['COLOR_ADJUSTMENT'] = d

            shader.bind()
            pm = bpy.context.region_data.perspective_matrix
            shader.uniform_float("perspective_matrix", pm)
            shader.uniform_float("object_matrix", o.matrix_world)
            shader.uniform_float("point_size", pcv.point_size)
            shader.uniform_float("alpha_radius", pcv.alpha_radius)
            shader.uniform_float("global_alpha", pcv.global_alpha)

            shader.uniform_float("exposure", pcv.color_adjustment_shader_exposure)
            shader.uniform_float("gamma", pcv.color_adjustment_shader_gamma)
            shader.uniform_float("brightness", pcv.color_adjustment_shader_brightness)
            shader.uniform_float("contrast", pcv.color_adjustment_shader_contrast)
            shader.uniform_float("hue", pcv.color_adjustment_shader_hue)
            shader.uniform_float("saturation", pcv.color_adjustment_shader_saturation)
            shader.uniform_float("value", pcv.color_adjustment_shader_value)
            shader.uniform_float("invert", pcv.color_adjustment_shader_invert)

            batch.draw(shader)

        # dev
        if (pcv.dev_bbox_enabled):
            vs = ci['vertices']

            use_stored = False
            if ('extra' in ci.keys()):
                t = 'BOUNDING_BOX'
                for k, v in ci['extra'].items():
                    if (k == t):
                        use_stored = True
                        batch = v['batch']
                        shader = v['shader']
                        break

            if (not use_stored):
                position = (0.0, 0.0, 0.0)
                shader = GPUShader(PCVShaders.bbox_vertex_shader, PCVShaders.bbox_fragment_shader,
                                   geocode=PCVShaders.bbox_geometry_shader, )
                # batch = batch_for_shader(shader, 'POINTS', {"position": [(0.0, 0.0, 0.0,)], }, )
                try:
                    batch = batch_for_shader(shader, 'POINTS', {"pos": position})
                except:
                    print("done\n")

                if ('extra' not in ci.keys()):
                    ci['extra'] = {}
                d = {'shader': shader,
                     'batch': batch, }
                ci['extra']['BOUNDING_BOX'] = d

            shader.bind()
            pm = bpy.context.region_data.perspective_matrix
            shader.uniform_float("perspective_matrix", pm)
            shader.uniform_float("object_matrix", o.matrix_world)
            # col = bpy.context.preferences.addons[__name__].preferences.normal_color[:]
            # col = tuple([c ** (1 / 2.2) for c in col]) + (pcv.vertex_normals_alpha, )

            # col = pcv.dev_bbox_color
            # col = tuple([c ** (1 / 2.2) for c in col]) + (1.0, )
            col = tuple(pcv.dev_bbox_color) + (pcv.dev_bbox_alpha,)

            shader.uniform_float("color", col, )
            # shader.uniform_float("length", pcv.vertex_normals_size, )
            shader.uniform_float("global_alpha", pcv.global_alpha)

            # # cx = np.sum(vs[:, 0]) / len(vs)
            # # cy = np.sum(vs[:, 1]) / len(vs)
            # # cz = np.sum(vs[:, 2]) / len(vs)
            # cx = np.median(vs[:, 0])
            # cy = np.median(vs[:, 1])
            # cz = np.median(vs[:, 2])
            # center = [cx, cy, cz]
            # # center = [0.0, 0.0, 0.0]
            # # print(center)
            # shader.uniform_float("center", center)

            # TODO: store values somewhere, might be slow if calculated every frame

            minx = np.min(vs[:, 0])
            miny = np.min(vs[:, 1])
            minz = np.min(vs[:, 2])
            maxx = np.max(vs[:, 0])
            maxy = np.max(vs[:, 1])
            maxz = np.max(vs[:, 2])

            def calc(mini, maxi):
                if (mini <= 0.0 and maxi <= 0.0):
                    return abs(mini) - abs(maxi)
                elif (mini <= 0.0 and maxi >= 0.0):
                    return abs(mini) + maxi
                else:
                    return maxi - mini

            dimensions = [calc(minx, maxx), calc(miny, maxy), calc(minz, maxz)]
            shader.uniform_float("dimensions", dimensions)

            center = [(minx + maxx) / 2, (miny + maxy) / 2, (minz + maxz) / 2]
            shader.uniform_float("center", center)

            mindim = abs(min(dimensions)) / 2 * pcv.dev_bbox_size
            shader.uniform_float("length", mindim)

            batch.draw(shader)

        # dev
        if (pcv.dev_minimal_shader_enabled):
            vs = ci['vertices']
            cs = ci['colors']
            l = ci['current_display_length']

            use_stored = False
            if ('extra' in ci.keys()):
                t = 'MINIMAL'
                for k, v in ci['extra'].items():
                    if (k == t):
                        if (v['length'] == l):
                            use_stored = True
                            batch = v['batch']
                            shader = v['shader']
                            break

            if (not use_stored):
                shader = GPUShader(PCVShaders.vertex_shader_minimal, PCVShaders.fragment_shader_minimal, )
                batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], })

                if ('extra' not in ci.keys()):
                    ci['extra'] = {}
                d = {'shader': shader,
                     'batch': batch,
                     'length': l, }
                ci['extra']['MINIMAL'] = d

            shader.bind()
            pm = bpy.context.region_data.perspective_matrix
            shader.uniform_float("perspective_matrix", pm)
            shader.uniform_float("object_matrix", o.matrix_world)
            shader.uniform_float("point_size", pcv.point_size)
            shader.uniform_float("global_alpha", pcv.global_alpha)
            batch.draw(shader)

        # dev
        if (pcv.dev_minimal_shader_variable_size_enabled):
            vs = ci['vertices']
            cs = ci['colors']
            l = ci['current_display_length']

            use_stored = False
            if ('extra' in ci.keys()):
                t = 'MINIMAL_VARIABLE_SIZE'
                for k, v in ci['extra'].items():
                    if (k == t):
                        if (v['length'] == l):
                            use_stored = True
                            batch = v['batch']
                            shader = v['shader']
                            sizes = v['sizes']
                            break

            if (not use_stored):
                # # generate something to test it, later implement how to set it
                # sizes = np.random.randint(low=1, high=10, size=len(vs), )

                if ('extra' in ci.keys()):
                    if ('MINIMAL_VARIABLE_SIZE' in ci['extra'].keys()):
                        sizes = ci['extra']['MINIMAL_VARIABLE_SIZE']['sizes']
                    else:
                        sizes = np.random.randint(low=1, high=10, size=len(vs), )
                else:
                    sizes = np.random.randint(low=1, high=10, size=len(vs), )

                if ('extra' in ci.keys()):
                    for k, v in ci['extra'].items():
                        if (k in ('MINIMAL_VARIABLE_SIZE', 'MINIMAL_VARIABLE_SIZE_AND_DEPTH', 'RICH_BILLBOARD',
                                  'RICH_BILLBOARD_NO_DEPTH',)):
                            # FIXME: it was recently switched, try to recover already generated data, both arrays are the same, so thay are in memory just once, no problem here, the problem is with storing reference to it, this should be fixed with something, for example, shader and batch should be generated on one spot, etc.. but have to be done together with all new PCVManager.render and unified shader management. this is just mediocre workaround..
                            sizes = ci['extra'][k]['sizes']
                            sizesf = ci['extra'][k]['sizesf']
                            break

                shader = GPUShader(PCVShaders.vertex_shader_minimal_variable_size,
                                   PCVShaders.fragment_shader_minimal_variable_size, )
                batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], "size": sizes[:l], })
                # batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], })

                if ('extra' not in ci.keys()):
                    ci['extra'] = {}

                d = {'shader': shader,
                     'batch': batch,
                     'sizes': sizes,
                     'length': l, }
                ci['extra']['MINIMAL_VARIABLE_SIZE'] = d

            shader.bind()
            pm = bpy.context.region_data.perspective_matrix
            shader.uniform_float("perspective_matrix", pm)
            shader.uniform_float("object_matrix", o.matrix_world)
            # shader.uniform_float("point_size", pcv.point_size)
            shader.uniform_float("global_alpha", pcv.global_alpha)
            batch.draw(shader)

        # dev
        if (pcv.dev_minimal_shader_variable_size_and_depth_enabled):
            vs = ci['vertices']
            cs = ci['colors']
            l = ci['current_display_length']

            use_stored = False
            if ('extra' in ci.keys()):
                t = 'MINIMAL_VARIABLE_SIZE_AND_DEPTH'
                for k, v in ci['extra'].items():
                    if (k == t):
                        if (v['length'] == l):
                            use_stored = True
                            batch = v['batch']
                            shader = v['shader']
                            sizes = v['sizes']
                            break

            if (not use_stored):
                # # generate something to test it, later implement how to set it
                # sizes = np.random.randint(low=1, high=10, size=len(vs), )

                if ('extra' in ci.keys()):
                    if ('MINIMAL_VARIABLE_SIZE_AND_DEPTH' in ci['extra'].keys()):
                        sizes = ci['extra']['MINIMAL_VARIABLE_SIZE_AND_DEPTH']['sizes']
                    else:
                        sizes = np.random.randint(low=1, high=10, size=len(vs), )
                else:
                    sizes = np.random.randint(low=1, high=10, size=len(vs), )

                if ('extra' in ci.keys()):
                    for k, v in ci['extra'].items():
                        if (k in ('MINIMAL_VARIABLE_SIZE', 'MINIMAL_VARIABLE_SIZE_AND_DEPTH', 'RICH_BILLBOARD',
                                  'RICH_BILLBOARD_NO_DEPTH',)):
                            # FIXME: it was recently switched, try to recover already generated data, both arrays are the same, so thay are in memory just once, no problem here, the problem is with storing reference to it, this should be fixed with something, for example, shader and batch should be generated on one spot, etc.. but have to be done together with all new PCVManager.render and unified shader management. this is just mediocre workaround..
                            sizes = ci['extra'][k]['sizes']
                            sizesf = ci['extra'][k]['sizesf']
                            break

                shader = GPUShader(PCVShaders.vertex_shader_minimal_variable_size_and_depth,
                                   PCVShaders.fragment_shader_minimal_variable_size_and_depth, )
                batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], "size": sizes[:l], })

                if ('extra' not in ci.keys()):
                    ci['extra'] = {}

                d = {'shader': shader,
                     'batch': batch,
                     'sizes': sizes,
                     'length': l, }
                ci['extra']['MINIMAL_VARIABLE_SIZE_AND_DEPTH'] = d

            shader.bind()
            pm = bpy.context.region_data.perspective_matrix
            shader.uniform_float("perspective_matrix", pm)
            shader.uniform_float("object_matrix", o.matrix_world)
            shader.uniform_float("global_alpha", pcv.global_alpha)

            if (len(vs) == 0):
                maxdist = 1.0
                cx = 0.0
                cy = 0.0
                cz = 0.0
            else:
                # NOTE: precalculating and storing following should speed up things a bit, but then it won't reflect edits..
                # TODO: calculating center of all points is not quite correct, visually it works, but (as i've seen in bounding box shader) it's not working when distribution of points is uneven, so have a check if it might be a bit better..
                cx = np.sum(vs[:, 0]) / len(vs)
                cy = np.sum(vs[:, 1]) / len(vs)
                cz = np.sum(vs[:, 2]) / len(vs)
                _, _, s = o.matrix_world.decompose()
                l = s.length
                # FIXME: here is error in max with zero length arrays, why are they zero length anyway, putting this single fix for now
                maxd = abs(np.max(vs))
                mind = abs(np.min(vs))
                maxdist = maxd
                if (mind > maxd):
                    maxdist = mind
            shader.uniform_float("maxdist", float(maxdist) * l)
            shader.uniform_float("center", (cx, cy, cz,))
            shader.uniform_float("brightness", pcv.dev_minimal_shader_variable_size_and_depth_brightness)
            shader.uniform_float("contrast", pcv.dev_minimal_shader_variable_size_and_depth_contrast)
            shader.uniform_float("blend", 1.0 - pcv.dev_minimal_shader_variable_size_and_depth_blend)

            batch.draw(shader)

        # dev
        if (pcv.dev_billboard_point_cloud_enabled):
            vs = ci['vertices']
            cs = ci['colors']
            l = ci['current_display_length']

            use_stored = False
            if ('extra' in ci.keys()):
                t = 'BILLBOARD'
                for k, v in ci['extra'].items():
                    if (k == t):
                        if (v['length'] == l):
                            use_stored = True
                            batch = v['batch']
                            shader = v['shader']
                            break

            if (not use_stored):
                shader = GPUShader(PCVShaders.billboard_vertex, PCVShaders.billboard_fragment,
                                   geocode=PCVShaders.billboard_geometry, )
                # shader = GPUShader(PCVShaders.billboard_vertex, PCVShaders.billboard_fragment, geocode=PCVShaders.billboard_geometry_disc, )
                batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], })

                if ('extra' not in ci.keys()):
                    ci['extra'] = {}
                d = {'shader': shader,
                     'batch': batch,
                     'length': l, }
                ci['extra']['BILLBOARD'] = d

            shader.bind()
            shader.uniform_float("view_matrix", bpy.context.region_data.view_matrix)
            shader.uniform_float("window_matrix", bpy.context.region_data.window_matrix)
            shader.uniform_float("object_matrix", o.matrix_world)
            shader.uniform_float("size", pcv.dev_billboard_point_cloud_size)
            shader.uniform_float("alpha", pcv.global_alpha)
            batch.draw(shader)

        # dev
        if (pcv.dev_rich_billboard_point_cloud_enabled):
            vs = ci['vertices']
            cs = ci['colors']
            l = ci['current_display_length']

            use_stored = False
            if ('extra' in ci.keys()):
                t = 'RICH_BILLBOARD'
                for k, v in ci['extra'].items():
                    if (k == t):
                        if (v['length'] == l):
                            use_stored = True
                            batch = v['batch']
                            shader = v['shader']
                            sizesf = v['sizesf']
                            break

            if (not use_stored):

                if ('extra' in ci.keys()):
                    if ('RICH_BILLBOARD' in ci['extra'].keys()):
                        sizesf = ci['extra']['RICH_BILLBOARD']['sizesf']
                    else:
                        sizesf = np.random.uniform(low=0.5, high=1.5, size=len(vs), )
                        sizesf = sizesf.astype(np.float32)
                else:
                    sizesf = np.random.uniform(low=0.5, high=1.5, size=len(vs), )
                    sizesf = sizesf.astype(np.float32)

                if ('extra' in ci.keys()):
                    for k, v in ci['extra'].items():
                        if (k in ('MINIMAL_VARIABLE_SIZE', 'MINIMAL_VARIABLE_SIZE_AND_DEPTH', 'RICH_BILLBOARD',
                                  'RICH_BILLBOARD_NO_DEPTH',)):
                            # FIXME: it was recently switched, try to recover already generated data, both arrays are the same, so thay are in memory just once, no problem here, the problem is with storing reference to it, this should be fixed with something, for example, shader and batch should be generated on one spot, etc.. but have to be done together with all new PCVManager.render and unified shader management. this is just mediocre workaround..
                            sizes = ci['extra'][k]['sizes']
                            sizesf = ci['extra'][k]['sizesf']
                            break

                shader = GPUShader(PCVShaders.billboard_vertex_with_depth_and_size,
                                   PCVShaders.billboard_fragment_with_depth_and_size,
                                   geocode=PCVShaders.billboard_geometry_with_depth_and_size, )
                batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], "sizef": sizesf[:l], })

                if ('extra' not in ci.keys()):
                    ci['extra'] = {}
                d = {'shader': shader,
                     'batch': batch,
                     'sizesf': sizesf,
                     'length': l, }
                ci['extra']['RICH_BILLBOARD'] = d

            shader.bind()
            shader.uniform_float("perspective_matrix", bpy.context.region_data.perspective_matrix)
            shader.uniform_float("view_matrix", bpy.context.region_data.view_matrix)
            shader.uniform_float("window_matrix", bpy.context.region_data.window_matrix)
            shader.uniform_float("object_matrix", o.matrix_world)
            shader.uniform_float("size", pcv.dev_rich_billboard_point_cloud_size)
            shader.uniform_float("alpha", pcv.global_alpha)

            if (len(vs) == 0):
                maxdist = 1.0
                cx = 0.0
                cy = 0.0
                cz = 0.0
            else:
                # NOTE: precalculating and storing following should speed up things a bit, but then it won't reflect edits..
                # TODO: calculating center of all points is not quite correct, visually it works, but (as i've seen in bounding box shader) it's not working when distribution of points is uneven, so have a check if it might be a bit better..
                cx = np.sum(vs[:, 0]) / len(vs)
                cy = np.sum(vs[:, 1]) / len(vs)
                cz = np.sum(vs[:, 2]) / len(vs)
                _, _, s = o.matrix_world.decompose()
                l = s.length
                # FIXME: here is error in max with zero length arrays, why are they zero length anyway, putting this single fix for now
                maxd = abs(np.max(vs))
                mind = abs(np.min(vs))
                maxdist = maxd
                if (mind > maxd):
                    maxdist = mind
            shader.uniform_float("maxdist", float(maxdist) * l)
            shader.uniform_float("center", (cx, cy, cz,), )

            shader.uniform_float("brightness", pcv.dev_rich_billboard_depth_brightness)
            shader.uniform_float("contrast", pcv.dev_rich_billboard_depth_contrast)
            shader.uniform_float("blend", 1.0 - pcv.dev_rich_billboard_depth_blend)

            batch.draw(shader)

        # dev
        if (pcv.dev_rich_billboard_point_cloud_no_depth_enabled):
            vs = ci['vertices']
            cs = ci['colors']
            l = ci['current_display_length']

            use_stored = False
            if ('extra' in ci.keys()):
                t = 'RICH_BILLBOARD_NO_DEPTH'
                for k, v in ci['extra'].items():
                    if (k == t):
                        if (v['length'] == l):
                            use_stored = True
                            batch = v['batch']
                            shader = v['shader']
                            sizesf = v['sizesf']
                            break

            if (not use_stored):

                if ('extra' in ci.keys()):
                    if ('RICH_BILLBOARD_NO_DEPTH' in ci['extra'].keys()):
                        sizesf = ci['extra']['RICH_BILLBOARD_NO_DEPTH']['sizesf']
                    else:
                        sizesf = np.random.uniform(low=0.5, high=1.5, size=len(vs), )
                        sizesf = sizesf.astype(np.float32)
                else:
                    sizesf = np.random.uniform(low=0.5, high=1.5, size=len(vs), )
                    sizesf = sizesf.astype(np.float32)

                if ('extra' in ci.keys()):
                    for k, v in ci['extra'].items():
                        if (k in ('MINIMAL_VARIABLE_SIZE', 'MINIMAL_VARIABLE_SIZE_AND_DEPTH', 'RICH_BILLBOARD',
                                  'RICH_BILLBOARD_NO_DEPTH',)):
                            # FIXME: it was recently switched, try to recover already generated data, both arrays are the same, so thay are in memory just once, no problem here, the problem is with storing reference to it, this should be fixed with something, for example, shader and batch should be generated on one spot, etc.. but have to be done together with all new PCVManager.render and unified shader management. this is just mediocre workaround..
                            sizes = ci['extra'][k]['sizes']
                            sizesf = ci['extra'][k]['sizesf']
                            break

                shader = GPUShader(PCVShaders.billboard_vertex_with_no_depth_and_size,
                                   PCVShaders.billboard_fragment_with_no_depth_and_size,
                                   geocode=PCVShaders.billboard_geometry_with_no_depth_and_size, )
                batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], "sizef": sizesf[:l], })

                if ('extra' not in ci.keys()):
                    ci['extra'] = {}
                d = {'shader': shader,
                     'batch': batch,
                     'sizesf': sizesf,
                     'length': l, }
                ci['extra']['RICH_BILLBOARD_NO_DEPTH'] = d

            shader.bind()
            # shader.uniform_float("perspective_matrix", bpy.context.region_data.perspective_matrix)
            shader.uniform_float("view_matrix", bpy.context.region_data.view_matrix)
            shader.uniform_float("window_matrix", bpy.context.region_data.window_matrix)
            shader.uniform_float("object_matrix", o.matrix_world)
            shader.uniform_float("size", pcv.dev_rich_billboard_point_cloud_size)
            shader.uniform_float("alpha", pcv.global_alpha)

            batch.draw(shader)

        # dev
        if (pcv.dev_phong_shader_enabled):
            vs = ci['vertices']
            ns = ci['normals']
            cs = ci['colors']
            l = ci['current_display_length']

            use_stored = False
            if ('extra' in ci.keys()):
                t = 'PHONG'
                for k, v in ci['extra'].items():
                    if (k == t):
                        if (v['length'] == l):
                            use_stored = True
                            batch = v['batch']
                            shader = v['shader']
                            break

            if (not use_stored):
                shader = GPUShader(PCVShaders.phong_vs, PCVShaders.phong_fs, )
                batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "normal": ns[:l], "color": cs[:l], })

                if ('extra' not in ci.keys()):
                    ci['extra'] = {}
                d = {'shader': shader,
                     'batch': batch,
                     'length': l, }
                ci['extra']['PHONG'] = d

            shader.bind()

            # shader.uniform_float("perspective_matrix", bpy.context.region_data.perspective_matrix)
            shader.uniform_float("view", bpy.context.region_data.view_matrix)
            shader.uniform_float("projection", bpy.context.region_data.window_matrix)
            shader.uniform_float("model", o.matrix_world)

            shader.uniform_float("light_position", bpy.context.region_data.view_matrix.inverted().translation)
            # shader.uniform_float("light_color", (1.0, 1.0, 1.0))
            shader.uniform_float("light_color", (0.8, 0.8, 0.8,))
            shader.uniform_float("view_position", bpy.context.region_data.view_matrix.inverted().translation)

            shader.uniform_float("ambient_strength", pcv.dev_phong_shader_ambient_strength)
            shader.uniform_float("specular_strength", pcv.dev_phong_shader_specular_strength)
            shader.uniform_float("specular_exponent", pcv.dev_phong_shader_specular_exponent)

            shader.uniform_float("alpha", pcv.global_alpha)
            shader.uniform_float("point_size", pcv.point_size)
            shader.uniform_float("alpha_radius", pcv.alpha_radius)

            # pm = bpy.context.region_data.perspective_matrix
            # shader.uniform_float("perspective_matrix", pm)
            # shader.uniform_float("object_matrix", o.matrix_world)
            # shader.uniform_float("point_size", pcv.point_size)
            # shader.uniform_float("global_alpha", pcv.global_alpha)
            batch.draw(shader)

        # dev
        if (pcv.clip_shader_enabled):
            vs = ci['vertices']
            cs = ci['colors']
            l = ci['current_display_length']

            use_stored = False
            if ('extra' in ci.keys()):
                t = 'CLIP'
                for k, v in ci['extra'].items():
                    if (k == t):
                        if (v['length'] == l):
                            use_stored = True
                            batch = v['batch']
                            shader = v['shader']
                            break

            if (not use_stored):
                shader = GPUShader(PCVShaders.vertex_shader_simple_clip, PCVShaders.fragment_shader_simple_clip, )
                batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], })

                if ('extra' not in ci.keys()):
                    ci['extra'] = {}
                d = {'shader': shader,
                     'batch': batch,
                     'length': l, }
                ci['extra']['CLIP'] = d

            if (pcv.clip_plane0_enabled):
                bgl.glEnable(bgl.GL_CLIP_DISTANCE0)
            if (pcv.clip_plane1_enabled):
                bgl.glEnable(bgl.GL_CLIP_DISTANCE1)
            if (pcv.clip_plane2_enabled):
                bgl.glEnable(bgl.GL_CLIP_DISTANCE2)
            if (pcv.clip_plane3_enabled):
                bgl.glEnable(bgl.GL_CLIP_DISTANCE3)
            if (pcv.clip_plane4_enabled):
                bgl.glEnable(bgl.GL_CLIP_DISTANCE4)
            if (pcv.clip_plane5_enabled):
                bgl.glEnable(bgl.GL_CLIP_DISTANCE5)

            shader.bind()

            shader.uniform_float("perspective_matrix", bpy.context.region_data.perspective_matrix)
            shader.uniform_float("object_matrix", o.matrix_world)
            shader.uniform_float("point_size", pcv.point_size)
            shader.uniform_float("alpha_radius", pcv.alpha_radius)
            shader.uniform_float("global_alpha", pcv.global_alpha)

            shader.uniform_float("clip_plane0", pcv.clip_plane0)
            shader.uniform_float("clip_plane1", pcv.clip_plane1)
            shader.uniform_float("clip_plane2", pcv.clip_plane2)
            shader.uniform_float("clip_plane3", pcv.clip_plane3)
            shader.uniform_float("clip_plane4", pcv.clip_plane4)
            shader.uniform_float("clip_plane5", pcv.clip_plane5)

            batch.draw(shader)

            if (pcv.clip_plane0_enabled):
                bgl.glDisable(bgl.GL_CLIP_DISTANCE0)
            if (pcv.clip_plane1_enabled):
                bgl.glDisable(bgl.GL_CLIP_DISTANCE1)
            if (pcv.clip_plane2_enabled):
                bgl.glDisable(bgl.GL_CLIP_DISTANCE2)
            if (pcv.clip_plane3_enabled):
                bgl.glDisable(bgl.GL_CLIP_DISTANCE3)
            if (pcv.clip_plane4_enabled):
                bgl.glDisable(bgl.GL_CLIP_DISTANCE4)
            if (pcv.clip_plane5_enabled):
                bgl.glDisable(bgl.GL_CLIP_DISTANCE5)

        # dev
        if (pcv.billboard_phong_enabled):
            vs = ci['vertices']
            ns = ci['normals']
            cs = ci['colors']
            l = ci['current_display_length']

            use_stored = False
            if ('extra' in ci.keys()):
                t = 'BILLBOARD_PHONG'
                for k, v in ci['extra'].items():
                    if (k == t):
                        if (v['circles'] != pcv.billboard_phong_circles):
                            use_stored = False
                            break
                        if (v['length'] == l):
                            use_stored = True
                            batch = v['batch']
                            shader = v['shader']
                            break

            if (not use_stored):
                use_geocode = PCVShaders.billboard_phong_fast_gs
                if (pcv.billboard_phong_circles):
                    use_geocode = PCVShaders.billboard_phong_circles_gs
                shader = GPUShader(PCVShaders.billboard_phong_vs, PCVShaders.billboard_phong_fs, geocode=use_geocode, )
                batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "normal": ns[:l], "color": cs[:l], })

                if ('extra' not in ci.keys()):
                    ci['extra'] = {}
                d = {'shader': shader,
                     'batch': batch,
                     'circles': pcv.billboard_phong_circles,
                     'length': l, }
                ci['extra']['BILLBOARD_PHONG'] = d

            shader.bind()

            shader.uniform_float("model", o.matrix_world)
            # shader.uniform_float("view", bpy.context.region_data.view_matrix)
            # shader.uniform_float("projection", bpy.context.region_data.window_matrix)

            shader.uniform_float("view_matrix", bpy.context.region_data.view_matrix)
            shader.uniform_float("window_matrix", bpy.context.region_data.window_matrix)

            shader.uniform_float("size", pcv.billboard_phong_size)

            shader.uniform_float("alpha", pcv.global_alpha)

            shader.uniform_float("light_position", bpy.context.region_data.view_matrix.inverted().translation)
            shader.uniform_float("light_color", (0.8, 0.8, 0.8,))
            shader.uniform_float("view_position", bpy.context.region_data.view_matrix.inverted().translation)

            shader.uniform_float("ambient_strength", pcv.billboard_phong_ambient_strength)
            shader.uniform_float("specular_strength", pcv.billboard_phong_specular_strength)
            shader.uniform_float("specular_exponent", pcv.billboard_phong_specular_exponent)

            batch.draw(shader)

        # dev
        if (pcv.skip_point_shader_enabled):
            vs = ci['vertices']
            cs = ci['colors']

            use_stored = False
            if ('extra' in ci.keys()):
                t = 'SKIP'
                for k, v in ci['extra'].items():
                    if (k == t):
                        use_stored = True
                        batch = v['batch']
                        shader = v['shader']
                        break

            if (not use_stored):
                indices = np.indices((len(vs),), dtype=np.int, )
                indices.shape = (-1,)

                shader = GPUShader(PCVShaders.vertex_shader_simple_skip_point_vertices,
                                   PCVShaders.fragment_shader_simple_skip_point_vertices, )
                batch = batch_for_shader(shader, 'POINTS', {"position": vs[:], "color": cs[:], "index": indices[:], })

                if ('extra' not in ci.keys()):
                    ci['extra'] = {}
                d = {'shader': shader,
                     'batch': batch, }
                ci['extra']['SKIP'] = d

            shader.bind()

            shader.uniform_float("perspective_matrix", bpy.context.region_data.perspective_matrix)
            shader.uniform_float("object_matrix", o.matrix_world)
            shader.uniform_float("point_size", pcv.point_size)
            shader.uniform_float("alpha_radius", pcv.alpha_radius)
            shader.uniform_float("global_alpha", pcv.global_alpha)

            sp = pcv.skip_point_percentage
            l = int((len(vs) / 100) * sp)
            if (sp >= 99):
                l = len(vs)
            shader.uniform_float("skip_index", l)

            batch.draw(shader)

        # and now back to some production stuff..

        # draw selection as a last step bucause i clear depth buffer for it
        if (pcv.filter_remove_color_selection):
            if ('selection_indexes' not in ci):
                return
            vs = ci['vertices']
            indexes = ci['selection_indexes']
            try:
                # if it works, leave it..
                vs = np.take(vs, indexes, axis=0, )
            except IndexError:
                # something has changed.. some other edit hapended, selection is invalid, reset it all..
                pcv.filter_remove_color_selection = False
                del ci['selection_indexes']

            shader = GPUShader(PCVShaders.selection_vertex_shader, PCVShaders.selection_fragment_shader, )
            batch = batch_for_shader(shader, 'POINTS', {"position": vs[:], })
            shader.bind()
            pm = bpy.context.region_data.perspective_matrix
            shader.uniform_float("perspective_matrix", pm)
            shader.uniform_float("object_matrix", o.matrix_world)
            sc = bpy.context.preferences.addons[__name__].preferences.selection_color[:]
            shader.uniform_float("color", sc)
            shader.uniform_float("point_size", pcv.point_size)
            shader.uniform_float("alpha_radius", pcv.alpha_radius)
            bgl.glClear(bgl.GL_DEPTH_BUFFER_BIT)
            batch.draw(shader)

        bgl.glDisable(bgl.GL_PROGRAM_POINT_SIZE)
        bgl.glDisable(bgl.GL_DEPTH_TEST)
        bgl.glDisable(bgl.GL_BLEND)

    @classmethod
    def handler(cls):
        bobjects = bpy.data.objects

        run_gc = False
        for k, v in cls.cache.items():
            if (not bobjects.get(v['name'])):
                v['kill'] = True
                run_gc = True
            if (v['ready'] and v['draw'] and not v['kill']):
                cls.render(v['uuid'])
        if (run_gc):
            cls.gc()

    @classmethod
    def update(cls, uuid, vs, ns=None, cs=None, ):
        if (uuid not in PCVManager.cache):
            raise KeyError("uuid '{}' not in cache".format(uuid))
        # if(len(vs) == 0):
        #     raise ValueError("zero length")

        # get cache item
        c = PCVManager.cache[uuid]
        l = len(vs)

        if (ns is None):
            ns = np.column_stack((np.full(l, 0.0, dtype=np.float32, ),
                                  np.full(l, 0.0, dtype=np.float32, ),
                                  np.full(l, 1.0, dtype=np.float32, ),))

        if (cs is None):
            col = bpy.context.preferences.addons[__name__].preferences.default_vertex_color[:]
            col = tuple([c ** (1 / 2.2) for c in col]) + (1.0,)
            cs = np.column_stack((np.full(l, col[0], dtype=np.float32, ),
                                  np.full(l, col[1], dtype=np.float32, ),
                                  np.full(l, col[2], dtype=np.float32, ),
                                  np.ones(l, dtype=np.float32, ),))

        # store data
        c['vertices'] = vs
        c['normals'] = ns
        c['colors'] = cs
        c['length'] = l
        c['stats'] = l

        o = c['object']
        pcv = o.point_cloud_visualizer
        dp = pcv.display_percent
        nl = int((l / 100) * dp)
        if (dp >= 99):
            nl = l
        c['display_length'] = nl
        c['current_display_length'] = nl

        # setup new shaders
        ienabled = c['illumination']
        if (ienabled):
            shader = GPUShader(PCVShaders.vertex_shader_illumination, PCVShaders.fragment_shader_illumination)
            batch = batch_for_shader(shader, 'POINTS', {"position": vs[:nl], "color": cs[:nl], "normal": ns[:nl], })
        else:
            shader = GPUShader(PCVShaders.vertex_shader_simple, PCVShaders.fragment_shader_simple)
            batch = batch_for_shader(shader, 'POINTS', {"position": vs[:nl], "color": cs[:nl], })
        c['shader'] = shader
        c['batch'] = batch

        # redraw all viewports
        for area in bpy.context.screen.areas:
            if (area.type == 'VIEW_3D'):
                area.tag_redraw()

    @classmethod
    def gc(cls):
        l = []
        for k, v in cls.cache.items():
            if (v['kill']):
                l.append(k)
        for i in l:
            del cls.cache[i]

    @classmethod
    def init(cls):
        if (cls.initialized):
            # print("Ply loading tool is ready")
            return
        cls.handle = bpy.types.SpaceView3D.draw_handler_add(cls.handler, (), 'WINDOW', 'POST_VIEW')
        bpy.app.handlers.load_pre.append(watcher)
        cls.initialized = True
        # print("Ply loading tool is ready")

    @classmethod
    def deinit(cls):
        if (not cls.initialized):
            return
        for k, v in cls.cache.items():
            v['kill'] = True
        cls.gc()

        bpy.types.SpaceView3D.draw_handler_remove(cls.handle, 'WINDOW')
        cls.handle = None
        bpy.app.handlers.load_pre.remove(watcher)
        cls.initialized = False

    @classmethod
    def add(cls, data, ):
        cls.cache[data['uuid']] = data

    @classmethod
    def new(cls):
        # NOTE: this is redundant.. is it?
        return {'uuid': None,
                'filepath': None,
                'vertices': None,
                'normals': None,
                'colors': None,
                'display_length': None,
                'current_display_length': None,
                'illumination': False,
                'shader': False,
                'batch': False,
                'ready': False,
                'draw': False,
                'kill': False,
                'stats': None,
                'length': None,
                'name': None,
                'object': None, }

    @classmethod
    def _redraw(cls):

        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if (area.type == 'VIEW_3D'):
                    area.tag_redraw()


class PCV_properties(PropertyGroup):
    filepath: StringProperty(name="PLY File", default="", description="", )
    uuid: StringProperty(default="", options={'HIDDEN', }, )

    def _instance_visualizer_active_get(self, ):
        return self.instance_visualizer_active_hidden_value

    def _instance_visualizer_active_set(self, value, ):
        pass

    # for setting value, there are handlers for save, pre which sets to False, and post which sets back to True if it was True before, instance visualizer have to be activated at runtime and this value should not be saved, this way it works ok.. if only there was a way to specify which properties should not save, and/or save only as default value..
    instance_visualizer_active_hidden_value: BoolProperty(default=False, options={'HIDDEN', }, )
    # for display, read-only
    instance_visualizer_active: BoolProperty(name="Instance Visualizer Active", default=False,
                                             get=_instance_visualizer_active_get, set=_instance_visualizer_active_set, )

    runtime: BoolProperty(default=False, options={'HIDDEN', }, )

    # TODO: add some prefix to global props, like global_size, global_display_percent, .. leave unprefixed only essentials, like uuid, runtime, ..
    point_size: IntProperty(name="Size", default=3, min=1, max=10, subtype='PIXEL', description="Point size", )
    alpha_radius: FloatProperty(name="Radius", default=1.0, min=0.001, max=1.0, precision=3, subtype='FACTOR',
                                description="Adjust point circular discard radius", )

    def _display_percent_update(self, context, ):
        if (self.uuid not in PCVManager.cache):
            return
        d = PCVManager.cache[self.uuid]
        dp = self.display_percent
        vl = d['length']
        l = int((vl / 100) * dp)
        if (dp >= 99):
            l = vl
        d['display_length'] = l

    display_percent: FloatProperty(name="Display", default=100.0, min=0.0, max=100.0, precision=0, subtype='PERCENTAGE',
                                   update=_display_percent_update,
                                   description="Adjust percentage of points displayed", )
    global_alpha: FloatProperty(name="Alpha", default=1.0, min=0.0, max=1.0, precision=2, subtype='FACTOR',
                                description="Adjust alpha of points displayed", )

    vertex_normals: BoolProperty(name="Normals", description="Draw normals of points", default=False, )
    vertex_normals_size: FloatProperty(name="Length", description="Length of point normal line", default=0.001,
                                       min=0.00001, max=1.0, soft_min=0.001, soft_max=0.2, step=1, precision=3, )
    vertex_normals_alpha: FloatProperty(name="Alpha", description="Alpha of point normal line", default=0.5, min=0.0,
                                        max=1.0, soft_min=0.0, soft_max=1.0, step=1, precision=3, )

    render_point_size: IntProperty(name="Size", default=3, min=1, max=100, subtype='PIXEL', description="Point size", )
    render_display_percent: FloatProperty(name="Count", default=100.0, min=0.0, max=100.0, precision=0,
                                          subtype='PERCENTAGE', description="Adjust percentage of points rendered", )
    render_path: StringProperty(name="Output Path", default="//pcv_render_###.png",
                                description="Directory/name to save rendered images, # characters defines the position and length of frame numbers, filetype is always png",
                                subtype='FILE_PATH', )
    render_resolution_x: IntProperty(name="Resolution X", default=1920, min=4, max=65536,
                                     description="Number of horizontal pixels in rendered image", subtype='PIXEL', )
    render_resolution_y: IntProperty(name="Resolution Y", default=1080, min=4, max=65536,
                                     description="Number of vertical pixels in rendered image", subtype='PIXEL', )
    render_resolution_percentage: IntProperty(name="Resolution %", default=10, min=1, max=100,
                                              description="Percentage scale for render resolution",
                                              subtype='PERCENTAGE', )
    render_smoothstep: BoolProperty(name="Smooth Circles", default=False,
                                    description="Currently works only for basic shader with/without illumination and generally is much slower than Supersampling, use only when Supersampling fails", )
    render_supersampling: IntProperty(name="Supersampling", default=1, soft_min=1, soft_max=4, min=1, max=10,
                                      description="Render larger image and then resize back, 1 - disabled, 2 - render 200%, 3 - render 300%, ...", )

    def _render_resolution_linked_update(self, context, ):
        if (not self.render_resolution_linked):
            # now it is False, so it must have been True, so for convenience, copy values
            r = context.scene.render
            self.render_resolution_x = r.resolution_x
            self.render_resolution_y = r.resolution_y
            self.render_resolution_percentage = r.resolution_percentage

    render_resolution_linked: BoolProperty(name="Resolution Linked", description="Link resolution settings to scene",
                                           default=True, update=_render_resolution_linked_update, )

    has_normals: BoolProperty(default=False, options={'HIDDEN', }, )
    # TODO: rename to 'has_colors'
    has_vcols: BoolProperty(default=False, options={'HIDDEN', }, )
    illumination: BoolProperty(name="Illumination", description="Enable extra illumination on point cloud",
                               default=False, )
    illumination_edit: BoolProperty(name="Edit", description="Edit illumination properties", default=False, )
    light_direction: FloatVectorProperty(name="Light Direction", description="Light direction", default=(0.0, 1.0, 0.0),
                                         subtype='DIRECTION', size=3, )
    # light_color: FloatVectorProperty(name="Light Color", description="", default=(0.2, 0.2, 0.2), min=0, max=1, subtype='COLOR', size=3, )
    light_intensity: FloatProperty(name="Light Intensity", description="Light intensity", default=0.3, min=0, max=1,
                                   subtype='FACTOR', )
    shadow_intensity: FloatProperty(name="Shadow Intensity", description="Shadow intensity", default=0.2, min=0, max=1,
                                    subtype='FACTOR', )
    # show_normals: BoolProperty(name="Colorize By Vertex Normals", description="", default=False, )

    mesh_type: EnumProperty(name="Type", items=[('VERTEX', "Vertex", ""),
                                                ('TRIANGLE', "Equilateral Triangle", ""),
                                                ('TETRAHEDRON', "Tetrahedron", ""),
                                                ('CUBE', "Cube", ""),
                                                ('ICOSPHERE', "Ico Sphere", ""),
                                                ('INSTANCER', "Instancer", ""),
                                                ('PARTICLES', "Particle System", ""), ], default='CUBE',
                            description="Instance mesh type", )
    mesh_size: FloatProperty(name="Size", description="Mesh instance size, instanced mesh has size 1.0", default=0.01,
                             min=0.000001, precision=4, max=100.0, )
    mesh_normal_align: BoolProperty(name="Align To Normal", description="Align instance to point normal",
                                    default=True, )
    mesh_vcols: BoolProperty(name="Colors", description="Assign point color to instance vertex colors", default=True, )
    mesh_all: BoolProperty(name="All", description="Convert all points", default=True, )
    mesh_percentage: FloatProperty(name="Subset", default=100.0, min=0.0, max=100.0, precision=0, subtype='PERCENTAGE',
                                   description="Convert random subset of points by given percentage", )
    mesh_base_sphere_subdivisions: IntProperty(name="Sphere Subdivisions", default=2, min=1, max=6,
                                               description="Particle instance (Ico Sphere) subdivisions, instance mesh can be change later", )
    mesh_use_instancer2: BoolProperty(name="Use Faster Conversion",
                                      description="Faster (especially with icosphere) Numpy implementation, use if you don't mind all triangles in result",
                                      default=False, )

    export_use_viewport: BoolProperty(name="Use Viewport Points", default=True,
                                      description="When checked, export points currently displayed in viewport or when unchecked, export data loaded from original ply file", )
    export_apply_transformation: BoolProperty(name="Apply Transformation", default=False,
                                              description="Apply parent object transformation to points", )
    export_convert_axes: BoolProperty(name="Convert Axes", default=False,
                                      description="Convert from blender (y forward, z up) to forward -z, up y axes", )
    export_visible_only: BoolProperty(name="Visible Points Only", default=False,
                                      description="Export currently visible points only (controlled by 'Display' on main panel)", )

    filter_simplify_num_samples: IntProperty(name="Samples", default=10000, min=1, subtype='NONE',
                                             description="Number of points in simplified point cloud, best result when set to less than 20% of points, when samples has value close to total expect less points in result", )
    filter_simplify_num_candidates: IntProperty(name="Candidates", default=10, min=3, max=100, subtype='NONE',
                                                description="Number of candidates used during resampling, the higher value, the slower calculation, but more even", )

    filter_remove_color: FloatVectorProperty(name="Color", default=(1.0, 1.0, 1.0,), min=0, max=1, subtype='COLOR',
                                             size=3, description="Color to remove from point cloud", )
    filter_remove_color_delta_hue: FloatProperty(name="Δ Hue", default=0.1, min=0.0, max=1.0, precision=3,
                                                 subtype='FACTOR', description="", )
    filter_remove_color_delta_hue_use: BoolProperty(name="Use Δ Hue", description="", default=True, )
    filter_remove_color_delta_saturation: FloatProperty(name="Δ Saturation", default=0.1, min=0.0, max=1.0, precision=3,
                                                        subtype='FACTOR', description="", )
    filter_remove_color_delta_saturation_use: BoolProperty(name="Use Δ Saturation", description="", default=True, )
    filter_remove_color_delta_value: FloatProperty(name="Δ Value", default=0.1, min=0.0, max=1.0, precision=3,
                                                   subtype='FACTOR', description="", )
    filter_remove_color_delta_value_use: BoolProperty(name="Use Δ Value", description="", default=True, )
    filter_remove_color_selection: BoolProperty(default=False, options={'HIDDEN', }, )

    def _project_positive_radio_update(self, context):
        if (not self.filter_project_negative and not self.filter_project_positive):
            self.filter_project_negative = True

    def _project_negative_radio_update(self, context):
        if (not self.filter_project_negative and not self.filter_project_positive):
            self.filter_project_positive = True

    def _filter_project_object_poll(self, o, ):
        if (o and o.type in ('MESH', 'CURVE', 'SURFACE', 'FONT',)):
            return True
        return False

    filter_project_object: PointerProperty(type=bpy.types.Object, name="Object", description="",
                                           poll=_filter_project_object_poll, )
    filter_project_search_distance: FloatProperty(name="Search Distance", default=0.1, min=0.0, max=10000.0,
                                                  precision=3, subtype='DISTANCE',
                                                  description="Maximum search distance in which to search for surface", )
    filter_project_positive: BoolProperty(name="Positive", description="Search along point normal forwards",
                                          default=True, update=_project_positive_radio_update, )
    filter_project_negative: BoolProperty(name="Negative", description="Search along point normal backwards",
                                          default=True, update=_project_negative_radio_update, )
    filter_project_discard: BoolProperty(name="Discard Unprojectable",
                                         description="Discard points which didn't hit anything", default=False, )
    filter_project_colorize: BoolProperty(name="Colorize", description="Colorize projected points", default=False, )
    filter_project_colorize_from: EnumProperty(name="Source", items=[
        ('VCOLS', "Vertex Colors", "Use active vertex colors from target"),
        ('UVTEX', "UV Texture",
         "Use colors from active image texture node in active material using active UV layout from target"),
        ('GROUP_MONO', "Vertex Group Monochromatic",
         "Use active vertex group from target, result will be shades of grey"),
        ('GROUP_COLOR', "Vertex Group Colorized",
         "Use active vertex group from target, result will be colored from red (1.0) to blue (0.0) like in weight paint viewport"),
        ], default='UVTEX', description="Color source for projected point cloud", )
    filter_project_shift: FloatProperty(name="Shift", default=0.0, precision=3, subtype='DISTANCE',
                                        description="Shift points after projection above (positive) or below (negative) surface", )

    def _filter_boolean_object_poll(self, o, ):
        if (o and o.type in ('MESH', 'CURVE', 'SURFACE', 'FONT',)):
            return True
        return False

    filter_boolean_object: PointerProperty(type=bpy.types.Object, name="Object", description="",
                                           poll=_filter_boolean_object_poll, )

    def _filter_join_object_poll(self, o, ):
        ok = False
        if (o):
            pcv = o.point_cloud_visualizer
            if (pcv.uuid != ''):
                for k, v in PCVManager.cache.items():
                    if (v['uuid'] == pcv.uuid):
                        if (v['ready']):
                            # if(v['draw']):
                            #     ok = True
                            ok = True
                        break
        return ok

    filter_join_object: PointerProperty(type=bpy.types.Object, name="Object", description="",
                                        poll=_filter_join_object_poll, )

    edit_initialized: BoolProperty(default=False, options={'HIDDEN', }, )
    edit_is_edit_mesh: BoolProperty(default=False, options={'HIDDEN', }, )
    edit_is_edit_uuid: StringProperty(default="", options={'HIDDEN', }, )
    edit_pre_edit_alpha: FloatProperty(default=0.5, options={'HIDDEN', }, )
    edit_pre_edit_display: FloatProperty(default=100.0, options={'HIDDEN', }, )
    edit_pre_edit_size: IntProperty(default=3, options={'HIDDEN', }, )

    def _edit_overlay_alpha_update(self, context, ):
        o = context.object
        p = o.parent
        pcv = p.point_cloud_visualizer
        pcv.global_alpha = self.edit_overlay_alpha

    def _edit_overlay_size_update(self, context, ):
        o = context.object
        p = o.parent
        pcv = p.point_cloud_visualizer
        pcv.point_size = self.edit_overlay_size

    edit_overlay_alpha: FloatProperty(name="Overlay Alpha", default=0.5, min=0.0, max=1.0, precision=2,
                                      subtype='FACTOR', description="Overlay point alpha",
                                      update=_edit_overlay_alpha_update, )
    edit_overlay_size: IntProperty(name="Overlay Size", default=3, min=1, max=10, subtype='PIXEL',
                                   description="Overlay point size", update=_edit_overlay_size_update, )

    # sequence_enabled: BoolProperty(default=False, options={'HIDDEN', }, )
    # sequence_frame_duration: IntProperty(name="Frames", default=1, min=1, description="", )
    # sequence_frame_start: IntProperty(name="Start Frame", default=1, description="", )
    # sequence_frame_offset: IntProperty(name="Offset", default=0, description="", )
    sequence_use_cyclic: BoolProperty(name="Cycle Forever", default=True,
                                      description="Cycle preloaded point clouds (ply_index = (current_frame % len(ply_files)) - 1)", )

    generate_source: EnumProperty(name="Source", items=[('VERTICES', "Vertices", "Use mesh vertices"),
                                                        ('SURFACE', "Surface", "Use triangulated mesh surface"),
                                                        ('PARTICLES', "Particle System", "Use active particle system"),
                                                        ], default='SURFACE', description="Points generation source", )
    generate_source_psys: EnumProperty(name="Particles", items=[('ALL', "All", "Use all particles"),
                                                                ('ALIVE', "Alive", "Use alive particles"),
                                                                ], default='ALIVE', description="Particles source", )
    generate_algorithm: EnumProperty(name="Algorithm", items=[('WEIGHTED_RANDOM_IN_TRIANGLE',
                                                               "Weighted Random In Triangle",
                                                               "Average triangle areas to approximate number of random points in each to get even distribution of points. If some very small polygons are left without points, increase number of samples. Mesh is triangulated before processing, on non-planar polygons, points will not be exactly on original polygon surface."),
                                                              ('POISSON_DISK_SAMPLING', "Poisson Disk Sampling",
                                                               "Warning: slow, very slow indeed.. Uses Weighted Random In Triangle algorithm to pregenerate samples with all its inconveniences."),
                                                              ], default='WEIGHTED_RANDOM_IN_TRIANGLE',
                                     description="Point generating algorithm", )
    generate_number_of_points: IntProperty(name="Approximate Number Of Points", default=100000, min=1,
                                           description="Number of points to generate, some algorithms may not generate exact number of points.", )
    generate_seed: IntProperty(name="Seed", default=0, min=0, description="Random number generator seed", )
    generate_colors: EnumProperty(name="Colors", items=[('CONSTANT', "Constant Color", "Use constant color value"),
                                                        ('VCOLS', "Vertex Colors", "Use active vertex colors"),
                                                        ('UVTEX', "UV Texture",
                                                         "Generate colors from active image texture node in active material using active UV layout"),
                                                        ('GROUP_MONO', "Vertex Group Monochromatic",
                                                         "Use active vertex group, result will be shades of grey"),
                                                        ('GROUP_COLOR', "Vertex Group Colorized",
                                                         "Use active vertex group, result will be colored from red (1.0) to blue (0.0) like in weight paint viewport"),
                                                        ], default='CONSTANT',
                                  description="Color source for generated point cloud", )
    generate_constant_color: FloatVectorProperty(name="Color", description="Constant color", default=(0.7, 0.7, 0.7,),
                                                 min=0, max=1, subtype='COLOR', size=3, )
    generate_exact_number_of_points: BoolProperty(name="Exact Number of Samples", default=False,
                                                  description="Generate exact number of points, if selected algorithm result is less points, more points will be calculated on random polygons at the end, if result is more points, points will be shuffled and sliced to match exact value", )
    generate_minimal_distance: FloatProperty(name="Minimal Distance", default=0.1, precision=3, subtype='DISTANCE',
                                             description="Poisson Disk minimal distance between points, the smaller value, the slower calculation", )
    generate_sampling_exponent: IntProperty(name="Sampling Exponent", default=5, min=1,
                                            description="Poisson Disk presampling exponent, lower values are faster but less even, higher values are slower exponentially", )

    # debug_shader: EnumProperty(name="Debug Shader", items=[('NONE', "None", ""),
    #                                                        ('DEPTH', "Depth", ""),
    #                                                        ('NORMAL', "Normal", ""),
    #                                                        ('POSITION', "Position", ""),
    #                                                        ], default='NONE', description="", )
    override_default_shader: BoolProperty(default=False, options={'HIDDEN', }, )

    # def _update_override_default_shader(self, context, ):
    #     if(self.dev_depth_enabled or self.dev_normal_colors_enabled or self.dev_position_colors_enabled):
    #         self.override_default_shader = True
    #     else:
    #         self.override_default_shader = False

    def _update_dev_depth(self, context, ):
        if (self.dev_depth_enabled):
            # FIXME: this is really getting ridiculous
            self.dev_normal_colors_enabled = False
            self.dev_position_colors_enabled = False

            self.dev_minimal_shader_enabled = False
            self.dev_minimal_shader_variable_size_enabled = False
            self.dev_minimal_shader_variable_size_and_depth_enabled = False
            self.dev_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_no_depth_enabled = False
            self.dev_phong_shader_enabled = False
            self.clip_shader_enabled = False
            self.billboard_phong_enabled = False
            self.skip_point_shader_enabled = False

            self.override_default_shader = True
        else:
            self.override_default_shader = False

    def _update_dev_normal(self, context, ):
        if (self.dev_normal_colors_enabled):
            # FIXME: this is really getting ridiculous
            self.dev_depth_enabled = False
            self.dev_position_colors_enabled = False

            self.dev_minimal_shader_enabled = False
            self.dev_minimal_shader_variable_size_enabled = False
            self.dev_minimal_shader_variable_size_and_depth_enabled = False
            self.dev_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_no_depth_enabled = False
            self.dev_phong_shader_enabled = False
            self.clip_shader_enabled = False
            self.billboard_phong_enabled = False
            self.skip_point_shader_enabled = False

            self.override_default_shader = True
        else:
            self.override_default_shader = False

    def _update_dev_position(self, context, ):
        if (self.dev_position_colors_enabled):
            # FIXME: this is really getting ridiculous
            self.dev_depth_enabled = False
            self.dev_normal_colors_enabled = False

            self.dev_minimal_shader_enabled = False
            self.dev_minimal_shader_variable_size_enabled = False
            self.dev_minimal_shader_variable_size_and_depth_enabled = False
            self.dev_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_no_depth_enabled = False
            self.dev_phong_shader_enabled = False
            self.clip_shader_enabled = False
            self.billboard_phong_enabled = False
            self.skip_point_shader_enabled = False

            self.override_default_shader = True
        else:
            self.override_default_shader = False

    # dev_depth_enabled: BoolProperty(name="Depth", default=False, description="", update=_update_override_default_shader, )
    dev_depth_enabled: BoolProperty(name="Depth", default=False, description="Enable depth debug shader",
                                    update=_update_dev_depth, )
    # dev_depth_edit: BoolProperty(name="Edit", description="Edit depth shader properties", default=False, )
    dev_depth_brightness: FloatProperty(name="Brightness", description="Depth shader color brightness", default=0.0,
                                        min=-10.0, max=10.0, )
    dev_depth_contrast: FloatProperty(name="Contrast", description="Depth shader color contrast", default=1.0,
                                      min=-10.0, max=10.0, )
    dev_depth_false_colors: BoolProperty(name="False Colors", default=False,
                                         description="Display depth shader in false colors", )
    dev_depth_color_a: FloatVectorProperty(name="Color A", description="Depth shader false colors front color",
                                           default=(0.0, 1.0, 0.0,), min=0, max=1, subtype='COLOR', size=3, )
    dev_depth_color_b: FloatVectorProperty(name="Color B", description="Depth shader false colors back color",
                                           default=(0.0, 0.0, 1.0,), min=0, max=1, subtype='COLOR', size=3, )
    # dev_normal_colors_enabled: BoolProperty(name="Normal", default=False, description="", update=_update_override_default_shader, )
    dev_normal_colors_enabled: BoolProperty(name="Normal", default=False, description="Enable normal debug shader",
                                            update=_update_dev_normal, )
    # dev_position_colors_enabled: BoolProperty(name="Position", default=False, description="", update=_update_override_default_shader, )
    dev_position_colors_enabled: BoolProperty(name="Position", default=False,
                                              description="Enable position debug shader", update=_update_dev_position, )

    # NOTE: icon for bounding box 'SHADING_BBOX' ?
    dev_bbox_enabled: BoolProperty(name="Bounding Box", default=True, description="", )
    dev_bbox_color: FloatVectorProperty(name="Color", description="", default=(0.7, 0.7, 0.7), min=0, max=1,
                                        subtype='COLOR', size=3, )
    dev_bbox_size: FloatProperty(name="Size", description="", default=0.4, min=0.1, max=0.9, subtype='FACTOR', )
    dev_bbox_alpha: FloatProperty(name="Alpha", description="", default=0.7, min=0.0, max=1.0, subtype='FACTOR', )

    def _dev_sel_color_update(self, context, ):
        bpy.context.preferences.addons[__name__].preferences.selection_color = self.dev_selection_shader_color

    dev_selection_shader_display: BoolProperty(name="Selection", default=False, description="", )
    dev_selection_shader_color: FloatVectorProperty(name="Color", description="", default=(1.0, 0.0, 0.0, 0.5), min=0,
                                                    max=1, subtype='COLOR', size=4, update=_dev_sel_color_update, )

    def _update_color_adjustment(self, context, ):
        if (self.color_adjustment_shader_enabled):
            # FIXME: this is really getting ridiculous
            self.dev_depth_enabled = False
            self.dev_normal_colors_enabled = False
            self.dev_position_colors_enabled = False

            self.dev_minimal_shader_enabled = False
            self.dev_minimal_shader_variable_size_enabled = False
            self.dev_minimal_shader_variable_size_and_depth_enabled = False
            self.dev_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_no_depth_enabled = False
            self.dev_phong_shader_enabled = False
            self.clip_shader_enabled = False
            self.billboard_phong_enabled = False
            self.skip_point_shader_enabled = False

            self.illumination = False
            self.override_default_shader = True
        else:
            self.override_default_shader = False

    color_adjustment_shader_enabled: BoolProperty(name="Enabled", default=False,
                                                  description="Enable color adjustment shader, other shaders will be overrided until disabled",
                                                  update=_update_color_adjustment, )
    color_adjustment_shader_exposure: FloatProperty(name="Exposure",
                                                    description="formula: color = color * (2 ** value)", default=0.0,
                                                    min=-5.0, max=5.0, )
    color_adjustment_shader_gamma: FloatProperty(name="Gamma", description="formula: color = color ** (1 / value)",
                                                 default=1.0, min=0.01, max=9.99, )
    color_adjustment_shader_brightness: FloatProperty(name="Brightness",
                                                      description="formula: color = (color - 0.5) * contrast + 0.5 + brightness",
                                                      default=0.0, min=-5.0, max=5.0, )
    color_adjustment_shader_contrast: FloatProperty(name="Contrast",
                                                    description="formula: color = (color - 0.5) * contrast + 0.5 + brightness",
                                                    default=1.0, min=0.0, max=10.0, )
    color_adjustment_shader_hue: FloatProperty(name="Hue",
                                               description="formula: color.h = (color.h + (value % 1.0)) % 1.0",
                                               default=0.0, min=0.0, max=1.0, )
    color_adjustment_shader_saturation: FloatProperty(name="Saturation", description="formula: color.s += value",
                                                      default=0.0, min=-1.0, max=1.0, )
    color_adjustment_shader_value: FloatProperty(name="Value", description="formula: color.v += value", default=0.0,
                                                 min=-1.0, max=1.0, )
    color_adjustment_shader_invert: BoolProperty(name="Invert", description="formula: color = 1.0 - color",
                                                 default=False, )

    def _update_minimal_shader(self, context, ):
        if (self.dev_minimal_shader_enabled):
            # FIXME: this is really getting ridiculous
            self.illumination = False
            self.dev_depth_enabled = False
            self.dev_normal_colors_enabled = False
            self.dev_position_colors_enabled = False
            self.color_adjustment_shader_enabled = False
            self.dev_minimal_shader_variable_size_enabled = False
            self.dev_minimal_shader_variable_size_and_depth_enabled = False
            self.dev_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_no_depth_enabled = False
            self.dev_phong_shader_enabled = False
            self.clip_shader_enabled = False
            self.billboard_phong_enabled = False
            self.skip_point_shader_enabled = False

            self.override_default_shader = True
        else:
            self.override_default_shader = False

    def _update_minimal_shader_variable_size(self, context, ):
        if (self.dev_minimal_shader_variable_size_enabled):
            # FIXME: this is really getting ridiculous
            self.illumination = False
            self.dev_depth_enabled = False
            self.dev_normal_colors_enabled = False
            self.dev_position_colors_enabled = False
            self.color_adjustment_shader_enabled = False
            self.dev_minimal_shader_enabled = False
            self.dev_minimal_shader_variable_size_and_depth_enabled = False
            self.dev_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_no_depth_enabled = False
            self.dev_phong_shader_enabled = False
            self.clip_shader_enabled = False
            self.billboard_phong_enabled = False
            self.skip_point_shader_enabled = False

            self.override_default_shader = True
        else:
            self.override_default_shader = False

    dev_minimal_shader_enabled: BoolProperty(name="Enabled", default=False, description="Enable minimal shader",
                                             update=_update_minimal_shader, )
    dev_minimal_shader_variable_size_enabled: BoolProperty(name="Enabled", default=False,
                                                           description="Enable minimal shader with variable size",
                                                           update=_update_minimal_shader_variable_size, )

    def _update_minimal_shader_variable_size_with_depth(self, context, ):
        if (self.dev_minimal_shader_variable_size_and_depth_enabled):
            # FIXME: this is really getting ridiculous
            self.illumination = False
            self.dev_depth_enabled = False
            self.dev_normal_colors_enabled = False
            self.dev_position_colors_enabled = False
            self.color_adjustment_shader_enabled = False
            self.dev_minimal_shader_enabled = False
            self.dev_minimal_shader_variable_size_enabled = False
            self.dev_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_no_depth_enabled = False
            self.dev_phong_shader_enabled = False
            self.clip_shader_enabled = False
            self.billboard_phong_enabled = False
            self.skip_point_shader_enabled = False

            self.override_default_shader = True
        else:
            self.override_default_shader = False

    dev_minimal_shader_variable_size_and_depth_enabled: BoolProperty(name="Enabled", default=False,
                                                                     description="Enable minimal shader with variable size with depth",
                                                                     update=_update_minimal_shader_variable_size_with_depth, )
    dev_minimal_shader_variable_size_and_depth_brightness: FloatProperty(name="Brightness", default=0.25, min=-10.0,
                                                                         max=10.0,
                                                                         description="Depth shader color brightness", )
    dev_minimal_shader_variable_size_and_depth_contrast: FloatProperty(name="Contrast", default=0.5, min=-10.0,
                                                                       max=10.0,
                                                                       description="Depth shader color contrast", )
    dev_minimal_shader_variable_size_and_depth_blend: FloatProperty(name="Blend", default=0.75, min=0.0, max=1.0,
                                                                    subtype='FACTOR',
                                                                    description="Depth shader blending with original colors", )

    def _update_dev_billboard_point_cloud_enabled(self, context, ):
        if (self.dev_billboard_point_cloud_enabled):
            # FIXME: this is really getting ridiculous
            self.illumination = False
            self.dev_depth_enabled = False
            self.dev_normal_colors_enabled = False
            self.dev_position_colors_enabled = False
            self.color_adjustment_shader_enabled = False
            self.dev_minimal_shader_enabled = False
            self.dev_minimal_shader_variable_size_enabled = False
            self.dev_minimal_shader_variable_size_and_depth_enabled = False
            self.dev_rich_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_no_depth_enabled = False
            self.dev_phong_shader_enabled = False
            self.clip_shader_enabled = False
            self.billboard_phong_enabled = False
            self.skip_point_shader_enabled = False

            self.override_default_shader = True
        else:
            self.override_default_shader = False

    dev_billboard_point_cloud_enabled: BoolProperty(name="Enabled", default=False,
                                                    description="Enable Billboard Shader",
                                                    update=_update_dev_billboard_point_cloud_enabled, )
    dev_billboard_point_cloud_size: FloatProperty(name="Size", default=0.002, min=0.0001, max=0.2, description="",
                                                  precision=6, )

    def _update_dev_rich_billboard_point_cloud_enabled(self, context):
        if (self.dev_rich_billboard_point_cloud_enabled):
            # FIXME: this is really getting ridiculous
            self.illumination = False
            self.dev_depth_enabled = False
            self.dev_normal_colors_enabled = False
            self.dev_position_colors_enabled = False
            self.color_adjustment_shader_enabled = False
            self.dev_minimal_shader_enabled = False
            self.dev_minimal_shader_variable_size_enabled = False
            self.dev_minimal_shader_variable_size_and_depth_enabled = False
            self.dev_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_no_depth_enabled = False
            self.dev_phong_shader_enabled = False
            self.clip_shader_enabled = False
            self.billboard_phong_enabled = False
            self.skip_point_shader_enabled = False

            self.override_default_shader = True
        else:
            self.override_default_shader = False

    dev_rich_billboard_point_cloud_enabled: BoolProperty(name="Enabled", default=False,
                                                         description="Enable Rich Billboard Shader",
                                                         update=_update_dev_rich_billboard_point_cloud_enabled, )
    dev_rich_billboard_point_cloud_size: FloatProperty(name="Size", default=0.01, min=0.0001, max=1.0, description="",
                                                       precision=6, )
    dev_rich_billboard_depth_brightness: FloatProperty(name="Brightness", default=0.25, min=-10.0, max=10.0,
                                                       description="Depth shader color brightness", )
    dev_rich_billboard_depth_contrast: FloatProperty(name="Contrast", default=0.5, min=-10.0, max=10.0,
                                                     description="Depth shader color contrast", )
    dev_rich_billboard_depth_blend: FloatProperty(name="Blend", default=0.75, min=0.0, max=1.0, subtype='FACTOR',
                                                  description="Depth shader blending with original colors", )

    def _update_dev_rich_billboard_point_cloud_no_depth_enabled(self, context):
        if (self.dev_rich_billboard_point_cloud_no_depth_enabled):
            # FIXME: this is really getting ridiculous
            self.illumination = False
            self.dev_depth_enabled = False
            self.dev_normal_colors_enabled = False
            self.dev_position_colors_enabled = False
            self.color_adjustment_shader_enabled = False
            self.dev_minimal_shader_enabled = False
            self.dev_minimal_shader_variable_size_enabled = False
            self.dev_minimal_shader_variable_size_and_depth_enabled = False
            self.dev_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_enabled = False
            self.dev_phong_shader_enabled = False
            self.clip_shader_enabled = False
            self.billboard_phong_enabled = False
            self.skip_point_shader_enabled = False

            self.override_default_shader = True
        else:
            self.override_default_shader = False

    dev_rich_billboard_point_cloud_no_depth_enabled: BoolProperty(name="Enabled", default=False,
                                                                  description="Enable Rich Billboard Shader Without Depth",
                                                                  update=_update_dev_rich_billboard_point_cloud_no_depth_enabled, )

    def _update_dev_phong_shader_enabled(self, context):
        if (self.dev_phong_shader_enabled):
            # FIXME: this is really getting ridiculous
            self.illumination = False
            self.dev_depth_enabled = False
            self.dev_normal_colors_enabled = False
            self.dev_position_colors_enabled = False
            self.color_adjustment_shader_enabled = False
            self.dev_minimal_shader_enabled = False
            self.dev_minimal_shader_variable_size_enabled = False
            self.dev_minimal_shader_variable_size_and_depth_enabled = False
            self.dev_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_no_depth_enabled = False
            self.clip_shader_enabled = False
            self.billboard_phong_enabled = False
            self.skip_point_shader_enabled = False

            self.override_default_shader = True
        else:
            self.override_default_shader = False

    dev_phong_shader_enabled: BoolProperty(name="Enabled", default=False, description="",
                                           update=_update_dev_phong_shader_enabled, )
    dev_phong_shader_ambient_strength: FloatProperty(name="ambient_strength", default=0.5, min=0.0, max=1.0,
                                                     description="", )
    dev_phong_shader_specular_strength: FloatProperty(name="specular_strength", default=0.5, min=0.0, max=1.0,
                                                      description="", )
    dev_phong_shader_specular_exponent: FloatProperty(name="specular_exponent", default=8.0, min=1.0, max=512.0,
                                                      description="", )

    debug_panel_show_properties: BoolProperty(default=False, options={'HIDDEN', }, )
    debug_panel_show_manager: BoolProperty(default=False, options={'HIDDEN', }, )
    debug_panel_show_sequence: BoolProperty(default=False, options={'HIDDEN', }, )
    debug_panel_show_cache_items: BoolProperty(default=False, options={'HIDDEN', }, )

    # store info how long was last draw call, ie get points from cache, join, draw
    pcviv_debug_draw: StringProperty(default="", )
    pcviv_debug_panel_show_info: BoolProperty(default=False, options={'HIDDEN', }, )
    # have to provide prop for indexing, not needed for anything in this case
    pcviv_material_list_active_index: IntProperty(name="Index", default=0, description="", options={'HIDDEN', }, )

    # testing / development stuff
    def _dev_transform_normals_target_object_poll(self, o, ):
        if (o and o.type in ('MESH',)):
            return True
        return False

    dev_transform_normals_target_object: PointerProperty(type=bpy.types.Object, name="Object", description="",
                                                         poll=_dev_transform_normals_target_object_poll, )

    # dev
    def _clip_shader_enabled(self, context):
        if (self.clip_shader_enabled):
            # FIXME: this is really getting ridiculous
            self.illumination = False
            self.dev_depth_enabled = False
            self.dev_normal_colors_enabled = False
            self.dev_position_colors_enabled = False
            self.color_adjustment_shader_enabled = False
            self.dev_minimal_shader_enabled = False
            self.dev_minimal_shader_variable_size_enabled = False
            self.dev_minimal_shader_variable_size_and_depth_enabled = False
            self.dev_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_no_depth_enabled = False
            self.dev_phong_shader_enabled = False
            self.billboard_phong_enabled = False
            self.skip_point_shader_enabled = False

            self.override_default_shader = True
        else:
            self.override_default_shader = False

    clip_shader_enabled: BoolProperty(name="Enabled", default=False, description="", update=_clip_shader_enabled, )
    clip_plane0_enabled: BoolProperty(name="Enabled", default=False, description="", )
    clip_plane1_enabled: BoolProperty(name="Enabled", default=False, description="", )
    clip_plane2_enabled: BoolProperty(name="Enabled", default=False, description="", )
    clip_plane3_enabled: BoolProperty(name="Enabled", default=False, description="", )
    clip_plane4_enabled: BoolProperty(name="Enabled", default=False, description="", )
    clip_plane5_enabled: BoolProperty(name="Enabled", default=False, description="", )
    clip_plane0: FloatVectorProperty(name="Plane 0", default=(0.0, 0.0, 0.0, 0.0), subtype='NONE', size=4,
                                     description="", )
    clip_plane1: FloatVectorProperty(name="Plane 1", default=(0.0, 0.0, 0.0, 0.0), subtype='NONE', size=4,
                                     description="", )
    clip_plane2: FloatVectorProperty(name="Plane 2", default=(0.0, 0.0, 0.0, 0.0), subtype='NONE', size=4,
                                     description="", )
    clip_plane3: FloatVectorProperty(name="Plane 3", default=(0.0, 0.0, 0.0, 0.0), subtype='NONE', size=4,
                                     description="", )
    clip_plane4: FloatVectorProperty(name="Plane 4", default=(0.0, 0.0, 0.0, 0.0), subtype='NONE', size=4,
                                     description="", )
    clip_plane5: FloatVectorProperty(name="Plane 5", default=(0.0, 0.0, 0.0, 0.0), subtype='NONE', size=4,
                                     description="", )

    def _clip_planes_from_bbox_object_poll(self, o, ):
        if (o and o.type in ('MESH',)):
            return True
        return False

    clip_planes_from_bbox_object: PointerProperty(type=bpy.types.Object, name="Object", description="",
                                                  poll=_clip_planes_from_bbox_object_poll, )

    def _billboard_phong_enabled(self, context):
        if (self.billboard_phong_enabled):
            # FIXME: this is really getting ridiculous
            self.illumination = False
            self.dev_depth_enabled = False
            self.dev_normal_colors_enabled = False
            self.dev_position_colors_enabled = False
            self.color_adjustment_shader_enabled = False
            self.dev_minimal_shader_enabled = False
            self.dev_minimal_shader_variable_size_enabled = False
            self.dev_minimal_shader_variable_size_and_depth_enabled = False
            self.dev_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_no_depth_enabled = False
            self.dev_phong_shader_enabled = False
            self.clip_shader_enabled = False
            self.skip_point_shader_enabled = False

            self.override_default_shader = True
        else:
            self.override_default_shader = False

    billboard_phong_enabled: BoolProperty(name="Enabled", default=False, description="",
                                          update=_billboard_phong_enabled, )
    billboard_phong_circles: BoolProperty(name="Circles (slower)", default=False, description="", )
    billboard_phong_size: FloatProperty(name="Size", default=0.002, min=0.0001, max=0.2, description="", precision=6, )
    billboard_phong_ambient_strength: FloatProperty(name="Ambient", default=0.5, min=0.0, max=1.0, description="", )
    billboard_phong_specular_strength: FloatProperty(name="Specular", default=0.5, min=0.0, max=1.0, description="", )
    billboard_phong_specular_exponent: FloatProperty(name="Hardness", default=8.0, min=1.0, max=512.0, description="", )

    def _skip_point_shader_enabled(self, context):
        if (self.skip_point_shader_enabled):
            # FIXME: this is really getting ridiculous
            self.illumination = False
            self.dev_depth_enabled = False
            self.dev_normal_colors_enabled = False
            self.dev_position_colors_enabled = False
            self.color_adjustment_shader_enabled = False
            self.dev_minimal_shader_enabled = False
            self.dev_minimal_shader_variable_size_enabled = False
            self.dev_minimal_shader_variable_size_and_depth_enabled = False
            self.dev_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_no_depth_enabled = False
            self.dev_phong_shader_enabled = False
            self.clip_shader_enabled = False
            self.billboard_phong_enabled = False

            self.override_default_shader = True
        else:
            self.override_default_shader = False

    skip_point_shader_enabled: BoolProperty(name="Enabled", default=False, description="",
                                            update=_skip_point_shader_enabled, )
    # skip_point_percentage: FloatProperty(name="Skip Percentage", default=100.0, min=0.0, max=100.0, precision=0, subtype='PERCENTAGE', description="", )
    skip_point_percentage: FloatProperty(name="Skip Percentage", default=100.0, min=0.0, max=100.0, precision=3,
                                         description="", )

    @classmethod
    def register(cls):
        bpy.types.Object.point_cloud_visualizer = PointerProperty(type=cls)

    @classmethod
    def unregister(cls):
        del bpy.types.Object.point_cloud_visualizer

class PCM_OT_load():

    def __init__(self):
        print("Point cloud visualizer is initiated")

    def __del__(self):
        print("Point cloud visualizer is terminated")

    def redraw(self, context):
        pcv = context.object.point_cloud_visualizer
        if (pcv.uuid != ""):
            if (pcv.uuid in PCVManager.cache):
                PCVManager.cache[pcv.uuid]['kill'] = True
                PCVManager.gc()
        c = PCVManager.cache[pcv.uuid]
        c['draw'] = True

    def loadply(self, context):
        obj = data_Storage.point_Cloud_Axis
        pcv = obj.point_cloud_visualizer

        if (len(pcv.uuid) != 0):
            if (len(PCVManager.cache) != 0):
                if (data_Storage.FLAG_RESTRICT_MAKE_POINT_CLOUD == True):
                    print("Cache was initialized.")
                    bpy.ops.screen.animation_play()
                PCVManager.cache[pcv.uuid]['kill'] = True
                PCVManager.gc()

        if (data_Storage.FLAG_RESTRICT_MAKE_POINT_CLOUD == False):
            ok = PCVManager.load_ply_to_cache(self, context)
            c = PCVManager.cache[pcv.uuid]
            if (ok):
                c['draw'] = True
        else:
            data_Storage.FLAG_RESTRICT_MAKE_POINT_CLOUD = False

        data_Storage.FLAG_PLY_LOAD_FINISHED = True

@persistent
def watcher(scene):
    PCVManager.deinit()