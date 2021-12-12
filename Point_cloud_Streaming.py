import os
import sys

def pre_Processing_Operation():
    try:
        sys.path = []
        home = os.path.expanduser('~')
        _local = home + "/.local/lib/"
        _config = home + "/.config/blender/"
        blender = home + "/blender/"
        blf = os.listdir(blender)
        for version in blf:
            if (len(version) < 5 and len(version) != 3):
                blv = version
        scripts = blender + blv + "/scripts/"
        lib = blender + blv + "/python/lib/"
        pyv = os.listdir(lib)[0]
        python = lib + pyv
        route_Scripts = ["startup", "modules", "freestyle/modules", "addons/modules", "addons"]
        route_Python = ["", "/lib-dynload", "/site-packages"]
        route_Local = ["/site-packages", "/scripts/addons/modules", "/scripts/addons"]
        for pick in route_Scripts:
            sys.path.append(scripts + pick)
        for pick in route_Python:
            sys.path.append(python + pick)
        for cnt in range(0, 3):
            if cnt == 0:
                sys.path.append(_local + pyv + route_Local[cnt])
            else:
                sys.path.append(_config + blv + route_Local[cnt])
        cwd = home + "/Blender-Point-cloud-streaming"
        sys.path.append(cwd)
        print("\nOS : Ubuntu linux")
    except:
        sys.path = []
        default_Path = "C:/Program Files/Blender Foundation/"
        dir = os.listdir(default_Path)[0]
        blender = default_Path + dir
        sys.path.append(blender)
        bl = blender + '/'
        ver = os.listdir(bl)
        for ele in ver:
            if (3 < len(ele) < 5):
                ver_ = ele
        blender = bl + ver_
        scripts = blender + '/scripts/'
        py = blender + '/python'
        sys.path.append(py)
        python = blender + '/python/'
        sclist = ['addons', 'addons/modules', 'addons_contrib', 'freestyle/modules', 'modules', 'startup']
        pylist = ['DLLs', 'lib', 'lib/site-packages']
        for ele in sclist:
            dir = scripts + ele
            sys.path.append(dir)
        for ele in pylist:
            dir = python + ele
            sys.path.append(dir)
        home_path = os.path.expanduser('~')
        dir = "/AppData/Roaming/Blender Foundation/Blender/"
        verpath = home_path + dir
        ver = os.listdir(verpath)[0]
        dir = "/scripts/addons/modules"
        modules = verpath + ver + dir
        sys.path.append(modules)
        cwd = home_path + "/Blender-Point-cloud-streaming"
        sys.path.append(cwd)
        print("\nOS : Windows")

pre_Processing_Operation()

import bpy
import threading
from Depth_Camera_Operator import *
from Point_Cloud_Visualizer import data_Storage as ds, PCM_OT_load

class Modal_Handler(bpy.types.Operator):

    bl_idname = "wm.modal_handler"
    bl_label = "Modal Handler"
    update_rate = 1 / 30

    def __init__(self):
        ds.FLAG_PROGRAM_START = True
        self.start_Depth_Camera()
        self.pcml = PCM_OT_load()

    def execute(self, context):
        wm = context.window_manager
        self.register_handlers(context)
        self._timer = wm.event_timer_add(self.update_rate, window=context.window)
        wm.modal_handler_add(self)

        return {'RUNNING_MODAL'}

    def register_handlers(self, context):
        self.draw_event = context.window_manager.event_timer_add(0.1, window=context.window)

    def unregister_handlers(self, context):
        context.window_manager.event_timer_remove(self.draw_event)
        self.draw_event = None

    def modal(self, context, event):
        if event.type == 'ESC' and event.value == 'PRESS':
            cleanup_and_quit()
            return {'CANCELLED'}

        # TAG : Point cloud streaming
        if (ds.FLAG_GET_CAMERA_DATA == True):
            self.get_Point_Cloud_Data(context)

        # TAG : Delete point cloud
        if (ds.FLAG_DELETE_POINT_CLOUD == True):
            self.delete_Point_Cloud(context)

        if event.type == 'S' and event.value == 'PRESS':
            ds.FLAG_PROGRAM_START = False
            del self.pcml
            return {'CANCELLED'}

        if event.type == 'C' and event.value == 'PRESS':
            if (ds.FLAG_KEEP_COLLECT_DEPTH_INFO == False):
                bpy.ops.screen.animation_play()
                ds.FLAG_PLY_LOAD_FINISHED = True
                ds.FLAG_KEEP_COLLECT_DEPTH_INFO = True
            else:
                print("Camera is busy...")

        if event.type == 'X' and event.value == 'PRESS':
            try:
                ds.FLAG_PLY_LOAD_FINISHED = False
                ds.FLAG_KEEP_COLLECT_DEPTH_INFO = False
                ds.FLAG_RESTRICT_MAKE_POINT_CLOUD = False
                ds.points = None
                del ds.depth_Camera
            except:
                print("The process of collecting depth information has already been completed.")

        bpy.context.view_layer.update()
        return {'RUNNING_MODAL'}

    def get_Point_Cloud_Data(self, context):
        ds.FLAG_GET_CAMERA_DATA = False
        self.pcml.loadply(context)

    def delete_Point_Cloud(self, context):
        ds.FLAG_DELETE_POINT_CLOUD = False
        ds.FLAG_RESTRICT_MAKE_POINT_CLOUD = True
        self.pcml.loadply(context)

    def start_Depth_Camera(self):
        thread = threading.Thread(target=self.execute_Depth_Camera)
        thread.daemon = True
        thread.start()

    def execute_Depth_Camera(self):
        while(ds.FLAG_PROGRAM_START):
            if (ds.FLAG_KEEP_COLLECT_DEPTH_INFO == True):
                ds.depth_Camera = Depth_Camera()
                ds.depth_Camera.execute()
            else:
                pass
            time.sleep(1)

    def _cancel(self, context):
        wm = context.window_manager
        wm.event_timer_remove(self._timer)

def cleanup_and_quit():
    unregister()
    bpy.ops.wm.quit_blender()

def register():
    bpy.utils.register_class(Modal_Handler)

def unregister():
    bpy.utils.unregister_class(Modal_Handler)

def main():
    register()
    bpy.ops.wm.modal_handler()

if __name__ == "__main__":
    main()