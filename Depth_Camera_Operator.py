import time
import numpy as np
import pyrealsense2 as rs
from Point_Cloud_Visualizer import data_Storage as ds
from numpy.lib.recfunctions import unstructured_to_structured as uts

# TAG : Point cloud streaming
class Depth_Camera():

    def __init__(self):
        print("Depth camera initialization is started...")
        self.pipeline = rs.pipeline()
        self.pc = rs.pointcloud()
        self.config = rs.config()
        self.align = None
        self.align_to = None
        self.get_Aligned_Frame = True

        # TAG : Intel L515
        self.depth_Width = 1024
        self.depth_Height = 768
        self.color_Width = 1280
        self.color_Height = 720
        self.fps = 30

        # TAG : Intel D435
        # self.depth_Width = 640
        # self.depth_Height = 480
        # self.color_Width = 640
        # self.color_Height = 480
        # self.fps = 6

        depth_Resolution = self.depth_Width * self.depth_Height
        color_Resolution = self.color_Width * self.color_Height
        if (depth_Resolution == color_Resolution):
            self.get_Aligned_Frame = False

        context = rs.context()
        connect_device = None
        try:
            if context.devices[0].get_info(rs.camera_info.name).lower() != 'platform camera':
                connect_device = context.devices[0].get_info(rs.camera_info.serial_number)

            self.config.enable_device(connect_device)

            self.config.enable_stream(rs.stream.depth, self.depth_Width, self.depth_Height, rs.format.z16, self.fps)
            self.config.enable_stream(rs.stream.color, self.color_Width, self.color_Height, rs.format.bgr8, self.fps)
            print("Depth camera is ready")
        except:
            ds.FLAG_KEEP_COLLECT_DEPTH_INFO = False
            print("┌─ * * * ────────────────────────────┐")
            print("│ Fail to connect with depth camera. │")
            print("│ Check connection status of it.     │")
            print("└──────────────────────────── * * * ─┘")

    def __del__(self):
        print("┌──────────────────────────────────────┐")
        print('│ Collecting of depth info is stopped. │')
        print("└──────────────────────────────────────┘")

    def execute(self):
        print("┌─────────────────────────────────┐")
        print('│ Collecting depth information... │')
        print("└─────────────────────────────────┘")
        try:
            self.pipeline.start(self.config)
            print("  > Camera type : Intel L515")
        except:
            self.depth_Width = 640
            self.depth_Height = 480
            self.color_Width = 640
            self.color_Height = 480
            self.fps = 6

            self.config.enable_stream(rs.stream.depth, self.depth_Width, self.depth_Height, rs.format.z16, self.fps)
            self.config.enable_stream(rs.stream.color, self.color_Width, self.color_Height, rs.format.bgr8, self.fps)
            print("  > Camera type : Intel D435")
            try:
                self.pipeline.start(self.config)
            except:
                print("┌─ * * * ──────────────────────────────────────┐")
                print("│ There is no signal sended from depth camera. │")
                print("│ Check connection status of camera.           │")
                print("└────────────────────────────────────── * * * ─┘")
                ds.FLAG_KEEP_COLLECT_DEPTH_INFO = False
                del ds.depth_Camera
                return
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)

        while (ds.FLAG_KEEP_COLLECT_DEPTH_INFO == True):
            if (ds.FLAG_PLY_LOAD_FINISHED == True):
                frames = self.pipeline.wait_for_frames()

                if (self.get_Aligned_Frame):
                    # ============================================================================================ #
                    # NOTE : Use "aligned frames" when the resolution of Depth frame and color frame is different.
                    # ============================================================================================ #
                    aligned_frames = self.align.process(frames)
                    depth_frame = aligned_frames.get_depth_frame()
                    color_frame = aligned_frames.get_color_frame()
                    color_image = np.array(color_frame.get_data())
                    ################################################################################################

                else:
                    # ============================================================================================ #
                    # NOTE : Use "frame" when the resolution of Depth frame and color frame is same.
                    # ============================================================================================ #
                    depth_frame = frames.get_depth_frame()
                    color_frame = frames.get_color_frame()
                    color_image = np.array(color_frame.get_data())
                    ################################################################################################

                points = self.pc.calculate(depth_frame)

                vertices = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)

                color = color_image.reshape(-1, 3)

                point_Cloud_Np_Data = np.append(vertices, color, axis=1)
                custom_type = np.dtype(
                    [('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('blue', 'u1'), ('green', 'u1'), ('red', 'u1')])
                points = uts(point_Cloud_Np_Data, dtype=custom_type)

                ds.points = points
                ds.FLAG_GET_CAMERA_DATA = True
                ds.FLAG_PLY_LOAD_FINISHED = False
            else:
                time.sleep(0.1)

        self.pipeline.stop()
        print("Pipeline terminated successfully.")
        ds.FLAG_DELETE_POINT_CLOUD = True
# ==================================================================================================================== #