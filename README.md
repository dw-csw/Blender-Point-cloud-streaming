Point cloud STREAMING v1.0
=================
## powered by point cloud visualizer   
### https://github.com/uhlik/bpy


![pcs](https://user-images.githubusercontent.com/63433646/132840803-f255335c-bd2a-411e-95d8-cc95f6fa8e91.gif)

***
***

## Addons for blender 2.83
  * Point cloud visualizer runs on blender version 2.80 or higher (less than 2.9).

## Required module
  * Point cloud visualizer (addon)
  * numpy
  * pyrealsense2

## Available camera
  * Realsense D435
  * Realsense L515
  * Realsense etc... (When you modify script related to resolution.)

## How to set the execution environment.    
  1. Clone Blender-Point-cloud-streaming using by git   
  ![1](https://user-images.githubusercontent.com/63433646/132858087-3611e73d-80db-467b-ab0a-9044e9a1dc81.png)

  2. Clone point cloud visualizer using by git   
![1](https://user-images.githubusercontent.com/63433646/132845452-ab73a94f-a3b1-489e-bd33-53dd644235a0.png)

  3. Get "space_view3d_point_cloud_visualizer.py" using by Install Add-on   
![2](https://user-images.githubusercontent.com/63433646/132845801-086fcf8c-eefd-44e0-ad41-54fd2e68d782.png)

  4. Enable installed add-on and save preferences   
![3](https://user-images.githubusercontent.com/63433646/132845919-08bdd435-ebe6-4c54-8c03-6f84d400107f.png)

  5. ※ If you found this icon which is in red box, please do as below first.   
![5](https://user-images.githubusercontent.com/63433646/132848661-2d8686ab-bdd9-4e1a-8d71-05f7041ba408.png)   

![6](https://user-images.githubusercontent.com/63433646/132848690-2a596336-9e79-441c-895b-5570ac9205a2.png)   

  6. Check whether "Point cloud visualizer" is loaded on panel and  press "run script" button   
![4](https://user-images.githubusercontent.com/63433646/132846038-092bdf86-2edf-464d-a0a9-f5dfe9aafb31.png)

## Key event
  * C : Start to collect depth information
  * X : Stop collecting depth information.
  * S : Stop the script that is running (Cancel modal).

## License
  * GNU General Public License v2.0 or later
