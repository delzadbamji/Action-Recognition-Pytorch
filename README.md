# Action-Recognition-Pytorch
This project uses the UFC-101 video dataset and a failure capturing dataset.

The video files are broken down into frames and the frames are used to determine the action class and probability.

Here, I have used 2 network models, C3D and R2+1D models.

####Here's a demo:

![punch demo](https://user-images.githubusercontent.com/20069712/119776036-6ab9db80-be92-11eb-9db1-e80f9b136003.gif)

####Some of the action frames with class and probabilities (for R2+1D and C3D) are:

 | ![punch](https://github.com/delzadbamji/Action-Recognition-Pytorch/blob/main/show_action_label_punching.png) | ![R2 cooking](https://github.com/delzadbamji/Action-Recognition-Pytorch/blob/main/snapshot_cooking_R2.png) |
 | ----------------------------------------| ---------------------------------|
| ![sports](https://github.com/delzadbamji/Action-Recognition-Pytorch/blob/main/snapshot_sports.png)| ![sports](https://github.com/delzadbamji/Action-Recognition-Pytorch/blob/main/snapshot_repairs_R2.png)|
| ![sports](https://github.com/delzadbamji/Action-Recognition-Pytorch/blob/main/snapshot_repairs.png) | ![cooking](https://github.com/delzadbamji/Action-Recognition-Pytorch/blob/main/snapshot_cooking.png) |
