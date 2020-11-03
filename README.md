# IRIS Tello Demo

Part of Project Vanguard's Fall 2020 Semester Projects, IRIS focused on providing a scalable machine learning model for object detection, that would serve as a demo for the features implemented on the team's main line of drones.

### How it Works

The Tello will carry out code running on a host computer connected to the broadcasting wifi signal coming from the drone itself. Through this network, the aircraft will execute receiving commands using UDP.

<img src="./IRIS%20Tello%20Demo%200c5e455c65a8456da11ddbd4c57b1610/diagram.png" alt="alt text" width="600"/>

source→[https://nanonets.com/blog/content/images/2018/11/thelist.png](https://nanonets.com/blog/content/images/2018/11/thelist.png)

Using the Tello SDK User Guide, as soon as the ML model detects a specified object in the input Tello video stream, it will send the Tello a command. The model uses the following classes:

background, aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow,
diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor

The bottle object is being used to carry out the demo. Once the object is detected, the *flip right* flip r, command is sent to the drone.

send("flip r")

<img src="./IRIS%20Tello%20Demo%200c5e455c65a8456da11ddbd4c57b1610/Draw-a-Water-Bottle-Step-11.jpg" alt="alt text" width="200"/>

<img src="./IRIS%20Tello%20Demo%200c5e455c65a8456da11ddbd4c57b1610/415yjHOgXVL.jpg" alt="alt text" width="285"/>

### Downloading the Code (Using Visual Studio Code)

Click source control or Ctrl+Shift+P on Linux and Windows, Command+Shift+P on macOS, then hit clone and then paste the following https clone link:

[https://github.com/ProjectVanguard/TelloObjectDetection](https://github.com/ProjectVanguard/TelloObjectDetection)