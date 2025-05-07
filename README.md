# RLrobot
  This is the starting github to creating a robot RL model with simulation.
  This contains a program for person detection for future use for robot pathing which is optimized to run on a raspberry pi 5. The robot design was used by dan makes thing's open source bipedal robot. The link to the github is here. https://github.com/makerforgetech/modular-biped. (shown here)

![image](https://github.com/user-attachments/assets/f12c94f2-571d-4671-a7fc-07c76ee1eab2)



Here is an exmaple image of the robot in the simulator after conversion to xml.
![image](https://github.com/user-attachments/assets/0f54da3a-b8bf-485b-8076-a7000a85b5e7)



# Camera
The camera object detection files include the .ipynb file and which prunes and quantizes the weights. The other files are used to test the camera system on the Pi5. demo.py is a file that runs the camera program on the Pi5. 

![image](https://github.com/user-attachments/assets/7ff5ff20-bfd2-445f-ba8f-b7473ef91ac9)

Here is an example of the setup.

![image](https://github.com/user-attachments/assets/00929b8e-23a8-47d4-957e-9527c0803e2c)


And here is an example of the usage.

Here is the example output of lower size and faster inference speed.

![image](https://github.com/user-attachments/assets/c4403839-38af-4352-b891-b7d93f3e02ea)


Future work includes to finish the simulation xml file of the robot. This was done in MujoCo which might not be the best for the setup of the OnShape file. Another consideration is checking how long the pie could run on batteries. Currently the battieres that are planned to use are "4pcs 1￵8￵6￵5￵0 Rechargeable Batter￵y W￵i￵th 18650 Battery Charger,Universal Smart Charger for 3￵.7V L￵ithium ion Batteries LSXdetoro". future work also includes training the Rienforcement Learning algorithm. Since the robot is controlled by only 6 servos and a gyro / accelerometer this would be a simple equation but a much harder problem going from simulation to the real world. 
