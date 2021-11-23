# interview-mleng

This repository contains the starting code for the ML Engineer technical assignment. 

The candidate has two tasks:
 * Given a trained model, make a Rest API to serve it. 
 * Once the Rest API is made, profile and optimise the model latency and memory wise. 


For both tasks we expect a MVP. 
Once you've completed both tasks you can choose to showcase your proficiency by exploring (one of) them more deeply.

Code cleanliness will be taken into account. 

### Model specification
* The model is a simple UNet trained for face landmark detection.
* It takes as input 96x96 images centered on faces and outputs a 15x96x96 tensor containing heatmaps indicating the position of face landmarks.
* The model implementation is given in model.py. 
* The model was trained using the default env in the docker image: nvcr.io/nvidia/pytorch:21.03-py3
* The model was trained using images centered on the face. 
* You can load the model by running:

```
from model import UNet

net = UNet()
net.load_state_dict(
    torch.load("simple_unet.pth", map_location="cpu")
)
```

### Rest API specification
* The API should be able to handle multiple concurrent requests.
* You should provide python scripts to interact with your api and print the results.

* The api sends back the facial landmarks coordinates predicted on the image in the output order of the network.



### Profiling and Optimisation of the network
* Your goal is to profile the model, find performance bottlenecks and optimise them when possible. Please optimise the network as much as possible. 
* If you have some ideas that aren't doable in this interview setting, please list them and how you would implement them. If possible rank them in terms of priority and perceived performance boost. 



### Optional

If you are done with the two previous tasks and still have some time left before the end of the assignment, you can check the following questions: 

How would modify your API to:

* perform automatic batching of the incoming requests?
* perform inference on inputs with variable sizes?
* perform inference using multiple models ?

Now, let's place ourselves in the case where you need to implement an API to serve a deep learning pipeline containing multiple models A, B
* A is a face detector. 
* B is the model above. 

In your pipeline you must detect the faces using A and applying the facial landmarks detector B.
(Disclaimer we know there are many models that do both steps at once, but for the sake of the exercise we will consider the situation described above.)
