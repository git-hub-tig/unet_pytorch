I deploy the UNet PyTorch model using Flask and expose a REST API for model inference, refer to
https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html
the image files will be sent via HTTP POST requests, and the server also accepts only POST requests.

I write the inference demo script, refer to
https://github.com/milesial/Pytorch-UNet

below are some steps and ideas I got:

1. start the webserver, and output the log to file flask_log.out! Run:
    FLASK_APP=run_inference_server.py flask run --host 0.0.0.0 > flask_log.out 2>&1 & 

2. use the requests library to send a POST request to the flask app
    python3 request_post.py

3. write the result into a file and show the image on the screen.

4. if want to perform inference on inputs with variable sizes, 
    just change the static variable scale to dynamic param,
    which can make image preprocessing accordingly.

5. I did not get more insight into the model structure in model.py, but to 
    optimize the model latency and memory size, I have some practice, 
    we can retrain the model using mixed precision, thus getting a better training process,
    or we can quantize the model by post-training or training aware ways to get less model size.

    optimizing the model latency have many ways, if we want to change the model network structure,
    maybe we can learn more from the other network style, like the ResNet, making some Shortcut in UNet
    maybe is a good try.

project dependency:
        flask
        PyTorch
        TorchVision
        NumPy
        matplotlib
        PIL
