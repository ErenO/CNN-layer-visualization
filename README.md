**Web app for filters visualisation**

If you choose the layer, then put the filter number, it will show you the image of maximization of the filter's activation, and save it to static/img/layers_name/
If you put no filter number, it will show you all the filters of the layer :

![alt text](https://github.com/ErenO/Landmark-detection/blob/master/img.png)

Here is some examples : 

![alt text](https://github.com/appchoose/layer-visualisation/blob/master/image/img1.png)

![alt text](https://github.com/appchoose/layer-visualisation/blob/master/image/img2.png)

All you have to do, is to replace the code of the function make_model() by your model architecture, in visualize.py,
put your model in the folder, and change the model's name in app.py.

To run flask, change the path to app.py in run.sh, execute the command 'source run.sh',
then 'flask run --port 5000 --host 0.0.0.0'.

Make sure to export the path before to app.py "export FLASK_APP='root/your_path'" 
