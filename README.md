# Linear_Regression
Using Linear Regression to predict california housing

import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from IPython.display import clear_output, Image, display, HTML
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
import numpy as np

###### Do not modify here ###### 
def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add() 
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = "<stripped %d bytes>"%size
    return strip_def

def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = graph_def
    #strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))
###### Do not modify  here ######

###### Implement Data Preprocess here ######
housing = fetch_california_housing()
print("Shape of dataset:", housing.data.shape)
print("Shape of label:", housing.target.shape)

#X_train, X_test, y_train, y_test = train_test_split(housing.data, housing.target, test_size=0.1, random_state=0) 
#90% for training set,10% for testing set (random)

housing.target = housing.target.reshape(-1,1)
#reshape the y data to a column

X_train = housing.data[:round(housing.data.shape[0]*0.9),:]
X_test = housing.data[round(housing.data.shape[0]*0.9):,:]
y_train = housing.target[:round(housing.target.shape[0]*0.9),:]
y_test = housing.target[round(housing.target.shape[0]*0.9):,:]
#split data for head 90% to train ,tail 10% to test (not random)

###### Bonus ######
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)
#Standardization
###### Bonus ######

X_train_with_bias = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_test_with_bias = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
# add bias to X data

X = tf.constant(X_train_with_bias, dtype=tf.float32, name="X")
y = tf.constant(y_train.reshape(-1, 1), dtype=tf.float32, name="y")
XT = tf.transpose(X)
# Set up constant for the normal equation

weight = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)
#use normal equation to find weight
###### Implement Data Preprocess here ######


###### Start TF session ######
with tf.Session() as sess:
    weight_value = sess.run(weight)
    show_graph(tf.get_default_graph().as_graph_def())
###### Start TF session ######

y_hat = np.dot(X_test_with_bias, weight_value)
#prediction

y_test = y_test.reshape(-1, 1)
error = np.abs((y_test-y_hat)/y_test)
average_error = np.mean(error)
print(average_error)
#Calculate error rate
