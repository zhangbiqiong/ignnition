# IGNNITION: Framework for fast prototyping of GNNs
#### D. Pujol Perich, J. Suárez-Varela, Miquel Ferriol, P. Barlet-Ros, A. Cabellos-Aparicio.
 
## Abstract
Recent years have witnessed the vast potential of Graph Neural Networks (GNN) to be applied to a plethora of problems where data is structured as graphs (e.g., computer networks, chemistry, physics). Typically, each new application requires custom GNN models adapted to the problem environment. In this context, implementing a GNN model is a cumbersome task that currently requires Machine Learning (ML) experts with high skills in neural network programming. For instance, to translate the model to complex tensor-wise operations in languages such as TensorFlow or PyTorch. We claim that, in order to approach GNNs to real-world applications it is essential to count on tools that abstract users from the complexity behind the implementation of such models. In this paper we propose IGNNITION, a novel framework to fast prototype complex GNN models. IGNNITION targets users with little to no background on neural network programming, while still providing great flexibility to implement a broad variety of GNN architectures. Our framework enables to define a GNN model as an intuitive human-readable description of the objects present in graphs and the type of relations they may have, and it automatically generates an efficient TensorFlow implementation of the model. In order to assist users, it provides advanced debugging mechanisms to easily identify and correct possible errors in GNN models. Our experimental results show that GNN models generated by our framework do not incur performance loss compared to native implementations in TensorFlow.
 
<!-- Add BibTex citation to paper -->
## Quick Start
### Requirements
First of all, you can prepare the Python environment for the demo executing the following command:

```
pip install -r requirements.txt
```

This command installs all the dependencies needed to execute IGNNITION.

### Reproducing an example
In order to start using IGNITTION, we provide two use-cases so that the user can start familiarizing himself with the workflow of this framework. The first use-case is RouteNet and the second is a much more complicated Message Passing model which uses additionaly the queue size information. Both these examples can be found in ['examples' directory](examples), each in its respective directory. Let us now breafly describe the content of these two use-cases:

#### Model_description
We provide the model_description.json file to generate the model.

### Dataset
A dataset.json file is incorporated, however this is just a subset of the real datatset. For this reason the user should follow several steps to produce the complete one.

In order to obtain the full datasets to properly replicate these experiments, it is first necessary to download some datasets from the [repository](https://github.com/knowledgedefinednetworking/NetworkModelingDatasets/tree/master/datasets_v0). In it you can find three datasets with samples simulated with a custom packed-level simulator in three different topologies: NSFNET(2GB), GEANT(6GB) and synthetic 50 nodes (28.7 GB) topologies. To download an decompress any of the three datasets you can use the followin commands:
```
wget "http://knowledgedefinednetworking.org/data/datasets_v0/nsfnet.tar.gz"
wget "http://knowledgedefinednetworking.org/data/datasets_v0/geant2.tar.gz"
wget "http://knowledgedefinednetworking.org/data/datasets_v0/synth50.tar.gz"
tar -xvzf nsfnet.tar.gz 
tar -xvzf geant2.tar.gz 
tar -xvzf synth50.tar.gz
```

Once any of them has been downloaded, we provide a script called migration.py in the same directory which is the one that will automatically migrate the raw dataset into the json format required by the framework. This files can be found be executed as follows:
```
./migrate_to_JSON/migration.sh  <dataset_path> <output_path>
```

#### Main.py
In the directory, the main.py file is also provided. It is IMPORTANT to note that the content of this file should be copied to the actual main.py of the framework, which can be found in ['code' directory](code).

#### Set the paths
At this point, the user must indicate the framework the paths for all these previously mentioned files. For this, modify the train_options.ini file in ['code' directory](code) with the paths of the specific example.

#### Execute the framework
Finally, execute the framework by placing the scope in the ['code' directory](code) and executing the following command line
```
python3 main.py
```

With this, a debug_model directory should be generated and the a trained model should be produced.


## 'How to' guide


### Designing the model's architecture
The first step is to define the model's architecture. This is done by creating a model_description.json file where the several sections of the model's design should be specified. The two examples provided contain its corresponding file, which should be a good starting point as most likely only a few modifications of the files are required.

### Migrating the dataset
The second step is to migrate the original datset that is intended to be used to JSON format, containing the key words used in the model description file. Again, the migration.py files corresponding to both the use-cases we provide should be good starting point, as to fully understand the final structure that any dataset should have to be a valid input to IGNNITION.

### Definining the main.py file
Once these has been done, we must fill appropriately the main.py file. In this file we must create the model and indicate the functionality that we want to use, which can be:

1) Create the model
```
model = framework.create_model()
```
2) Train and evaluate
```
framework.train_and_evaluate(model)
```

3) Generate debug model
```
framework.debug(model)
```
4) Make predictions
```
framework.predict(model)
```

Furthermore, both the normalization and denormalization functions should be defined in case the model_description.json uses any. In this case, it is important that the names used in the model_description.json match with the name of the function in the main.py.

### Indicating the paths
Finally, the last step is to indicate the framework the paths were all the  previous data can be found. Thus, the user should modify the train_options.ini found in ['code' directory](code) with its corresponding paths and the appropriate training options to be use.

### Generating the model
At this point, we are in position of running the framework, for which the user must use the following command line:

```
python3 main.py
```

### Visualization of the debug model

In case the user chose to generate the debug model of the model description indicated, a new directory debug_model will be created ['code' directory] which contains the event files that can be visualized using the Tensorboard tool. For this, type the following command line:
```
tensorboard --logdir <path_to_debug_model_dir>
```

Then, you can connect in a web browser to 'http://localhost:6006/' and see a graphical representation of the model's architecture.


## License
See [LICENSE](LICENSE) for full of the license text.
```
Copyright Copyright 2020 Universitat Politècnica de Catalunya & AGH University of Science and Technology

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
