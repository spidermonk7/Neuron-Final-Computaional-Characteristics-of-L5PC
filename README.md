# Neuron-Final-Computaional-Characteristics-of-L5PC

![image](https://github.com/spidermonk7/Neuron-Final-Computaional-Characteristics-of-L5PC/assets/98212025/6afd1819-50e8-407b-84ee-34b697b5e9db)



# Quick Start:
First of all, make sure that your environment is equipped with NEURON.
Then run this for a simple simulation:
\
```python run.py```

And also, here are some implemented arguments which could be helpful:


| args      | --help |      default |
| ----------- | ----------- | :-----------: | 
| --seed      | random seed       | 200|
| --start_time   | start time of the spike train        |150|
| --end_time  |   end time of the spike train     |600|
|  --spike_train_distribution  |  distribution of spike train: must in [normal, laplacian, uniform]      |normal|
| --spike_numbers   |     the number of spikes in each train   | 10|
|  --train_numbser  |     the number of spike trian   |50|
| --noise_distribution   |   distribution of noise: must in [normal, laplacian, uniform]  |uniform|
|  --noise_level  |   the level of the noise     |1|
|  --apic_weight  |   the weight of apical synapse     |0.1|
|  --basal_weight  |  the weight of basal synapse      |0.1|
|  --scale  |   the scale of the spike train(Only used for laplacian distribution) |75|
