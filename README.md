# Neuron-Final-Computaional-Characteristics-of-L5PC

![image](https://github.com/spidermonk7/Neuron-Final-Computaional-Characteristics-of-L5PC/assets/98212025/6afd1819-50e8-407b-84ee-34b697b5e9db)



# Quick Start:
First of all, make sure that your environment is equipped with NEURON.
Then run this for a simple simulation:
\
```python run.py```

And also, here are some implemented arguments which could be helpful:


    parser.add_argument('--seed', type=int, default=200, help='random seed')
    parser.add_argument('--start_time', type=int, default=150, help='start time of the spike train')
    parser.add_argument('--end_time', type=int, default=600, help='end time of the spike train')
    parser.add_argument('--spike_train_distribution', type=str, default='normal', help='the distribution of the spike train')
    parser.add_argument('--morpho', type=int, default=1, help='the cell morpho to use')
    parser.add_argument('--train_numbers', type=int, default=50, help='the number of spike train')
    parser.add_argument('--spike_numbers', type=int, default=10, help='the number of spikes in each train')
    parser.add_argument('--noise_distribution', type=str, default='uniform', help='the distribution of the noise')
    parser.add_argument('--noise_level', type=float, default=1, help='the level of the noise')
    parser.add_argument('--apic_weight', type=float, default=0.1, help='the weight of the synapse')
    parser.add_argument('--basal_weight', type=float, default=0.1, help='the weight of the synapse')
    parser.add_argument('--scale', type=float, default=75, help='the scale of the spike train')


| args      | --help |
| ----------- | ----------- |
| --seed      | random seed       |
| --start_time   | start time of the spike train        |
| --end_time  |   end time of the spike train     |
|  --spike_train_distribution  |  distribution of spike train: must in [normal, laplacian, uniform]      |
| --noise_distribution   |   distribution of noise: must in [normal, laplacian, uniform]  |
| --spike_numbers   |     the number of spikes in each train   |
|  --train_numbser  |     the number of spike trian   |
|  --noise_level  |   the level of the noise     |
|  --apic_weight  |   the weight of apical synapse     |
|  --basal_weight  |  the weight of basal synapse      |
|  --scale  |   the scale of the spike train(Only used for laplacian distribution) |
