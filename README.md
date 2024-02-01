# Neuron-Final-Computaional-Characteristics-of-L5PC

![git_demo](https://github.com/spidermonk7/Neuron-Final-Computaional-Characteristics-of-L5PC/assets/98212025/492bb7ba-7b38-4d53-9898-f769dff46fd8)

This is the code for report.pdf, aims at discussing the computational robustness of L5b PC detailed neuron model(presented by (Hay et.al 2011)[https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1002107]). We randomly map spike trains under different distributions to a single detailed neuron(in form of synapse input), define and calculate MSE and AWE for measuring the robustness. Our experimental results indicate that: 

* a tall-PC neuron is naturally suitable for handling information with different robustness(On apical and basal dendrites).
* the L5b PC prefer to handle highly clustered spike signals(i.e. better laplacian distribution than uniform distribution)
  


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
