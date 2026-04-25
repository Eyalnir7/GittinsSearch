import pandas
import numpy

baseline_df = pandas.read_csv("baseline.csv", sep=" ")
baseline_agg_df = baseline_df.groupby(['config','bound']).agg(['mean'])


df = pandas.read_csv("raw.csv", sep="\t")
agg_df = df.groupby(['config','N','conflict','gamma','novelty']).agg(['mean'])




Vconfig = numpy.unique(df.config.values)
VN = numpy.unique(df.N.values)
Vconflict = ["lazy", "mr", "eager"]

Vbound = numpy.unique(baseline_df.bound.values)
#Vgamma = numpy.unique(df.gamma.values)
#Vnovelty = numpy.unique(df.novelty.values)
sep="\t"
end="\n"
for config in Vconfig:
    # Title line
    print(config, "BFS", "",sep=sep,end="")
    for N in VN:
        print(N, "", sep=sep,end="")
    print("", sep=sep,end=end)

    for i, conflict in enumerate(Vconflict):
        print(conflict, "", sep=sep,end="")
        print(baseline_df[(baseline_df.config == config) & (baseline_df.bound == i)].time.mean(), "", sep=sep,end="")
        for N in VN:
            print(df[(df.config == config) & (df.N == N) & (df.conflict == conflict)].time.mean(), "", sep=sep,end="")
        print("", sep=sep,end=end)
    print("", sep=sep,end=end)

        
mean_rt_df = df[df.solved == True].groupby(['N','conflict','gamma','novelty']).agg(['mean','count'])
mean_baseline_rt_df = baseline_df[baseline_df.time <= 600].groupby(['bound']).agg(['mean','count'])

mean_rt_df.to_csv("mean_rt.csv")
mean_baseline_rt_df.to_csv("mean_baseline_rt.csv")

det_4mr_df = df[(df.N == 4) & (df.conflict == "mr")].groupby(['config']).mean()
det_baseline_df = baseline_df[baseline_df.bound == 0].groupby(['config']).mean()

det_df = det_baseline_df.join(det_4mr_df, on='config', lsuffix='baseline',rsuffix='4mr')
det_df.to_csv("det.csv")


