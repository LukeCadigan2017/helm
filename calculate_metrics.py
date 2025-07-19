


import pandas as pd
from helm.benchmark.runner import InstanceGenerations,GenerationSummary
from typing import Any, List
import json
from helm.common.request import (GeneratedOutput, Token)

import PostMetric
import pandas as pd

from helm.benchmark.metrics.statistic import Stat
from typing import Dict, Optional

from helm.benchmark.augmentations.perturbation_description import (
    PerturbationDescription)
from dataclasses import dataclass
from process_gens import *
from process_gen_utils import *
from sklearn.metrics import r2_score
from dcor import distance_correlation

from scipy.stats import linregress




class Calculate_Metrics():
    def __init__(self, df, compare_metric):

        if df is None:
            return

        unique_models=df['model'].unique()
        assert len(unique_models)==1
        self.model_name=unique_models[0]
        self.compare_metric=compare_metric


        self.pivoted = df.pivot(index="instanceID",columns="rank", values=self.compare_metric )
        self.mean_pivoted=self.pivoted.mean()
        self.median_pivoted=self.pivoted.median()
        self.compare_metric=self.compare_metric
        self.df=df

        self.grouped = df.groupby("example_idx")[["rank", self.compare_metric, "completion_length"]].mean()
        self.X=self.grouped["rank"].values.reshape(-1,1).astype(np.float64)
        self.x=self.X.reshape(-1)
        self.y=self.grouped[self.compare_metric].values.reshape(-1).astype(np.float64)


        self.metrics={}
        self.basic_metrics()
        self.stats_metrics()
        self.gam_metrics()
        self.get_length_stats()
        self.get_calculated_stats()







    def basic_metrics(self):
        self.ave_val=float(self.mean_pivoted.mean())
        self.metrics["Average Score"]=self.ave_val

        best_rank = self.mean_pivoted.idxmax()


        self.metrics["0_rank"]= float(self.mean_pivoted[1])
        self.metrics["100_rank"]= float(self.mean_pivoted[100])
        self.metrics["100_median"]=float(self.median_pivoted[100])

        self.metrics["best_rank"]=int(best_rank)
        self.metrics["best_score"]=float(self.mean_pivoted[best_rank])
        self.metrics["best_median"]=float(self.median_pivoted[100])

        self.metrics["model_name"]=self.model_name
        self.metrics["Entropy"]=float( -1.0* self.df["output_logprob"].mean())


        def get_win_rate(row, col1:str, col2:str) -> float:
            if row[col1]==row[col2]:
                return 0.5
            return float((row[col1] - row[col2])>0)


        self.pivoted["win_rate_of_best"] = self.pivoted.apply( lambda row: get_win_rate(row,best_rank, 100) , axis=1)
        win_rate=self.pivoted["win_rate_of_best"].mean()
        self.metrics["win_rate"]=win_rate

    def stats_metrics(self):
        dcor = distance_correlation(self.X, self.y)
        self.metrics["dcor"]=float(dcor)

        res=linregress(self.x, self.y)
        lin_effect= 100*res.slope
        self.metrics["PQ Slope"]=lin_effect


    def gam_metrics(self):
        # Assuming df is your dataframe

        gam = LinearGAM(s(0)).gridsearch(self.X, self.y)


        #R2
        r2 = r2_score(self.y,  gam.predict(self.X))
        self.metrics["r2"]=r2


        #find peak
        all_x=np.linspace(0,100,1000)
        all_y_pred=gam.predict(all_x)
        argmax_idx=np.argmax(all_y_pred)
        pred_peak_x=all_x[argmax_idx]
        pred_peak_y=all_y_pred[argmax_idx]
        self.metrics["Peak Rank"]=float(pred_peak_x)
        self.metrics["pred_peak_y"]=float(pred_peak_y)




        gam_ave_diff= np.mean(np.abs(all_y_pred-self.ave_val))

        self.metrics["PQ Effect"]=float(gam_ave_diff)

        #degen_integral
        num_slices=1000
        degen_x=np.linspace(pred_peak_x,100,num_slices)
        degen_y=gam.predict(degen_x)
        ave_diff=np.mean(pred_peak_y-degen_y)

        degen_intral=ave_diff*(100-pred_peak_x)/100

        self.metrics["PQ Dropoff"]=degen_intral


    def get_length_stats(self):
        length_y = self.grouped["completion_length"].values.reshape( -1)
        res=linregress(self.x, length_y)
        length_lin_effect= 100*res.slope
        self.metrics["Length Bias"]=-1*length_lin_effect
        self.metrics["length_100"]=length_y[0]
        self.metrics["length_0"]=length_y[-1]
        self.metrics["length_ave"]=np.mean(length_y)


    def get_calculated_stats(self):

        #top and bottom of distribution
        self.metrics["0_rank_diff"]=self.metrics["0_rank"] - self.metrics["Average Score"]
        self.metrics["100_rank_diff"]=self.metrics["100_rank"] - self.metrics["Average Score"]
        self.metrics["PQ Tradeoff Peak"] =self.metrics["pred_peak_y"] - self.metrics["100_rank"]


    # Probability-Quality metrics
    # gam_ave_diff          Is there a correlation between probability and quality?
    # lin_effect            Is there a positive correlation between probability and quality?
    # iom_est               How large is probability-quality tradeoff at worst?                      

    # Investigative metrics
    # ave_val               Average score of model
    # entropy               Do models with higher entropy 
    # length_lin_effect     Is there a length bias? 

    def get_x_y_cols(self):
        x_metrics=["Average Score", "Entropy", "Length Bias"]
        # y_metrics= ["PQ Effect", "PQ Slope", "PQ Tradeoff Peak", "PQ Dropoff", "Peak Rank"]
        y_metrics = ["PQ Slope", "PQ Dropoff"]

        return x_metrics, y_metrics


    def get_best_metrics(self):
        x_metrics, y_metrics=self.get_x_y_cols()
        best_metrics_cols=x_metrics+y_metrics+["model_name"]
        return_metrics={}
        for metric_col in best_metrics_cols:
            return_metrics[metric_col] = self.metrics[metric_col]
        return return_metrics
