
import statistics
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import process_gens
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
from process_gens import ProcessGens



from scipy import stats
import cvxpy as cp
import warnings


from typing import List

def get_text(example):
    return "".join([token.text for token in example.tokens])
def get_ids(example):
    return [token.token_id for token in example.tokens]





def analyze_completion_by_beam(processGens:ProcessGens , num_beams_list:List[int], num_instances:int):
    first_beam=next(iter(num_beams_list))
    ids=[instance_generations.instance_id for instance_generations in processGens.beam_num_to_summary[first_beam].instance_generations]
    for id in ids[:num_instances]:
        print("\n")
        for beam_num in num_beams_list:
            for instance_generation in processGens.beam_num_to_summary[beam_num].instance_generations:
                if(instance_generation.instance_id==id):
                    max_logprobs=max([example.logprob for example in instance_generation.examples])
                    # print(f"beam_num:{beam_num}\t log_prob:{instance_generation.completion_logprob}")
                    print(f"beam_num:{beam_num}  \t max_p:{instance_generation.completion_logprob}\tcompletion:{instance_generation.completion}")


def get_beam_probs(processGens:ProcessGens , num_beams_list:List[int]):
    for beam_num in num_beams_list:
        max_log_probs=[]
        for instance_generation in processGens.beam_num_to_summary[beam_num].instance_generations:
            example_logprobs=[example.logprob for example in instance_generation.examples]
            max_log_probs.append(max(example_logprobs))
        print(f"beam num: {beam_num}. \tAve:{statistics.mean(max_log_probs)}")
    
def get_beam_means(processGens:ProcessGens , num_beams_list:List[int]):
    for beam_num in num_beams_list:
        data=[]
        for instance_generation in processGens.beam_num_to_summary[beam_num].instance_generations:
            data.append(len(instance_generation.completion))
        print(f"beam num: {beam_num}. \tAve:{statistics.mean(data)}")
     
# id24769
def analyze_output_by_instance(processGens:ProcessGens , beam_num, num_instances, num_examples):
    for instance_generation in processGens.beam_num_to_summary[beam_num].instance_generations[:num_instances]:
        print(f"\n\nid: {instance_generation.instance_id}\t reference {instance_generation.reference}")
        for example in instance_generation.examples[:num_examples]:
            print(f"\tp:{example.logprob}\ttext:{example.text}")


def see_overlap_per_instance_generation(processGens:ProcessGens , beam_num, num_instances):
    for instance_generation in processGens.beam_num_to_summary[beam_num].instance_generations[:num_instances]:
        texts=[example.text for example in instance_generation.examples]
        probs=[example.logprob for example in instance_generation.examples]

        print(f"id: {instance_generation.instance_id}\tn_unique_text: {len(set(texts))}\tn:{len(set(probs))}\tn_unique_probs")



def get_instance_generations_by_id(processGens, num_beams_list:List[int]):
    ids=[instance_generations.instance_id for instance_generations in processGens.beam_num_to_summary[2].instance_generations]

    #dict id --) beam_num --) instance_generation
    instance_generations_by_id={}
    for id in ids[:10]:
        instance_generations_by_id[id]={}
        
        for beam_num in num_beams_list:
        
            for instance_generation in processGens.beam_num_to_summary[beam_num].instance_generations:
                if(instance_generation.instance_id==id):
                    instance_generations_by_id[id][beam_num]=instance_generation
    return instance_generations_by_id




def check_completion_logprob(processGens, beam_num):
    for instance_generation in processGens.beam_num_to_summary[beam_num].instance_generations:
        completion_logprob=instance_generation.completion_logprob
        for example in instance_generation.examples:
            assert example.logprob<=completion_logprob

def check_sentence_logprob(processGens,beam_num, num_instances=10, num_examples=10):
    for instance_generation in processGens.beam_num_to_summary[beam_num].instance_generations[:num_instances]:
        completion_logprob=instance_generation.completion_logprob
        for example in instance_generation.examples[:num_examples]:
            assert example.logprob<=completion_logprob



def beam_diff_check():
    instance_generations_by_id=get_instance_generations_by_id()
    for instance_idx, instance_generation_by_beam_num in instance_generations_by_id.items():
        # print(type(instance_generation_by_beam_num))

        for idx2, beam2_example in enumerate(instance_generation_by_beam_num[2].examples):
            for idx128, beam128_example in enumerate(instance_generation_by_beam_num[128].examples):
                if(beam2_example.text==beam128_example.text):
                    p1=beam2_example.logprob
                    p2=beam128_example.logprob
                    diff= abs(  (p1-p2)/ min(p1,p2) )
                    
                    print(f"Match! Beam2: {p1} vs {p2} is {diff}, {instance_idx} {idx2}, {idx128}")
                    assert(diff<0.1)

def get_dfs(processGens, num_beams_list):
    examples_df = pd.DataFrame(processGens.metrics_dicts)
    completions_df=examples_df.loc[examples_df['isCompletion'] == True]

    print(completions_df.columns)
    print(f"Num examples: {examples_df.shape[0]}")
    print(f"Num completions: {completions_df.shape[0]}")

    return examples_df, completions_df


def compare_beams_by_metric(analysis_df,compare_metric,compare_beams, compare_func= lambda a,b: b-a,plot_histogram=True):

    for beam_num in compare_beams:
        filtered_df=analysis_df.loc[analysis_df['beam_num']==beam_num]
        print(f"Mean {compare_metric} for {beam_num}:\t {filtered_df[compare_metric].mean()}")
    col_names=[f"{compare_metric}_{beam_num}" for beam_num in compare_beams]
    dif_col=f'{compare_metric}_dif'

    result = analysis_df[analysis_df['beam_num'].isin(compare_beams)][['instanceID', 'beam_num', compare_metric]]
    pivoted = result.pivot(index='instanceID', columns='beam_num', values=compare_metric)
    pivoted.columns = col_names
    pivoted = pivoted.reset_index()
    pivoted[dif_col] = pivoted.apply(lambda row: compare_func(row[col_names[0]],row[col_names[1]]), axis=1)
    print(f"Mean Change:{pivoted[dif_col].mean()}")
    print(f"Median Change:{pivoted[dif_col].median()}")
    if(plot_histogram):
        pivoted.hist(column=dif_col,bins=40)



def plot_keys(df, xlabel, ylabel, title=None, ax=None):
    if(ax is None):
        _, ax = plt.subplots()
    x=df[xlabel]
    y=df[ylabel]
    ax.scatter(x,y)
    ax.set_xlabel(xlabel)
    ax.set_xlabel(ylabel)
    if(title):
        ax.set_title(title)

    try:
        ax.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
    except:
        pass
    
    # ax.show()





def iterate_through_instances(processGens, instance_func, n_instances=10):

    instances_dict=processGens.instances_dict
    instance_stats_dict=processGens.instance_stats_dict
    return_list=[]
    for model in instances_dict.keys():        
        for task_name in instances_dict[model].keys():
            for beam_num in instances_dict[model][task_name].keys():
                for instance_id, instance_generation in instances_dict[model][task_name][beam_num].items():
                    instance_stats = instance_stats_dict[model][task_name][beam_num]
                    obj=instance_func(instance_generation=instance_generation, instance_stats=instance_stats, model=model, task_name=task_name, beam_num=beam_num)
                    return_list.append(obj)


import math






#------------------------------------ Analyze Completion Instruction------------------------------------

@dataclass(frozen=False)
class SimplifiedCompletion:
    rating:str
    completion:str
    evaluation:str

@dataclass(frozen=False)
class BeamOutputsPerInstance:
    instance_id:int
    reference:str
    prompt:str
    simplifiedCompletions:Dict[str,SimplifiedCompletion]



from helm.common.general import ensure_directory_exists, write, asdict_without_nones


def analyze_completion_by_beam(processGens:ProcessGens, num_instances:int=20):
    instances_dict=processGens.instances_dict
    

    # model, task, beam_num
    indexed_by_model={}
    counter=0
    for model_num,model in enumerate(models):
        indexed_by_task={}
        for task_num, task_name in enumerate(task_names):
            
            task_ids = list(instances_dict[0][task_num][num_beams_list[0]].keys())
            indexed_by_id={}
            for id in task_ids:
                simplifiedCompletions={}
                instance_id=None
                reference=None
                prompt=None
                for beam_num in num_beams_list:
                    instance_generation=instances_dict[model_num][task_num][beam_num][id]
                    if(counter<1):
                        # print(instance_generation.stats_dict)
                        counter+=1
                    rating = instance_generation.stats_dict["example_themis"] if instance_generation.stats_dict else -1
                    simplifiedCompletion= SimplifiedCompletion(completion=instance_generation.completion, evaluation=instance_generation.evaluation, rating=rating)
                    simplifiedCompletions[beam_num]=simplifiedCompletion
                    instance_id=instance_generation.instance_id
                    reference=instance_generation.reference
                    prompt=instance_generation.prompt
                beamOutputsPerInstance = BeamOutputsPerInstance(instance_id=instance_id, prompt=prompt, reference=reference, simplifiedCompletions=simplifiedCompletions)
                beamOutputsPerInstanceDict= asdict_without_nones(beamOutputsPerInstance)
                indexed_by_id[id]=beamOutputsPerInstanceDict
            indexed_by_task[task_name]=indexed_by_id
        indexed_by_model[model]=indexed_by_task
    return indexed_by_model


def plot_grouped(df, xlabel, ylabel, groupby='example_idx', title=None, trend_line="None",ax=None, nbins=20):
    if(ax is None):
        _, ax = plt.subplots()
    warnings.simplefilter(action='ignore', category=FutureWarning)
    if(groupby=="bins"):
        df["bins"]=pd.qcut(df[xlabel],nbins)
    
    grouped = df.groupby(groupby)[[xlabel, ylabel]].agg(['mean', 'count', 'std'])
    
    x = grouped[(xlabel, 'mean')]
    y = grouped[(ylabel, 'mean')]
    yerr = grouped[(ylabel, 'std')]

    yerr=[]
    for i in grouped.index:
        # print(grouped.loc[i][ylabel])
        _, c, s = grouped.loc[i][ylabel]
        yerr.append(1.96*s/math.sqrt(c))

    # Plot with error bars (standard deviation)
    ax.errorbar(x, y, yerr=yerr, fmt='o', ecolor='gray', capsize=3, label='Data with std dev')

    # plt.scatter(x,y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if(title):
        ax.set_title(title)

    if(trend_line=="None"):
        pass
    elif(trend_line=="linear"):
        try:
            ax.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
        except:
            pass
    else: 
        raise Exception("Plot_keys errors: did not recognize trend_line type")
    
    # plt.show()

def plot_all(dfs_by_model, compare_metric):
    for model_name, filtered_dfs in dfs_by_model.items():
        # plot by rank within sentence
        plot_grouped(df=filtered_df, xlabel="output_logprob", ylabel=compare_metric, title=f"{model_name} by rank within sentence")

        # plot 
        plot_keys(df=examples_df, xlabel='output_logprob', ylabel=compare_metric, title=f"{model_name}")

        # just plot metric / probability (normalized) 
        plot_keys(df=filtered_df, xlabel='output_logprob_norm', ylabel=compare_metric+'_norm', title=model_name)

        # plot: group into equally-sized bins (ignores examples example_id)
        plot_grouped(df=filtered_dfs, xlabel="output_logprob", groupby="bins", ylabel=compare_metric, title=f"{model_name} with equal-bins")
        plot_grouped(df=filtered_df, xlabel='output_logprob_norm',  groupby="bins", ylabel=compare_metric+'_norm', title=f"{model_name} with equal-bins")

def plot_spline(df, xlabel, ylabel, groupby='example_idx', title=None, trend_line="None",ax=None, nbins=20):
    if(ax is None):
        _, ax = plt.subplots()
    warnings.simplefilter(action='ignore', category=FutureWarning)
    if(groupby=="bins"):
        
        df["bins"]=pd.qcut(df[xlabel],nbins)
    
    grouped = df.groupby(groupby)[[xlabel, ylabel]].agg(['mean', 'count', 'std'])
    x = grouped[(xlabel, 'mean')]
    y = grouped[(ylabel, 'mean')]
    yerr = grouped[(ylabel, 'std')]

    yerr=[]
    for i in grouped.index:
        # print(grouped.loc[i][ylabel])
        _, c, s = grouped.loc[i][ylabel]
        yerr.append(1.96*s/math.sqrt(c))

    # Plot with error bars (standard deviation)
    ax.errorbar(x, y, yerr=yerr, fmt='o', ecolor='gray', capsize=3, label='Data with std dev')
    y_fit = cp.Variable(len(x))

    second_diffs = y_fit[:-2] - 2 * y_fit[1:-1] + y_fit[2:]
    objective = cp.Minimize(cp.sum_squares(y_fit - y))
    constraints = [second_diffs <= 0]  
    prob = cp.Problem(objective, constraints)
    prob.solve()


    ax.plot(x, y_fit.value, '-r', label='Concave fit (spline)')
    ax.set_title("Concave Spline Fit (Second Derivative ≤ 0)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid(True)

#returns rate that col1 is higher than col1
#if they're equal, return 0.5
def get_win_rate(row, col1:str, col2:str) -> float:
    if row[col1]==row[col2]:
        return 0.5
    return float((row[col1] - row[col2])>0)



def get_winrate_by_rank(df,compare_metric,ax=None):
    # plot_grouped(df, xlabel, ylabel, groupby='example_idx', title=None, trend_line="None",ax=None, nbins=20):
    # grouped = df.groupby("example_idx")[[xlabel, ylabel]].agg(['mean', 'count', 'std'])
    pivoted = df.pivot(columns='example_idx', index="instanceID", values=compare_metric )
    
    

    max_example_idx = examples_df["example_idx"].max()
    col1=max_example_idx
    
    x=[]
    y=[]
    for col2 in range(max_example_idx):
        x.append(col2)
        
        pivoted[f"win_rate"] = pivoted.apply( lambda row: get_win_rate(row,col2, col1) , axis=1)
        win_rate=pivoted["win_rate"].mean()
        y.append(win_rate)

        stat_rel=stats.ttest_rel(pivoted[col1], pivoted[col2]).pvalue<0.05

        print(stat_rel)
    if(ax is None):
        _, ax = plt.subplots()
    ax.scatter(x,y)
    ax.set_xlabel('example_idx')
    ax.set_ylabel(f'win rate vs idx {max_example_idx}')
