
import statistics
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from calculate_metrics import Calculate_Metrics

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

from pygam import LinearGAM, s
import matplotlib.pyplot as plt

def get_text(example):
    return "".join([token.text for token in example.tokens])
def get_ids(example):
    return [token.token_id for token in example.tokens]
READABLE_LABELS={"beam_num": "Beam Num", "example_themis":"Themis Score", "BLEU_4": "BLEU_4", "final_num_exact_match":"Final Num Match", "example_comet":"COMET"}


def get_model_details(model_name):


    info_dict={
        #olmo
        "allenai_OLMo_2_0425_1B_Instruct":{"size": 1, "suite":  "OLMo","model_type":"instruct", "name":"Olmo 1B Instruct"},
        "allenai_OLMo_2_0425_1B":{"size": 1, "suite":  "OLMo","model_type":"base", "name":"Olmo 1B Base"},

        "allenai_OLMo_2_1124_7B_Instruct":{"size": 7, "suite":  "OLMo","model_type":"instruct", "name":"Olmo 7B Instruct" },
        "allenai_OLMo_2_1124_7B":{"size": 7, "suite":  "OLMo","model_type":"base", "name":"Olmo 7B Base" },

        "allenai_OLMo_2_1124_13B_Instruct":{"size": 13, "suite":  "OLMo","model_type":"instruct", "name":"Olmo 13B Instruct" },
        "allenai_OLMo_2_1124_13B":{"size": 13, "suite":  "OLMo","model_type":"base", "name":"Olmo 13B Base" },

        #llama instruct
        "meta_llama_Llama_3.2_1B_Instruct":{"size": 1, "suite": "Llama","model_type":"instruct",  "name":"Llama 1B Instruct"},
        "meta_llama_Llama_3.2_1B":{"size": 1, "suite": "Llama","model_type":"base",  "name":"Llama 1B Base"},

        "meta_llama_Llama_3.1_8B_Instruct":{"size": 8, "suite": "Llama","model_type":"instruct",  "name":"Llama 8B Instruct"},
        "meta_llama_Llama_3.1_8B":{"size": 8, "suite": "Llama","model_type":"base",  "name":"Llama 8B Base"},

        
        "meta-llama/Meta-Llama-3-70B-Instruct":{"size": 70, "suite": "Llama","model_type":"instruct",  "name":"Llama 70B Instruct"},
        "allenai/OLMo-2-0325-32B-Instruct":{"size": 32, "suite": "OLMo","model_type":"base",  "name":"Olmo 32 Base"},
        
        # #compare types
        "allenai_OLMo_2_1124_7B_DPO":{"size": 7, "suite":  "OLMo","model_type":"dpo", "name":"Olmo 7B DPO" },
        "allenai_OLMo_2_1124_7B_SFT":{"size": 7, "suite":  "OLMo","model_type":"sft", "name":"Olmo 7B SFT" },

        "allenai_OLMo_2_1124_13B_DPO":{"size": 13, "suite":  "OLMo","model_type":"dpo", "name":"Olmo 13B DPO" },
        "allenai_OLMo_2_1124_13B_SFT":{"size": 13, "suite":  "OLMo","model_type":"sft", "name":"Olmo 13B SFT" },

        "allenai_OLMo_2_1124_13B_DPO":{"size": 13, "suite":  "OLMo","model_type":"dpo", "name":"Olmo 13B DPO" },
        "allenai_OLMo_2_1124_13B_SFT":{"size": 13, "suite":  "OLMo","model_type":"sft", "name":"Olmo 13B SFT" },

        "Qwen_Qwen3_0.6B":{"size": 0.6, "suite":  "Qwen","model_type":"base", "name":"Qwen 0.6B" },
        "Qwen_Qwen3_1.7B":{"size": 1.7, "suite":  "Qwen","model_type":"base", "name":"Qwen 1.7B" },
        "Qwen_Qwen3_4B":{"size": 4, "suite":  "Qwen","model_type":"base", "name":"Qwen 4B" },
        "Qwen_Qwen3_8B":{"size": 8, "suite":  "Qwen","model_type":"base", "name":"Qwen 8B" },
        "Qwen_Qwen3_32B":{"size": 32, "suite":  "Qwen","model_type":"base", "name":"Qwen 32B" },

        "meta_llama_Llama_3.1_8B_Instruct_template":{"size": 8, "suite": "Llama","model_type":"instruct",  "name":"Llama 8B Instruct Template"},
        "allenai_OLMo_2_1124_13B_Instruct_template":{"size": 13, "suite":  "OLMo","model_type":"instruct", "name":"Olmo 13B Instruct Template" },

        'fairseq_sparsemax':{"name":"Sparsemax"},
                'fairseq_softmax':{"name":"Softmax"}

        
    }
    
    for new_name, dict_name in [ ("Qwen/Qwen3-0.6B","Qwen_Qwen3_0.6B"),
        ("Qwen/Qwen3-1.7B","Qwen_Qwen3_1.7B"),
        ("Qwen/Qwen3-4B","Qwen_Qwen3_4B"),
        ("Qwen/Qwen3-8B","Qwen_Qwen3_8B"),
        ("Qwen/Qwen3-32B","Qwen_Qwen3_32B")]:
        info_dict[new_name] = info_dict[dict_name] 

    
    return info_dict[model_name]


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


def plot_grouped(df, xlabel, ylabel, groupby='example_idx', title=None, trend_line="None",ax=None, nbins=20, error_bar=True):
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
        if c>0:
            yerr.append(1.96*s/math.sqrt(c))

    # Plot with error bars (standard deviation)
    if error_bar:
        ax.errorbar(x, y, yerr=yerr, fmt='o', ecolor='gray', capsize=3, label='Data with std dev')
    else:
        ax.scatter(x,y)

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
    return ax


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



#Good resource for spline interpretation https://bookdown.org/ssjackson300/Machine-Learning-Lecture-Notes/splines.html 
#created with this:
#https://docs.scipy.org/doc/scipy/tutorial/interpolate/smoothing_splines.html
from scipy.interpolate import make_smoothing_spline
def plot_smooth_spline(df, xlabel, ylabel, groupby='example_idx', title=None, trend_line="None",ax=None, nbins=20, error_bar=False, figsize=(50,50)):
    if(ax is None):
        _, ax = plt.subplots(figsize=figsize)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    if(groupby=="bins"):
        
        df["bins"]=pd.qcut(df[xlabel],nbins)
    
    grouped = df.groupby(groupby)[[xlabel, ylabel]].agg(['mean', 'count', 'std'])
    
    grouped = grouped.sort_values(by=(xlabel, 'mean'))

    x = grouped[(xlabel, 'mean')].values
    y = grouped[(ylabel, 'mean')].values
    

    yerr = grouped[(ylabel, 'std')].values
    
    yerr=[]
    if error_bar:
        for i in grouped.index:
            # print(grouped.loc[i][ylabel])
            _, c, s = grouped.loc[i][ylabel]
            yerr.append(1.96*s/math.sqrt(c))

    # Plot with error bars (standard deviation)

    if(error_bar):
        ax.errorbar(x, y, yerr=yerr, fmt='o', ecolor='gray', capsize=3, label='Data with std dev')
    else:
        ax.scatter(x,y)
    spl = make_smoothing_spline(x, y)
    ax.plot(x, spl(x), '-.')
    return ax
    






def plot_gam(df, compare_metric, ax,color, label):

    if(ax is None):
        _, ax = plt.subplots(figsize=(10, 10))
    
    grouped = df.groupby("example_idx")[["rank", compare_metric]].mean()


    # Assuming df is your dataframe
    X = grouped["rank"].values
    y = grouped[compare_metric].values


    gam = LinearGAM(s(0))

    # Fit the model to the data
    gam.fit(X, y)

    X_pred = np.linspace(0, 100,200).reshape(-1, 1)
    y_pred = gam.predict(X_pred)

    # Plot the results
    ax.scatter(X, y, alpha=0.5, color=color, label=label)
    ax.plot(X_pred, y_pred, color=color)
    return ax




    ax.scatter(X, y, label='Data', alpha=0.5)
    ax.plot(X_pred, y_pred, label='GAM Prediction', color='red')
    return ax

def plot_spline(df, xlabel, ylabel, groupby='example_idx', title=None, trend_line="None",ax=None, nbins=20, error_bar=False):
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

    if(error_bar):
        ax.errorbar(x, y, yerr=yerr, fmt='o', ecolor='gray', capsize=3, label='Data with std dev')
    else:
        ax.scatter(x,y)

    y_fit = cp.Variable(len(x))
    second_diffs = y_fit[:-2] - 2 * y_fit[1:-1] + y_fit[2:]
    lam=1
    objective = cp.Minimize(cp.sum_squares(y_fit - y)   + lam*second_diffs^2)
    constraints = [second_diffs <= 0]  
    prob = cp.Problem(objective, constraints)
    prob.solve()


    ax.plot(x, y_fit.value, '-r', label='Concave fit (spline)')
    ax.set_title("Concave Spline Fit (Second Derivative ≤ 0)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    if(title):
        ax.set_title(title)
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

def plot_constrained_spline(df, xlabel, ylabel, groupby='example_idx', title=None, trend_line="None",ax=None, nbins=20, error_bar=False):
    if(ax is None):
        _, ax = plt.subplots(figsize=(10, 10))
    warnings.simplefilter(action='ignore', category=FutureWarning)
    if(groupby=="bins"):
        
        df["bins"]=pd.qcut(df[xlabel],nbins)
    
    grouped = df.groupby(groupby)[[xlabel, ylabel]].agg(['mean', 'count', 'std'])
    
    grouped = grouped.sort_values(by=(xlabel, 'mean'))

    x = grouped[(xlabel, 'mean')].values
    y = grouped[(ylabel, 'mean')].values
        
    n = len(x)

    # Sort x (just in case)
    sorted_idx = np.argsort(x)
    x = x[sorted_idx]
    y = y[sorted_idx]

    # Step size
    h = np.diff(x)

    # Variables: fitted y values at knots
    f = cp.Variable(n)
    second_diff = [(f[i+1] - 2*f[i] + f[i-1]) / ((h[i-1] + h[i-1]) / 2)**2 for i in range(1, n-1)]

    lambda_reg = 0 
    objective = cp.Minimize(cp.sum_squares(f - y) + lambda_reg * cp.sum_squares(cp.hstack(second_diff)))

    constraints = [d <= 0 for d in second_diff]

    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Plot
    ax.scatter(x, y)
    ax.plot(x, f.value, color='red')
    ax.legend()
    ax.set_title('Smoothing spline with non-negative derivative')



def qualitative_plots(models_array, dfs_by_model, compare_metric, figsize=None):
    

    nrows=len(models_array)
    ncols=max(len(s) for s in models_array)

    if figsize is None:
        figsize=(nrows*5, ncols*5)
        print(f"figsize is {figsize}")
    _, axes=plt.subplots(nrows=nrows, ncols= ncols, figsize=figsize)
    
    for row, suite_models in enumerate(models_array):
        for col in range(ncols):
            ax=axes[row][col]
            if col < len(suite_models):
                model_name=suite_models[col]
                print(f"model_name is {model_name}")

                filtered_df=dfs_by_model[model_name]
                def calculate_title():
                    readable_model=get_model_details(model_name)['name']
                    return f"{readable_model}: {suptitle}" if ax is None else readable_model
                # plot by rank within sentence
                suptitle="Grouped by rank within sentence"
                ax=plot_grouped(df=filtered_df, xlabel="rank",groupby='example_idx', ylabel=compare_metric, title=calculate_title(), ax=ax, error_bar=False)  
                ax.set_xlabel(None)
                ax.set_ylabel(None)
            else:
                ax.axis('off')

        
    plt.tight_layout()




def append_to_dict(cur_dict, key_list, value):
    cur_key=key_list[0]
    
    #make sure it exists
    if cur_key not in cur_dict.keys():
        cur_dict[cur_key]={}

    #append recursively if not
    if(len(key_list)>1):
        append_to_dict(cur_dict[cur_key], key_list[1:], value)
    else:
        cur_dict[cur_key]=value


def get_metrics_models_dict(dfs_by_model, compare_metric):
    metrics_dict={}
    for model, model_df in dfs_by_model.items():
            metrics=Calculate_Metrics(model_df, compare_metric).get_best_metrics()
            for metric_name, metric_value in metrics.items():
                if metric_name != "model_name":
                    append_to_dict(metrics_dict, [metric_name,model ], metric_value)   
    return metrics_dict


def create_plots(kwargs_array, graph_func, figsize=None):
    nrows=len(kwargs_array)
    ncols=max(len(s) for s in kwargs_array)

    if figsize is None:
        figsize=(nrows*5, ncols*5)
        print(f"figsize is {figsize}")
    _, axes=plt.subplots(nrows=nrows, ncols= ncols, figsize=figsize)
    
    for row, kwargs_vect in enumerate(kwargs_array):
        for col in range(ncols):
            ax=axes[row][col]
            if col < len(kwargs_vect):
                kwargs=kwargs_vect[col]
                kwargs["ax"]=ax
                ax=graph_func(**kwargs)
            else:
                ax.axis('off')
    plt.tight_layout()


def create_plots(kwargs_array, graph_func, compare_metric, figsize=None):


    #green-blue attempt
    # color_array=['#ff0015', "#ff00ea", "#9500ff", "#1500ff"]

    cmap = plt.get_cmap('seismic')
    color_array = [cmap(i) for i in [0.15, 0.35, 0.65, 0.85]]
    # color_array = [cmap(i) for i in np.linspace(0.25, 0.6, 4)]

    # =["#1f77b4",  "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    ncols=len( kwargs_array)


    if figsize is None:
        figsize=(1*5, ncols*5)
        print(f"figsize is {figsize}")
    _, axes=plt.subplots(nrows=1, ncols= ncols, figsize=figsize)
    
    for col, kwargs_vect in enumerate(kwargs_array):

        ax=axes[col] if ncols>1 else axes
        for idx, kwargs in enumerate(kwargs_vect):
            kwargs["ax"]=ax

            if "color" not in kwargs.keys():
                kwargs["color"]=color_array[idx]
            ax=graph_func(**kwargs)
        ax.legend()
        ax.set_xlabel("Rank")

    left_ax=  axes[0] if ncols>1 else axes
    left_ax.set_ylabel(READABLE_LABELS[compare_metric])
    plt.tight_layout()
    plt.xlabel="Rank"