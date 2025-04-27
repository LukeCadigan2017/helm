





#checking probabilities
# def get_strange_exampes(instanceA, instanceB):
#     diffs=[]
#     print(f"Lens: {len(instanceA.examples)} {len(instanceB.examples)}")
#     for exampleA in instanceA.examples:
#         for exampleB in instanceB.examples:
#             if(get_ids(exampleA)==get_ids(exampleB)):
#             # if(get_text(exampleA)==get_text(exampleB)):
#                 # print("Found match!")
#                 logprobA=exampleA.logprob #.logprob
#                 logprobB=exampleB.logprob #.logprob
#                 diff=abs((logprobA-logprobB)/min(logprobA,logprobB))
                
#                 diffs.append(diff)
#                 if(diff>0.1):
#                     print(f"text {get_text(exampleA)}")
#                     print(f"logprobA: {logprobA}, logprobB: {logprobB}, diff: {diff}")
#                     return exampleA,exampleB
#     return None,None

# summaries=[processGens.beam_num_to_summary[beam_num] for beam_num in [num_beams_list[0],num_beams_list[-1]] ]


# for i, instance in enumerate(summaries[1].instance_generations):
#     if(instance.instance_id=="id24769"):
#         idx=i
#         break

# instanceA=summaries[0].instance_generations[idx]
# instanceB=summaries[1].instance_generations[idx]
# exampleA,exampleB=get_strange_exampes(instanceA,instanceB)

# if exampleA is None:
#     print("No irregularities found!")
# else:
#     logprobA=exampleA.logprob #.logprob
#     logprobB=exampleB.logprob #.logprob
#     print(f"text {get_text(exampleA)}")
#     diff=abs((logprobA-logprobB)/min(logprobA,logprobB))
#     print(f"logprobA: {logprobA}, logprobB: {logprobB}, diff: {diff}")

#     tokensA=exampleA.tokens
#     tokensB=exampleB.tokens
#     for i,tokenA in enumerate(tokensA):
#         tokenA=tokensA[i]
#         tokenB=tokensB[i]
        
#         if(tokenA.text != tokenB.text):
#             print(f"{i} text is different: {tokenA.text} vs {tokenB.text}")
#         if(tokenA.token_id != tokenB.token_id):
#             print(f"{i} id is different: {tokenA.token_id} vs {tokenB.token_id}")
#         if(tokenA.logprob != tokenB.logprob):
#             print(f"{i} logprob is different {tokenA.logprob} vs {tokenB.logprob} for text {tokenA.text}")


#investigation of two different ones
#     for textA, logprobA in text_to_log_probA.items():
#         for textB, logprobB in text_to_log_probB.items():
#             if(textA==textB):
#             print("textA: ",textA)
#             diff=abs((logprobA-logprobA)/min(logprobA,logprobB))
#             print(f"logprobA: {logprobA}, logprobB: {logprobB}, diff: {diff}")
#             assert diff<0.05

#  example1:GeneratedOutput=processGens.beam_num_to_summary[2].instance_generations[0].examples[0]
# example2:GeneratedOutput=processGens.beam_num_to_summary[128].instance_generations[0].examples[0]
# token_texts=[]
# for example in [example1, example2]:

#     prob_sum = sum([token.logprob for token in example.tokens ])
#     token_text = " ".join([token.text for token in example.tokens ])
#     # print(example.keys())
#     logprob = example.logprob
#     print(f"text: {example.text}")
#     print(f"prob_sum: {prob_sum}")
#     print(f"logprob: {logprob}")
#     token_texts.append(token_text)

# assert(example1.text==example2.text)
# assert(token_texts[0]==token_texts[1])

# assert len(example1.tokens)== len(example2.tokens)
# for i in range(len(example1.tokens)):
#     token1=example1.tokens[i]
#     token2=example2.tokens[i]
#     print(f"i is {i}")
#     print(f"{token1['logprob']} vs {token2['logprob']} ")
#     percent_diff=abs((token1.logprob-token2.logprob)/token1.logprob)
#     print(f"percent_diff is {percent_diff}")
#     assert token1.text==token2.text
#     assert percent_diff<0.05

# sum1=0
# sum2=0

# sum1s=[]
# sum2s=[]
# diffs=[]
# for i in range(len(example1.tokens)):
#     sum1+=example1.tokens[i].logprob
#     sum2+=example2.tokens[i].logprob
#     # print(f"{sum1} vs {sum2}")
#     sum1s.append(sum1)
#     sum2s.append(sum1)
#     diffs.append(sum1-sum2)

# import matplotlib.pyplot as plt
# import numpy as np

# fig,ax = plt.subplots(1)
# x=np.arange(len(example1.tokens))
# ax.plot(x,diffs)
# ax.plot(x,sum2s)