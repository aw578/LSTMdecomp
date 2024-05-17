# Introduction

This README will cover our reimplementation of generalized contextual decomposition for our CS 4782 final project. Generalized contextual decomposition is a method of analyzing language models that was introduced in the paper Analysing Neural Language Models: Contextual Decomposition Reveals Default Reasoning in Number and Gender Assignment (Jumulet, Zuidema, and Hupkes, presented at CoNLL 2019). 

One of the central goals of neural network interpretability is analyzing how models come to their conclusions by tracking how information flows through them. Generalized contextual decomposition falls into this category, but accomplishes this much more efficiently than prior methods. While other methods are either interpretable or reusable at the cost of the other, generalized contextual decomposition achieves both qualities.

The basic idea behind generalized contextual decomposition begins with defining "inside" and "outside" inputs to each cell. This allows us to then partition the activation values into the parts caused by inside and outside values during the forward pass. By doing this, we avoid having to train a new model for each hypothesis, achieving reusability. The results also clearly show how each part of the model contribute towards the output, leading to greater interpretability. As a bonus, we can also select whatever we want to test as the "inside" input, not just individual tokens.

# Chosen Result

Although the paper we chose also examines subject-verb agreement, we chose to replicate their experiment on anaphora resolution, or choosing which gendered pronoun to output in an ambiguous sentence. For example, in the sentence "the wife loved her husband because ... ", "he" and "she" can both be the next word. We can see the relative probability of picking each pronoun, but generalized contextual decomposition lets us measure how different parts of the model contribute to those probabilities. We can also see how those contributions change depending on the gender of the subject and object, as well as whether they are unambiguously gendered (like the words king or queen) or stereotypically gendered (like the words nurse or doctor), for a total of 8 datasets. The authors chose to measure the contributions of the subject, object, and biases for each dataset, so we calculated the same values in our reimplementation.  

The two results we tried to reproduce were these tables:
**![](https://lh7-us.googleusercontent.com/C3FbdrX4IRiK0-5J3oOLFBUVEa8g9ErjBTjg2mVGmrz9glJALF_xCfdJYSYuVJlJykKhFvQ40XP9RnrTmCkc_8WcI263AECgBfoxyoUzXSaUE5madV-FvXCfkWfCT7-xZKcmw_3YaJ1LDdGCddmh-_Mviw=s2048)**

The first set of tables show how the subject and object contributed to predictions when sentences contained either unambiguously or stereotypically gendered referents. The values in the tables were obtained by subtracting the word's relative contribution to "she" (β_z(she) / z(she)) from its contribution to "he" (β_z(he) / z(he)). The results here show that for the unambiguous dataset, female referents contribute strongly towards predicting "she" and all other words contribute weakly towards predicting "he." For the stereotypical dataset, however, all referents contribute towards predicting "he," and female pronouns only contribute less strongly.

**![](https://lh7-us.googleusercontent.com/JZXZIiLud_CsDsyS6mgg_lnG72Dbz8PrB3lG8CFlrJMeYIt2klDJdp4siQ-dGVxJgaFS-5uU0v6vTr5tRJERXTFty_CbO5GtEQsYuJu0SjGzFMBUc7aY_Vmkib4gQkMwFV5R3mfSCVZCaa8it2V7QPHeQw=s2048)**

The second set of tables breaks down the contribution of various aspects of the model towards predicting "he" over "she." The columns show how the subject, object, and intercepts (or biases) contribute to the percentage chance of predicting "he" over "she."  The parentheses show the contributions when the decoder bias is removed. The results here demonstrate that the decoder and intercepts are strongly biased towards predicting "he," and that the decoder bias plays a strong role in biasing predictions towards "he" (as seen by the results in parentheses). Using two datasets also lets us see that the model is strongly biased towards predicting "he" when using stereotypical referents, regardless of their gender.

These results cover the main advantages contextual decomposition has over other methods. Separating "inside" and "outside" parts lets us identify how outputs attends to the input tokens, while also providing a theoretical framework to isolate the contribution of any arbitrary part of the model toward a prediction. By trying to replicate these two results, we can validate that contextual decomposition produces interpretable results and that it is (relatively) easy to calculate the inside and outside states for a given aspect of the model. In addition, checking that our model matches these results lets us verify that we have correctly reimplemented it. 
# Re-implementation Details

### Model Architecture

Although we covered the idea of an inside and outside state in the introduction, we need to look at how they're defined to understand the implementation of the method. The starting inside and outside state are passed in from the previous cell state, but we also need to classify the values produced by combining those states. For example, when taking the product of the input and cell gate, we'd want to add the inside candidate values to the inside cell state and vice-versa for the outside candidate values. As such, contextual decomposition requires classifying the interactions between the inside state, outside state, and bias in the LSTM.

This raises another problem: in order to analyze those interactions, we need to handle each one separately. This is straightforward for the forget gate, which is just a dot product, but the input values for the other gates are first filtered through nonlinearities, like the tanh and sigmoid functions. We can't isolate the contributions of the inside and outside states to the output values, but we can estimate them using approximate Shapley values. To do this, we calculate the value of the tanh function before and after adding the component. Since the components we're adding have no natural order, we also have to average over all orderings. This takes exponential time, but since the nonlinearities we're trying to decompose only have 3 components and the forward pass is so fast it's not a problem in practice.

**![](https://lh7-us.googleusercontent.com/DbNjDBZ8rYyu6Gns_MHCXGRk05Jz29xPg5VXULrlYF0y-g6Sl0ruHzo37EAlqCGCw5aih5xBdUjppfOA92Wsb29k7z62raU66bgUtLH73QUWVw8ZQt0PLqeXI4fPwkwRlRNGJnjE3A5ezD2aejmJsz1phg=s2048)**

After separating the states in each cell into components and calculating the approximate Shapley values, we get this classification setup:

**![](https://lh7-us.googleusercontent.com/zyV94eL3Z-aXn5Q35NqszT-86LKzwbqkspAX-xzmTAA-FmbTjx4zPvNnrwZvIrGwaSGAKONTzlTAD581Wcu8R7eQ_BOBFr_nn7ilh5dlwhJXWuYyo6oOdlNY5-DgVy2S5zXknZVVuEzC0s_9otQPnBbz1w=s2048)**

To quickly summarize the reasoning behind this setup, the tanh candidate value gate calculates the actual values that get added while the sigmoid forget gate only controls how strongly they affect the output state. As such, the classification in the tanh gates and previous hidden states are the main determinant of which state output values get added to, while the forget gate only serves as a tiebreaker when the candidate value is neither inside nor outside. Input tokens are added to the appropriate hidden state based on whether we classify them as inside or outside in the hypothesis.

### Experimental Setup
In our implementation, we used Facebook's pretrained LSTM (Gulordava et al. 2018). This was the same model the authors used to obtain the chosen results, so picking it eliminated a source of variance in our results. Pretraining also meant we didn't have to spend time training our own model from scratch.

After loading the model, we obtain the weights, then split them into input, output, cell, and forget gate weights. From there, we re-implemented the LSTM forward pass algorithm, adding the split weights and decomposition where necessary. We then added the inside and outside states and verified that the sum matched the normal forward pass states as a sanity check. Beyond that, we compared the results in the paper listed above to the ones we generated as our evaluation metric.

### Dataset Generation
Unfortunately, the authors did not publish the datasets they used in the paper, but we tried to recreate them based on what their descriptions. They used the WinoBias corpus as their dataset for stereotypical referents, but for the unambiguous referent dataset they only mention using the same template. We used ChatGPT to generate a list of sentences for each category using the WinoBias template, but since we don't have access to more information we're not sure whether our dataset matches the original. We're also unsure how the authors obtained their datasets for MM / FF pairs, as the WinoBias dataset only had MF and FM pairs and they did not elaborate further.

**![](https://lh7-us.googleusercontent.com/0gCADSM-Pa7dAbmUnxm-VS7qL7nrRy_aQFKx80-o7iJ8qxYsR81lZhtXbvQxjSO6RNZ_jd7ZMpjjVd7o609iJmJnuJKZxeGyXJ5OApjlS8dpUGWHKGJ4gZq0x6XWGtNsSVDQxgAef3fNdqgdvb_HMyRvSg=s2048)**
*Sample sentences from the FM unambiguous dataset.*

**![](https://lh7-us.googleusercontent.com/9zX7_NRCpSqUYpDDGSE2A4on163yiMXvrMnrLUwRO6XLP-KFnhR_u2rkIHjYBq0CjEJDR3rK6SSML0IyMUtkRc5vwk99wrJmYXjvNBB8K60E53T2pOv6x9izQO2KCubaAneUKNOg2I4hCNCseM4LAdCfGg=s2048)**
*Sample sentences from the MF stereotypical dataset.*

# Results and Analysis
**![](https://lh7-us.googleusercontent.com/HVpQlUz8FXM-hr5rWxKWDqoJEwFTcBBL-vbUVLTu6dtt_Gf6f6WIBWZ0OleaJ8XmCWZkULvhoCbmixf26S7uBOGJ8ztRht_JSOSvHRPIvkCi6rD5rKn83hvBnn8Kx5umepBH4q5pav5o7ZuSOb8KFOt1uw=s2048)**

The first table here displays somewhat similar results to the corresponding tables in the original paper. Although the magnitude here is higher across the board, it accurately determines that female referents predict "she" more strongly and male referents predict "he" more strongly. The second table here displays lower magnitudes than the corresponding table in the original paper, but we hypothesize this may be due to dataset issues we'll cover later.

**![](https://lh7-us.googleusercontent.com/sSCwguhHs5Cq354FlUc5DU_P9jovbWTWOr4wbXaD_F4mA1JrwOyC9fFJ18s4F15MjYJBA9_u_SbvSYsAfO8IkigaSMO6nB6ZMWG5AcvOPHFkoHwUAZDPNgRXXGyLkmGNeeVmMWHlc0rWZNlVw7Mcpxjp5g=s2048)**

The tables here also display somewhat similar results to the corresponding tables in the original paper. They accurately show roughly even probabilities for the unambiguous dataset and uniformly high male probabilities for the stereotypical dataset, largely based off of the contributions of the decoder bias and intercept. However, the exact percentages here are off again, possibly again due to dataset issues.  

Looking at the results, it's clear that our implementation of generalized contextual decomposition is able to successfully measure the contributions of individual "inside" parts towards predictions. Although they're slightly off, our implementation is at least directionally correct. The different performance may be due to the fact that the authors did not describe how they created their datasets beyond one example sentence. When we created ours using ChatGPT, we didn't correct for verbs, which might bias predictions towards the more appropriate referent. Since we don't have the original dataset, we're unsure whether they did this or not, but we hypothesize that this, along with other divergences in dataset generation, could have caused the differences. 

# Conclusion and Future Work

Key takeaways: ? 
- We did successfully implement the thing
- Dataset issues
- Whatever you want to say here

In the future, we plan to generate new hypotheses to see whether the results we obtained here generalize to other tasks, as the original paper only tests claims about subject-verb agreement and gender-based anaphora resolution. We also plan to compare the results we obtained to attention scores from transformers to see whether they assign similar scores to the same words in each sentence.

# References

Murdoch, W. J., Liu, P. J., & Yu, B. (2018). Beyond word importance: Contextual decomposition to extract interactions from lstms. arXiv preprint arXiv:1801.05453.
Jumelet, J., Zuidema, W., & Hupkes, D. (2019). Analysing neural language models: Contextual decomposition reveals default reasoning in number and gender assignment. arXiv preprint arXiv:1909.08975.
