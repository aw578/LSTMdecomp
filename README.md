# Introduction

Hi! We'll be going over our reimplementation of generalized contextual decomposition in this document. Generalized contextual decomposition is a method of analyzing language models that was introduced in the paper Analysing Neural Language Models: Contextual Decomposition Reveals Default Reasoning in Number and Gender Assignment (Jumulet, Zuidema, and Hupkes, presented at CoNLL 2019). 

One of the central goals of neural network interpretability is analyzing how models come to their conclusions by tracking how information flows through them. Generalized contextual decomposition falls into this category, but accomplishes this much more efficiently than prior methods. While other methods are either interpretable or reusable at the cost of the other, generalized contextual decomposition achieves both qualities.

The basic idea behind generalized contextual decomposition begins with defining "inside" and "outside" inputs to each cell. This allows us to then partition the activation values into the parts caused by inside and outside values during the forward pass. By doing this, we avoid having to train a new model for each hypothesis, achieving reusability. The results also clearly show how each part of the model contribute towards the output, leading to greater interpretability. As a bonus, we can also select whatever we want to test as the "inside" input, not just individual tokens.

# Chosen Result

The two results we tried to reproduce were these tables:
**![](https://lh7-us.googleusercontent.com/C3FbdrX4IRiK0-5J3oOLFBUVEa8g9ErjBTjg2mVGmrz9glJALF_xCfdJYSYuVJlJykKhFvQ40XP9RnrTmCkc_8WcI263AECgBfoxyoUzXSaUE5madV-FvXCfkWfCT7-xZKcmw_3YaJ1LDdGCddmh-_Mviw=s2048)**

The first set of tables show the average relative contributions words in the unambiguous and stereotypical dataset made towards predicting the pronouns "he" and "she." The values in the tables were obtained by subtracting the word's relative contribution to she (β_z(she) / z(she)) from its contribution to he (β_z(he) / z(he)). The results here show that for the unambiguous dataset, female pronouns contribute strongly towards predicting "she" and all other words contribute weakly towards predicting "he." For the stereotypical dataset, however, all pronouns contribute towards predicting "he," and female pronouns only contribute less strongly.

**![](https://lh7-us.googleusercontent.com/JZXZIiLud_CsDsyS6mgg_lnG72Dbz8PrB3lG8CFlrJMeYIt2klDJdp4siQ-dGVxJgaFS-5uU0v6vTr5tRJERXTFty_CbO5GtEQsYuJu0SjGzFMBUc7aY_Vmkib4gQkMwFV5R3mfSCVZCaa8it2V7QPHeQw=s2048)**

The second set of tables breaks down the contribution of various aspects of the model towards predicting "he" over "she." The columns show how the subject, object, and intercepts (or biases) contribute to the percentage chance of predicting "he" over "she."  The parentheses show the contributions when the decoder bias is removed. The results here demonstrate that the decoder and intercepts are strongly biased towards predicting "he," and that the decoder bias plays a strong role in biasing predictions towards "he" (as seen by the results in parentheses). Using two datasets also lets us see that the model is strongly biased towards predicting "he" when using stereotypical referents, regardless of their gender.

These results cover the main advantages contextual decomposition has over other methods. Separating "inside" and "outside" parts lets us identify how outputs attends to the input tokens, while also providing a theoretical framework to isolate the contribution of any arbitrary part of the model toward a prediction. By trying to replicate these two results, we can validate that contextual decomposition produces interpretable results and that it is (relatively) easy to calculate the inside and outside states for a given aspect of the model. In addition, checking that our model matches these results lets us verify that we have correctly reimplemented it. 
# Re-implementation Details

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

Beyond Word Importance: Contextual Decomposition to Extract Interactions from LSTMs: https://arxiv.org/abs/1801.05453
Analysing Neural Language Models: Contextual Decomposition Reveals Default Reasoning in Number and Gender Assignment: https://arxiv.org/abs/1909.08975
