# Introduction
Provide a brief introduction to your project, including the chosen paper’s
title, authors, and publication venue.
• A brief description of the main method.
• Explain the main contribution(s) of the paper.

Hi! We'll be going over our reimplementation of generalized contextual decomposition in this document. Generalized contextual decomposition is a method of analyzing language models that was introduced in the paper Analysing Neural Language Models: Contextual Decomposition Reveals Default Reasoning in Number and Gender Assignment (Jumulet, Zuidema, and Hupkes, presented at CoNLL 2019). 

One of the central goals of neural network interpretability is analyzing how models come to their conclusions by tracking how information flows through them. Generalized contextual decomposition falls into this category, but accomplishes this much more efficiently than prior methods. While other methods require either retraining a model for each tested hypothesis or analyzing large amounts of raw data

# Chosen Result

The two results we tried to reproduce were these tables:
**![](https://lh7-us.googleusercontent.com/C3FbdrX4IRiK0-5J3oOLFBUVEa8g9ErjBTjg2mVGmrz9glJALF_xCfdJYSYuVJlJykKhFvQ40XP9RnrTmCkc_8WcI263AECgBfoxyoUzXSaUE5madV-FvXCfkWfCT7-xZKcmw_3YaJ1LDdGCddmh-_Mviw=s2048)**
The first set of tables show the average relative contributions words in the unambiguous and stereotypical dataset made towards predicting the pronouns "he" and "she." The values in the tables were obtained by subtracting the word's relative contribution to she (β_z(she) / z(she)) from its contribution to he (β_z(he) / z(he)). The results here show that for the unambiguous dataset, female pronouns contribute strongly towards predicting "she" and all other words contribute weakly towards predicting "he." For the stereotypical dataset, however, all pronouns contribute towards predicting "he," and female pronouns only contribute less strongly.

**![](https://lh7-us.googleusercontent.com/JZXZIiLud_CsDsyS6mgg_lnG72Dbz8PrB3lG8CFlrJMeYIt2klDJdp4siQ-dGVxJgaFS-5uU0v6vTr5tRJERXTFty_CbO5GtEQsYuJu0SjGzFMBUc7aY_Vmkib4gQkMwFV5R3mfSCVZCaa8it2V7QPHeQw=s2048)**
The second set of tables breaks down the contribution of various aspects of the model towards predicting "he" over "she." The columns show how the subject, object, and intercepts (or biases) contribute to the percentage chance of predicting "he" over "she."  The parentheses show the contributions when the decoder bias is removed. The results here demonstrate that the decoder and intercepts are strongly biased towards predicting "he," and that the decoder bias plays a strong role in biasing predictions towards "he" (as seen by the results in parentheses). Using two datasets also lets us see that the model is strongly biased towards predicting "he" when using stereotypical referents, regardless of their gender.

These results cover the main advantages contextual decomposition has over other methods. Separating "inside" and "outside" parts lets us identify how outputs attends to the input tokens, while also providing a theoretical framework to isolate the contribution of any arbitrary part of the model toward a prediction. By trying to replicate these two results, we can validate that contextual decomposition produces interpretable results and that it is (relatively) easy to calculate the inside and outside states for a given aspect of the model. In addition, checking that our model matches these results lets us verify that we have correctly reimplemented it. 
# Re-implementation Details

# Results and Analysis

# Conclusion and Future Work

Conclusion:
Model works, separates out effects of various parts 
Directoinally correct, might need to clean dataset?

Future work:
Generating new “hypotheses” to see whether results generalize to other tasks, datasets
Original paper only tests subject-verb agreement and gender-based anaphora resolution
Compare LSTM and transformer results to determine differences
Transformers already have attention which works analogously to the methods here

# References

Beyond Word Importance: Contextual Decomposition to Extract Interactions from LSTMs: https://arxiv.org/abs/1801.05453
Analysing Neural Language Models: Contextual Decomposition Reveals Default Reasoning in Number and Gender Assignment: https://arxiv.org/abs/1909.08975
