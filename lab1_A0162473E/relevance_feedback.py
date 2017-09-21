import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
def relevance_feedback(vec_docs, vec_queries, sim, n=10):
    rf_sim = sim
    alpha = 0.8
    beta = 0.2
    print('please input the iteration you want to perform')
    iteration = int(input())
    for i in range (0,iteration):
        for j in range (0, vec_queries.shape[0]):
            # Get the k most relevant documents.
            top = np.argsort(-rf_sim[:, j])[:n]
            for k in top:
                vec_queries[j] += vec_docs[k]*alpha
            bottom = np.argsort(rf_sim[:, j])[:n] 
            # Get the k most irrelevant documents
            for k in bottom:
                vec_queries[j] -= vec_docs[k]*beta
        # recompute the similarity matrix
        rf_sim = cosine_similarity(vec_docs, vec_queries)
   
    return rf_sim


def relevance_feedback_exp(vec_docs, vec_queries, sim, tfidf_model, n=10):
    rf_sim = sim
    alpha = 0.8
    beta = 0.2
    number_of_terms = 10
    inv_dic = {v: k for k, v in tfidf_model.vocabulary_.items()}
    for j in range (0, vec_queries.shape[0]):
        new_term_set = []
        top = np.argsort(-rf_sim[:, j])[:n]
        # Get the k most relevant documents.
        for k in top:
            vec_queries[j] += vec_docs[k]*alpha
        bottom = np.argsort(rf_sim[:, j])[:n]
        # Get the k most irrelevant documents
        for k in bottom:
            vec_queries[j] -= vec_docs[k]*beta
        for k in top:
            top_term_indices = np.argsort(-vec_docs[k,:])[:number_of_terms]
            # indices of top terms
            for term_index in top_term_indices:
                new_term_set.append(inv_dic[term_index])
        vec_queries[j] += tfidf_model.transform(new_term_set)[0][:]
        
    rf_sim = cosine_similarity(vec_docs, vec_queries)
    
    return rf_sim