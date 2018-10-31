import numpy as np
from scipy import spatial
import pandas as pd
import os

def cosine_similarity(vec1, vec2):
    return 1.0 - spatial.distance.cosine(vec1, vec2)

def get_avg_vec(emb, word, k):
    tmp = []
    tmp.append(emb.wv[word])
    for closest_word, similarity in emb.most_similar(positive=word, topn=k):
        print(f"Close word to {word}: {closest_word}")
        tmp.append(emb.wv[closest_word])
    avg_vec = np.mean(tmp, axis=0)
    return avg_vec

def transform_antonym_to_axis(emb, antonym, k):
    if k == 0:
        return emb.wv[antonym[1]] - emb.wv[antonym[0]]

    else:
        vec_antonym_1 = get_avg_vec(emb, antonym[1], k)
        vec_antonym_0 = get_avg_vec(emb, antonym[0], k)

        return vec_antonym_1 - vec_antonym_0 

def project_word_on_axis(emb, word, antonym, k=10):
    return cosine_similarity(emb.wv[word], transform_antonym_to_axis(emb, antonym, k)) 


def main():
    import gensim

    EMBEDDING_PATH = "/usr0/home/mamille2/erebor/word_embeddings" ### Your path to embedding directory 

    ######################################################################
    ### Google News embedding (Download: https://code.google.com/archive/p/word2vec/). Note that for SemAxis, bin file needs to be converted to text file: see https://stackoverflow.com/questions/27324292/convert-word2vec-bin-file-to-text)
    
    #test_path = "%s/GoogleNews-vectors-negative300.txt" % (EMBEDDING_PATH)
    ######################################################################

    ######################################################################
    ## Reddit20M embedding (Download: https://drive.google.com/file/d/1ewmS5Uu4tWAkwWsuY8FZVgLr85vvZXye/view?usp=sharing) 

    #test_path = "%s/Reddit20M.cbow.300.100.txt" % (EMBEDDING_PATH)
    test_path = "%s/friends_GoogleNews_300d.txt" % (EMBEDDING_PATH)
    #test_path = "%s/detroit_GoogleNews_300d.txt" % (EMBEDDING_PATH)
    ######################################################################

    print("Loading word vectors...")
    test_embed = gensim.models.KeyedVectors.load_word2vec_format(test_path)
    print("Projecting words on axes...")
    
    axes = (
            ('them', 'us'),
            ('fake', 'real'),
            ('wrong', 'right'),
            )

    words = [
                'gay',
                'lesbian',
                'queer',
                'homosexual',
                'heterosexual',
                'transgender',
                'trans',
                'cisgender',
                'cis',
            ]

    header = [
                'embeddings',
                'term',
                'axis',
                'value',
                ]

    output = []
    outname = os.path.splitext(os.path.basename(test_path))[0]

    for w in words:

        # 3-word axes
        for axis in axes:
            outline = [outname, 
                        w, 
                        axis, 
                        project_word_on_axis(test_embed, w, axis, k=0)
                ]
            output.append(outline)
    
    ## Test results (with k=3) should be: 
    ## 0.16963434219360352 with Google News embedding
    ## 0.31472429633140564 with Reddit20M embedding

    # Save output
    out_df = pd.DataFrame(output, columns=header)
    outpath = f'/usr0/home/mamille2/erebor/fanfiction-project/output/semaxis/{outname}_semaxis_results.csv'
    out_df.to_csv(outpath, index=False)

    print(f'Results saved to {outpath}')

if __name__ == '__main__':
    main()
