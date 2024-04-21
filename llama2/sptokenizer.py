import  sentencepiece  as spm

def getTokenizerModel(modelpath):
    sp = spm.SentencePieceProcessor()
    return sp.Load(model_file=modelpath)