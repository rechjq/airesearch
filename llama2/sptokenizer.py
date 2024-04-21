import  sentencepiece  as spm

def getTokenizerModel(modelpath):
    return spm.SentencePieceProcessor(model_file=modelpath)

if __name__ == "__main__":
    tokenizer=getTokenizerModel("tokenizer/mymodel.model")
    print(tokenizer.special_tokens['<eos>'])
