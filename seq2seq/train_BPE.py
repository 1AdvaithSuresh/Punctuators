import sys
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase, NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer



def main():
    input_file = sys.argv[1]
    output_path = "./data/BPE/"
    #tokenizer model
    tokenizer = Tokenizer(BPE()) #Byte pair encoding model
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.normalizer = Sequence([ Lowercase()]) #need to use strip normalization on punctuation
    #train the tokenizer
    trainer = BpeTrainer(vocab_size=50000, show_progress=True, initial_alphabet=ByteLevel.alphabet())
    tokenizer.train(trainer, files=[input_file])
    print("Trained vocab size: {}".format(tokenizer.get_vocab_size()))
    tokenizer.model.save(output_path)
    print("Trained BPE tokenizer")

if __name__ == "__main__":
    main()


