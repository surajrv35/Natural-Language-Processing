from LanguageLoader import LanguageLoader
from RNN import RNN
import torch

en_path = 'data/europarl-v7.fr-en.en'
fr_path = 'data/europarl-v7.fr-en.fr'

max_length = 20
num_epochs = 10
num_batches = 1000
vocab_size = 15000
data = LanguageLoader(en_path, fr_path, vocab_size, max_length)

def main():
    
    rnn = RNN(data.input_size, data.output_size)

    losses = []
    for epoch in range(num_epochs):
        print("=" * 50 + ("  EPOCH %i  " % epoch) + "=" * 50)
        for i, batch in enumerate(data.sentences(num_batches)):
            input, target = batch
            #print(target)
            loss, outputs = rnn.train(input, target.copy())
            losses.append(loss)

            if i % 100 == 0:
                print("Loss at step %d: %.2f" % (i, loss))
                print("Truth: \"%s\"" % data.vec_to_sentence(target))
                print("Guess: \"%s\"\n" % data.vec_to_sentence(outputs[:-1]))
                rnn.save()
    torch.save(rnn.state_dict(), "models/baseline.module")

def translate():
    #data = LanguageLoader(en_path, fr_path, vocab_size, max_length)
    #rnn = RNN(data.input_size, data.output_size)
    model = RNN(data.input_size, data.output_size) 
    model.load_state_dict(torch.load('models/baseline.module'))
    vecs = data.sentence_to_vec("Madam  president<EOS>")
    print("in translate-- ",vecs)
    translation = model.eval(vecs)
    print("final result ",data.vec_to_sentence(translation))


main()
translate()