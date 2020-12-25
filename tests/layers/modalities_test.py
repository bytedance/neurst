import numpy

from neurst.layers.modalities.text_modalities import WordEmbeddingSharedWeights


def test_word_embedding():
    embedding_layer = WordEmbeddingSharedWeights(
        embedding_dim=5, vocab_size=10,
        share_softmax_weights=False)
    inputs2d = numpy.random.randint(0, 9, size=(3, 4))
    inputs1d = numpy.random.randint(0, 9, size=(3,))
    emb_for_2d = embedding_layer(inputs2d)
    emb_for_1d = embedding_layer(inputs1d)
    assert len(embedding_layer.get_weights()) == 1
    assert numpy.sum(
        (embedding_layer.get_weights()[0][inputs1d] - emb_for_1d.numpy()) ** 2) < 1e-9
    assert numpy.sum(
        (embedding_layer.get_weights()[0][inputs2d] - emb_for_2d.numpy()) ** 2) < 1e-9
    assert "emb/weights" in embedding_layer.trainable_weights[0].name

    emb_shared_layer = WordEmbeddingSharedWeights(
        embedding_dim=5, vocab_size=10,
        share_softmax_weights=True)
    emb_for_2d = emb_shared_layer(inputs2d)
    emb_for_1d = emb_shared_layer(inputs1d)
    logits_for_2d = emb_shared_layer(emb_for_2d, mode="linear")
    logits_for_1d = emb_shared_layer(emb_for_1d, mode="linear")
    assert len(emb_shared_layer.get_weights()) == 2
    for w in emb_shared_layer.trainable_weights:
        if "bias" in w.name:
            bias = w
        else:
            weights = w
    assert numpy.sum(
        (weights.numpy()[inputs1d] - emb_for_1d.numpy()) ** 2) < 1e-9
    assert numpy.sum(
        (weights.numpy()[inputs2d] - emb_for_2d.numpy()) ** 2) < 1e-9
    assert numpy.sum(
        (numpy.dot(emb_for_2d.numpy(), numpy.transpose(weights.numpy())
                   ) + bias.numpy() - logits_for_2d.numpy()) ** 2) < 1e-9
    assert numpy.sum(
        (numpy.dot(emb_for_1d.numpy(), numpy.transpose(weights.numpy())
                   ) + bias.numpy() - logits_for_1d.numpy()) ** 2) < 1e-9
    assert "shared/weights" in weights.name
    assert "shared/bias" in bias.name


if __name__ == "__main__":
    test_word_embedding()
