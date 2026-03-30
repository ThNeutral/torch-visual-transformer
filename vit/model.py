import copy
from torch import nn

from .mutliheadattention import MultiHeadAttention
from .positionwisefeedforward import PositionwiseFeedForward
from .positionalencoding import PositionalEncoding
from .encoderdecoder import EncoderDecoder
from .encoderlayer import EncoderLayer
from .encoder import Encoder
from .decoderlayer import DecoderLayer
from .decoder import Decoder
from .embeddings import Embeddings
from .generator import Generator

def make_model(
	src_vocab,
	tgt_vocab,
	N=6,
	d_model=512,
	d_ff=2048,
	h=8,
	dropout=0.1
) -> EncoderDecoder:
	c = copy.deepcopy
	attn = MultiHeadAttention(h, d_model)
	ff = PositionwiseFeedForward(d_model, d_ff, dropout)
	position = PositionalEncoding(d_model, dropout)
	model = EncoderDecoder(
		Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
		Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
		nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
		nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
		Generator(d_model, tgt_vocab)
	)

	for p in model.parameters():
		if p.dim() > 1:
			nn.init.xavier_uniform_(p)
	
	return model