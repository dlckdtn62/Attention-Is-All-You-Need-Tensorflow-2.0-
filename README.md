# Attention-Is-All-You-Need-Tensorflow-2.0-
 Attention Is All You Need full code by tensorflow

Scaled_Dot_Attention
- Paper shows that Softmax(Query*Key)*Value is the way how we can find appropriate answer

- Optionally we can use Masking(different way how we can compute on BERT)
 1. Masking can be used for preventing we near on next words

- We get parameters(d_emb, d_reduced)
 1. d_emb : original dimension on input embedding
 2. d_reduced : for multi_head_attention(for parallell)
 
 Multi_Head_Attention
 - on self.sequence list we append 'Scaled_Dot_Attention'ed layer
 
 - after finishing appending we concat the result for restoring to original dimension
 
 Encoder
 - Paper shows inner-layer has dimension 4*d. So, after we got input_shape on call method we build Feed-Forward Networks as input_shape[-1]*4
 
 - After first feed forward network(ffn) then we have to restore the changed dimension to original input_shape[-1]. So we finally put the input on ffn_3 layer
 
 Decoder
 - on decoder level we have to use values which is from Encoder
 - At first we do same thing as Encoder's Multi Head Attention
 - Then, we declare context as Encoder's value put these two variable on Multi Head Attention(as [x, context, context])
 (details can be seen on paper's description picture)
 
 Transformer
 - Embedding the original dimension into d_emb by using tf.keras.layers.Embedding
 - enc_count is used for Multi_Head_Attention's reducing diemsion process
