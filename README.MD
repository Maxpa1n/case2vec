# Case2Vec:Joint VAE for Case Text Embedding Representation
Represent case text as a computable vector in the same space.

* train vae+selfatt model run:<br>
  python emb_train.py  
 
* train single vae model run:<br>
  python emb_single_train.py
  
* emb_config.py:<br>
  model structure | dataset | training | evauation | config file


In this paper ,we use the BOW get statistics information and TRANSFORMER get content information.Final ,we use VAE Align two types of information into the same space.
![](./imgs/content.png)
<br>
<br>
<br>
Model structure
![](./imgs/model.png)
<br>
<br>
<br>
we use vae encode case texts to vectors,and visualize. 
![](./imgs/VAEATT.png)
<br>
other results are in imgs/
<br>
<br>
<br>





