- [x] Write circuit decompositions
- [x] Make heatmaps
- [ ] SVD decomposition (section 3.1)
- [ ] Bug: EQKE is 0.1x expected
- [ ] Bug: EQKP is 10x expected
- [ ] Bug: EU missing "copying" diagonal

### Circuit decomposition

We start with the QK/OV/Direct Path decompositions from Anthropic's Mathematical Transformers. Then we further decompose QK and OV into position-independent and dependent terms. This helps to lower bound error. See "mean+diff" in the paper for more details.

$$ \mathcal{M}(t) = \sigma^*\left(\underbrace{(x_{\text{query}}E + P_{\text{query}}) Q K^T (xE + P)^T / \sqrt{d}}_{\text{QK circuit}} \cdot \underbrace{(xE + P) V O U}_{\text{OV circuit}} + \underbrace{(x_{\text{query}}E + P_{\text{query}}) U}_{\text{direct path}}\right)
 $$

 Let's start with QK circuit

$$ \underbrace{(x_{\text{query}}E + P_{\text{query}})}_{} Q K^T (xE + P)^T $$

Factor out x_query

$$ x_{\text{query}}(E+x_{\text{query}}^{-1}P_{\text{query}})$$

Define a new shorthand

$$P_q = x_{\text{query}}^{-1}P_{\text{query}}$$


$$x_{query}(E+P_q)$$

Can we make it more compact?


$$ E_q = E + P_q $$

QK circuit can be rewritten

$$ x_{\text{query}} E_q Q K^T \underbrace{(xE + P)}^T $$

$$ (xE + P)^T = (xE)^T + P^T$$

$$ (xE + P)^T = E^Tx^T + P^T$$

$P_{\text{avg}}$ is the position averaged positional embedding

$$ (xE + P)^T = (E^Tx^T+P_{\text{avg}}^T) + \underbrace{(P^T-P_{\text{avg}}^T)}$$

Define $ \hat{P} = P - P_{\text{avg}} $

$$ (xE + P)^T = (\underbrace{E^Tx^T+P_{\text{avg}}^T}) + \hat{P}^T$$


$$ E^Tx^T+P_{\text{avg}}^T = (E^Tx^T{x^{T}}^{-1}+P_{\text{avg}}^T{x^{T}}^{-1})x^T $$
$$ E^Tx^T+P_{\text{avg}}^T = (E^T+P_{\text{avg}}^T{x^{T}}^{-1})x^T $$
$$ E^Tx^T+P_{\text{avg}}^T = (E+{x^{-1}}P_{\text{avg}})^Tx^T $$
$$ E^Tx^T+P_{\text{avg}}^T = (x\underbrace{(E+{x^{-1}}P_{\text{avg}})})^T $$

Define $\overline{P}={x^{-1}}P_{\text{avg}}$
And $ \overline{E} = E + \overline{P}$

$$ E^Tx^T+P_{\text{avg}}^T = (x\overline{E})^T $$

$$ (xE + P)^T = (x\overline{E})^T + \hat{P}^T$$

$$ (xE + P)^T = \overline{E}^Tx^T + \hat{P}^T$$

Going back to the QK circuit

$$ x_{\text{query}} E_q Q K^T (xE + P)^T = x_{\text{query}} E_q Q K^T (\overline{E}^Tx^T + \hat{P}^T) $$



![Heatmap grid but it's buggy](heatmap_grid.png)

