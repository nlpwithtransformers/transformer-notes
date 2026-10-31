# Scaled dot-product attention

Given a query `Q` of shape `(batch, heads, tokens, d_k)`, keys `K` and values `V` with the same
token axis, attention is:

```text
scores = Q @ K.transpose(-2, -1) / sqrt(d_k)
weights = softmax(scores, dim=-1)
output = weights @ V
```

The division by `sqrt(d_k)` keeps the logits in a range where softmax still has a useful gradient.
Without it, large `d_k` pushes most of the probability mass onto one position and the gradients
through the remaining positions vanish.

## Causal masking

A decoder adds a mask before softmax so position `t` cannot read position `> t`. Apply the mask
with a large negative value, not zero: adding zero leaves the future logits untouched and leaks
information from later tokens.

## Padding

Padding masks belong on the key axis. Masking the query axis hides a real output; masking the key
axis prevents a padded token from contributing to any query.
