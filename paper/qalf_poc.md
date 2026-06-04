# Quantum Associative Language Fields

## Abstract

This proof of concept explores a generative language model that represents
tokens as normalized complex vectors in a finite Hilbert space, keeps dialogue
context as a density matrix, learns local language relations as complex
operators, and decodes candidate tokens with Born-style probabilities. The goal
is not to outperform transformer language models. The goal is to demonstrate a
different mathematical route that can be implemented, inspected, and scaled.

## Motivation

Most practical language models refine real-valued token vectors through
attention and feed-forward layers. QALF asks a narrower experimental question:
can a small model generate coherent replies when language is represented as a
complex associative field? In this framing, meaning is not just vector position.
It also includes phase, interference, and measurement.

## Related Work

Compositional distributional semantics and DisCoCat unify grammar reductions
with vector-space meaning, as in the framework proposed by Coecke, Sadrzadeh,
and Clark. Quantum NLP maps related diagrammatic structures to quantum circuits,
as in the near-term QNLP pipeline of Meichanetzidis and collaborators. Tensor
network methods such as matrix product states show how large quantum states can
be factorized, and Orus provides a practical introduction to those methods.
QALF borrows from these ideas but uses a directly generative software model
rather than a grammar parser, classifier, or quantum circuit.

## Model

Each token is represented by a normalized complex vector:

```text
|w> in C^d,  || |w> ||_2 = 1
```

A context window is converted into a phase-rotated superposition and normalized
into a state vector. The diagnostic density context is:

```text
rho = |c><c| / Tr(|c><c|)
```

Learned relation operators transform the context:

```text
|r> = normalize(sum_k softmax(a)_k A_k |c>)
```

Candidate token amplitudes are measured by inner product:

```text
amp(w) = <w|r>
p(w) proportional to |amp(w)|^2
```

A small count-based next-token prior is added as an associative stabilizer. It
is not a pretrained language model; it is learned only from the local seed
corpus.

The chat and evaluation commands also expose a prompt-level associative
attractor. It stores only seed-corpus prompt/reply pairs and returns a matching
reply when token overlap is high enough. This is part of the PoC's associative
memory rather than an external model; pure token-by-token decoding remains
available with `--no-attractor`.

## Implementation

The implementation is Python with PyTorch complex tensors. The CLI commands are:

```bash
conda run -n EXPLLM python -m qalf.train --data data/seed_corpus.jsonl --out runs/qalf_poc
conda run -n EXPLLM python -m qalf.eval --checkpoint runs/qalf_poc/model.pt
conda run -n EXPLLM python -m qalf.chat --checkpoint runs/qalf_poc/model.pt
conda run -n EXPLLM python -m unittest discover -s tests
```

The current shell has a CUDA-enabled PyTorch build, but CUDA runtime visibility
is not available, so the first verified run uses CPU fallback.

The verified training run used:

```text
environment: EXPLLM
device: cpu fallback
examples: 50
training windows: 1520
dimension: 64
context window: 24
relation operators: 4
epochs: 120
```

The model trained without pretrained weights, transformer layers, attention, or
recurrent gates.

## Findings

The training loss fell from `4.3771` at epoch 1 to `0.9602` at epoch 120. On
the seed corpus windows, the evaluation loss was `0.9276`, corresponding to a
perplexity of `2.5285`. Density diagnostics stayed stable:

```text
trace_mean: 1.0
purity_mean: 1.0
context_norm_mean: 0.99999994
```

The hybrid QALF mode, using prompt attractors plus Born decoding, produced the
following fixed evaluation samples:

| Prompt | Reply | Source |
| --- | --- | --- |
| What is QALF? | QALF is a small quantum associative language field. It represents words as complex states and answers by measuring possible next words. | prompt attractor |
| Explain the density context in one sentence. | Density is a normalized matrix view of the active language state, useful for measuring and diagnosing context. | prompt attractor |
| How does the model choose the next word? | It transforms the context with learned complex relation operators, compares every token state, and samples from Born style probabilities. | prompt attractor |
| What happens when the answer is uncertain? | When QALF is uncertain, the probability field is flatter. The sampled answer becomes shorter, more cautious, and less specific. | prompt attractor |
| Give a tiny example of phase in language. | The word bank can lean toward money or river. Context rotates the state so one meaning resonates more strongly. | prompt attractor |

The raw Born decoder, with the prompt attractor disabled, also reproduced some
in-domain answers under greedy decoding:

```text
Prompt: What is QALF?
Reply: Qalf is a small quantum associative language field. it represents words as complex states and answers by measuring possible next words.

Prompt: How does the model choose the next word?
Reply: It transforms the context with learned complex relation operators, compares every token state, and samples from born style probabilities.
```

However, it remained brittle for weaker or paraphrased prompts:

```text
Prompt: Give a tiny example of phase in language.
Reply: The qalf? a context.
```

This is the central empirical result of the PoC: the complex field can learn
local in-domain continuations, while the prompt-level associative attractor is
currently needed to make the system reliably usable as a small assistant.

## Larger Dataset Follow-Up

The next experiment moved from 50 self-description examples to a 2,000-example
mixed corpus prepared from TinyStories validation text plus the original QALF
seed examples. The preparer downloads the raw text, splits stories, and converts
each story into a prompt/reply continuation task:

```bash
conda run -n EXPLLM python -m qalf.prepare_text --source tinystories-valid --out data/tinystories_qalf.jsonl --max-examples 2000
```

After the sandbox GPU configuration was corrected, QALF trained successfully on
the GTX 1660 with CUDA:

```text
examples: 2000
training windows: 200000
vocabulary size after corpus filtering: 4702
dimension: 96
context window: 48
relation operators: 4
epochs: 40
device: cuda
```

Training loss fell from `3.9765` to `2.5535`. Batched evaluation over the full
mixed corpus gave loss `2.4847` and perplexity `11.9973`, a major improvement
over the smaller CPU fallback run, which reached perplexity `45.7727`. Density
diagnostics remained stable with trace mean `1.0`, purity mean `1.0`, and
context norm mean `1.0`.

The larger prompt attractor can load all 2,000 prepared examples at chat time:

```bash
conda run -n EXPLLM python -m qalf.chat --checkpoint runs/qalf_tinystories/model.pt --attractor-data data/tinystories_qalf.jsonl
```

For exact or close story prompts, this creates coherent continuation behavior.
Example:

```text
Prompt: Continue this story: Tom and lily went to the zoo with their mom. they saw many animals,
Reply: Like lions, monkeys, and birds. but their favorite was the big gorilla. he was amazing. he could swing from the trees, beat his chest, and make funny noises...
```

With the attractor disabled, raw story generation improved in corpus loss but
still shows surface rhythm rather than stable narrative planning:

```text
Prompt: Continue this story: A child found a shiny key under the old tree.
Reply: , " her the brave was very excited. he was a hard about and said,". she could...
```

The CUDA v2 result strengthens the diagnosis: scaling data and training improves
token modeling substantially, but the current rank-one Hilbert-field decoder does
not yet maintain multi-token plans on its own. The next architecture step should
add higher-order associative memory or mixed-state dynamics rather than only
adding more examples.



## QALF-Mixed Update

The next architecture replaces the rank-one context state with a mixed density
context. Instead of compressing a whole prompt into one pure vector, QALF-Mixed
builds several phase-weighted context components, measures candidate tokens under
each component, and combines those Born-style distributions with learned mixture
weights. The density diagnostic should now show `purity_mean < 1.0`, which means
the model is no longer pretending that one pure state is enough to represent the
active context.

A CUDA smoke run with four components produced `purity_mean` around `0.43-0.56`,
confirming that the density object is genuinely mixed. The DGX comparison run
should test whether this improves raw `--no-attractor` story generation, not only
training loss.

## DGX Spark Plan

The next planned run moves QALF from first-order/bigram stabilization toward
higher-order sparse associative memory. Training now builds a trigram memory
keyed by the previous two tokens. During forward passes, the model adds a
weighted top-k trigram prior to the Born decoder logits. This is still a local
associative memory learned from the dataset, not a pretrained language model.

Train and evaluation jobs can now write JSONL logs through `--log-file`, making
long DGX runs easier to inspect and share. The recommended DGX Spark run uses
dimension `192`, context window `96`, eight complex relation operators, up to
one million sampled windows, and trigram top-k `64`. The key evidence to inspect
after that run is whether raw `--no-attractor` story generation improves in
addition to loss/perplexity.

## Limitations

The first corpus is tiny and self-referential. The model can demonstrate the
mechanism but cannot contain broad world knowledge. The density matrix is
currently rank one during generation, so richer mixed-state dynamics remain an
important future step.

Raw token sampling is unstable at this scale. Temperature sampling often
produces fragments or repetitions, while greedy decoding is more readable but
less exploratory. The prompt attractor improves usability by returning coherent
seed-memory replies, but future versions should make the token-level field
strong enough that this attractor becomes optional rather than central.

## Scale-Up Path

Future work should use larger corpora, wider Hilbert spaces, more relation
operators, low-rank or tensor-network factorization, and GPU training. A larger
version could replace the dense relation bank with tensor-product factorizations
to move toward billion-parameter scale without abandoning the mathematical
concept.

Concrete next experiments:

1. Replace rank-one context with mixed density states built from multiple recent
   semantic components.
2. Factor relation operators as tensor networks or low-rank complex products.
3. Train on a broader instruction corpus and compare raw Born decoding against
   the attractor-assisted mode.
4. Restore CUDA runtime visibility and benchmark GTX 1660 training throughput.
5. Add baselines such as count-only bigram decoding and real-valued versions of
   the same architecture.
6. Add higher-order quantum associative memory, such as sparse trigram
   operators or low-rank phrase-state projectors, so raw decoding can maintain
   multi-token plans without direct prompt retrieval.

## References

- Bob Coecke, Mehrnoosh Sadrzadeh, and Stephen Clark. "Mathematical Foundations
  for a Compositional Distributional Model of Meaning."
  https://arxiv.org/abs/1003.4394
- Konstantinos Meichanetzidis, Stefano Gogioso, Giovanni de Felice, Nicolo
  Chiappori, Alexis Toumi, and Bob Coecke. "Quantum Natural Language Processing
  on Near-Term Quantum Computers." https://arxiv.org/abs/2005.04147
- Roman Orus. "A Practical Introduction to Tensor Networks: Matrix Product
  States and Projected Entangled Pair States." https://arxiv.org/abs/1306.2164
